"""
head_RL.py  —  cloth crumple → UV inference → 2-arm grab/drag.

Two modes:
  default  VLA: capture → UV Mapper labels every point (u,v) → Haiku reads the
           (xyz + uv) point table and picks 2-arm grabs (in UV space) + drag.
  --rl     REINFORCE training loop driving the RLPolicy network.

Headless by default. Add --gui to open the Isaac Sim window.

    python head_RL.py            # VLA (Haiku drives 2 arms)
    python head_RL.py --rl       # RL policy training
    python head_RL.py --gui      # VLA with a window open
"""

import os, sys, argparse, subprocess, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--ball",         action="store_true")
parser.add_argument("--ball-radius",  type=float, default=0.08)
parser.add_argument("--ball-lift",    type=float, default=0.35)
parser.add_argument("--ball-offset",  type=float, default=0.0)
parser.add_argument("--drape-frames", type=int,   default=60)
parser.add_argument("--settle",       type=int,   default=100)
parser.add_argument("--substeps",     type=int,   default=10)
parser.add_argument("--iters",        type=int,   default=4)
parser.add_argument("--mass",         type=float, default=0.2)
parser.add_argument("--prad",         type=float, default=9e-3)
parser.add_argument("--config",       type=str,   default=None)
parser.add_argument("--model",        type=str,   default=None)
parser.add_argument("--infer-env",    type=str,   default="infer")
parser.add_argument("--no-capture",   dest="capture", action="store_false")
parser.add_argument("--normal-k",     type=int,   default=30)
parser.add_argument("--gui",          action="store_true", help="open Isaac Sim window (default: headless)")
# ── grab-drape-drop crumple (default) ────────────────────────────────────────
parser.add_argument("--grab-reach",      type=float, default=0.20, help="max grab offset from centroid (m)")
parser.add_argument("--grab-radius-min", type=float, default=0.03, help="min grab patch radius (m)")
parser.add_argument("--grab-radius-max", type=float, default=0.05, help="max grab patch radius (m)")
parser.add_argument("--grab-height",     type=float, default=0.50, help="hang height during drape (m)")
# ── RL training mode ──────────────────────────────────────────────────────────
parser.add_argument("--rl",           action="store_true", help="run RL training loop")
parser.add_argument("--rl-turns",     type=int,   default=4,  help="grasp turns per crumpled state")
parser.add_argument("--rl-k",         type=int,   default=6,  help="episodes between policy updates")
parser.add_argument("--rl-buffer",    type=str,   default="rl_buffer.json")
parser.add_argument("--rl-policy",    type=str,   default=None, help="RL policy checkpoint path")
parser.add_argument("--rl-settle",    type=int,   default=80,  help="settle frames after each drag")
parser.add_argument("--rl-drag-fr",   type=int,   default=60,  help="frames to execute drag motion")
parser.add_argument("--vlm-actions",  action="store_true", help="(deprecated no-op) VLA is the default mode now")
parser.add_argument("--il-dir",       type=str, default=None, help="IL dataset dir (default <root>/il_dataset); VLA mode logs (state,action) here")
parser.add_argument("--no-il",        dest="il", action="store_false", help="disable IL data logging in VLA mode")
parser.set_defaults(il=True)
args = parser.parse_args()

_ROOT  = os.path.dirname(os.path.abspath(__file__))
IL_DIR = args.il_dir or os.path.join(_ROOT, "il_dataset")

# ── 1. Isaac ─────────────────────────────────────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless":  not args.gui,
    "multi_gpu": False,
    "renderer":  "RaytracedLighting",   # RTX annotators work headless; default RealTimePathTracing does not
})

import numpy as np
import warp as wp
import newton
from newton import ParticleFlags
from newton.solvers import style3d
from isaacsim.core.api import World
from pxr import Usd, UsdGeom, UsdLux, Gf, Vt
import omni.replicator.core as rep
from scipy.spatial import cKDTree
from utils.pointcloud import furthest_point_sampling_idx
from utils.style3d_nan_patch import apply_style3d_nan_patch
import il_dataset
apply_style3d_nan_patch()   # 0/0 NaN guard for pinned-vs-pinned self-contacts

MESH = os.path.join(_ROOT, "assets", "garments", "majca2.usdc")
CAM_Z, CAM_W, CAM_H, CAM_WARMUP, N_PCD = 1.0, 640, 480, 5, 4096
CAM_FX = 24.0 / 36.0 * CAM_W
CAM_FY = 24.0 / 24.0 * CAM_H
CAM_CX, CAM_CY = CAM_W / 2.0, CAM_H / 2.0


# ── helpers ──────────────────────────────────────────────────────────────────────────────
def estimate_normals(pts, k=30):
    k = min(k, len(pts))
    _, idx = cKDTree(pts).query(pts, k=k, workers=-1)
    nbr = pts[idx]; c = nbr - nbr.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / k
    _, vecs = np.linalg.eigh(cov)
    n = vecs[:, :, 0]; n[n[:, 2] < 0] *= -1
    return n.astype(np.float32)

def load_usd_mesh(path):
    stage = Usd.Stage.Open(path)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            V = np.array(m.GetPointsAttr().Get(), dtype=np.float32)
            cnt = np.array(m.GetFaceVertexCountsAttr().Get())
            idx = np.array(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            faces, o = [], 0
            for c in cnt:
                for i in range(1, c - 1):
                    faces += [idx[o], idx[o+i], idx[o+i+1]]
                o += c
            return V, np.array(faces, dtype=np.int32)
    raise ValueError(f"no UsdGeom.Mesh in {path}")


# ── 2. Geometry ───────────────────────────────────────────────────────────────────────────
V, F = load_usd_mesh(MESH)
V *= 0.01
ext = V.max(0) - V.min(0); thin = int(ext.argmin())
order = [i for i in (0,1,2) if i != thin] + [thin]
V = V[:, order]; V[:, :2] -= V[:, :2].mean(0); V[:, 2] -= V[:, 2].min()
BALL_C = np.array([args.ball_offset, 0.0, args.ball_lift + args.ball_radius], dtype=np.float64)
_spawn = (args.ball_lift + 2.0*args.ball_radius + 0.10) if args.ball else 0.02
V[:, 2] += _spawn

_panel = np.load(os.path.join(_ROOT, "reference", "majca_panel_xatlas.npz"))
vmapping, uv_indices, uvs = _panel["vmapping"], _panel["indices"], _panel["uvs"].astype(np.float64)
tri3d = vmapping[uv_indices]
T3 = V[tri3d]
area3d = np.abs(0.5 * np.cross(T3[:,1]-T3[:,0], T3[:,2]-T3[:,0])[:,2]).sum()
Tu = uvs[uv_indices]; e1=Tu[:,1]-Tu[:,0]; e2=Tu[:,2]-Tu[:,0]
areaUV = np.abs(0.5*(e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0])).sum()
uv_scale = float(np.sqrt(area3d/areaUV)) if areaUV > 0 else 1.0
panel = (uvs * uv_scale).astype(np.float32)
F = tri3d.reshape(-1).astype(np.int32)
panel_area = float(area3d)
DENSITY = args.mass / panel_area if panel_area > 0 else 0.3

_g = np.load(os.path.join(_ROOT, "reference", "majca_mesh_graph.npz"))
PANEL_ID_ALL   = _g["node_panel"].astype(np.int32)
PANEL_UV_ALL   = _g["node_uv"].astype(np.float32)
_flat_ref      = np.load(os.path.join(_ROOT, "reference", "majca_flat_reference_uv.npz"))
FLAT_REF_PTS   = _flat_ref["points"].astype(np.float32)  # (22139, 3) flat garment at table level


# ── 3. Newton cloth ──────────────────────────────────────────────────────────────────────
builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
newton.solvers.SolverStyle3D.register_custom_attributes(builder)
style3d.add_cloth_mesh(
    builder,
    pos=wp.vec3(0,0,0), rot=wp.quat_identity(), vel=wp.vec3(0,0,0),
    vertices=V.tolist(), indices=F.tolist(),
    panel_verts=panel.tolist(), panel_indices=uv_indices.reshape(-1).tolist(),
    density=DENSITY, scale=1.0, particle_radius=args.prad,
    tri_aniso_ke=wp.vec3(2e1, 2e1, 2e0),
    edge_aniso_ke=wp.vec3(2e-5, 2e-5, 5e-6),
)
builder.add_ground_plane()
if args.ball:
    bbody = builder.add_body(xform=wp.transform(p=wp.vec3(*BALL_C), q=wp.quat_identity()),
                             is_kinematic=True, label="ball")
    bcfg = newton.ModelBuilder.ShapeConfig(); bcfg.density, bcfg.mu = 0.0, 0.5
    builder.add_shape_sphere(bbody, radius=args.ball_radius, cfg=bcfg)

model = builder.finalize()
model.soft_contact_radius = 0.35e-2; model.soft_contact_margin = 0.45e-2
model.soft_contact_ke = 5; model.soft_contact_kd = 1e-3; model.soft_contact_mu = 1
model.set_gravity((0,0,-9.81))
solver = newton.solvers.SolverStyle3D(model=model, iterations=args.iters)
solver._precompute(builder)
s0, s1 = model.state(), model.state()
_q0  = s0.particle_q.numpy().copy()
_qd0 = s0.particle_qd.numpy().copy()
_rng = np.random.default_rng()
control, contacts = model.control(), model.contacts()
sim_dt = (1.0/60.0) / args.substeps

def simulate():
    global s0, s1
    model.collide(s0, contacts)
    for _ in range(args.substeps):
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, sim_dt)
        s0, s1 = s1, s0

_graph = None
if wp.get_device().is_cuda and args.substeps % 2 == 0:
    simulate(); wp.synchronize()
    with wp.ScopedCapture() as _cap: simulate()
    _graph = _cap.graph
    print("[demo] CUDA graph captured")

# Graph replay reads particle_flags / particle_q from device memory each launch,
# so in-place assign() during the lift is picked up — no eager fallback needed
# (eager runs ~6x slower with multi-second collision spikes → GUI "freezes").
def step():
    if _graph: wp.capture_launch(_graph)
    else: simulate()

def verts(): return s0.particle_q.numpy()[:, :3].astype(np.float32)


# ── 4. Isaac scene ───────────────────────────────────────────────────────────────────────
world = World(physics_dt=1/60.0, backend="numpy")
world.scene.add_ground_plane(size=25.0, color=np.array([0.5,0.5,0.5]))
stage = world.scene.stage
UsdGeom.Imageable(stage.GetPrimAtPath("/World/groundPlane")).MakeInvisible()

vmesh = UsdGeom.Mesh.Define(stage, "/World/garment_vis")
vmesh.GetFaceVertexIndicesAttr().Set(F.tolist())
vmesh.GetFaceVertexCountsAttr().Set([3]*(len(F)//3))
vmesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
vmesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.20, 0.45, 0.85)]))

if args.ball:
    vball = UsdGeom.Sphere.Define(stage, "/World/ball_vis")
    vball.GetRadiusAttr().Set(float(args.ball_radius))
    vball.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.85, 0.25, 0.20)]))
    xb = UsdGeom.Xformable(vball.GetPrim()); xb.ClearXformOpOrder()
    xb.AddTranslateOp().Set(Gf.Vec3d(*BALL_C.tolist()))

for path, az in (("/World/SunF", 0.0), ("/World/SunB", 180.0)):
    light = UsdLux.DistantLight.Define(stage, path)
    light.CreateIntensityAttr(8000.0); light.CreateAngleAttr(0.5)
    xf = UsdGeom.Xformable(light.GetPrim()); xf.ClearXformOpOrder()
    xf.AddRotateZOp().Set(az); xf.AddRotateXOp().Set(60.0)

# overhead camera
cam_path = "/World/DepthCamera"
_cam = UsdGeom.Camera.Define(stage, cam_path)
_cam.GetFocalLengthAttr().Set(24.0)
_cam.GetHorizontalApertureAttr().Set(36.0)
_cam.GetVerticalApertureAttr().Set(24.0)
_cam.GetClippingRangeAttr().Set((0.01, 10.0))
_rp = rep.create.render_product(cam_path, (CAM_W, CAM_H))
rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb"); rgb_ann.attach(_rp)

def place_camera(cxy):
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path)); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(cxy[0]), float(cxy[1]), float(CAM_Z)))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1,0,0,0))
    return np.array([float(cxy[0]), float(cxy[1]), float(CAM_Z)], dtype=np.float64)

world.reset()
for _ in range(CAM_WARMUP):
    world.step(render=True); rep.orchestrator.step(pause_timeline=False)

def pump(i, tag=""):
    p = verts()
    if args.gui:
        vmesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(p))
    world.step(render=args.gui)
    if i % 30 == 0:
        print(f"  {tag}frame {i:3d}  z∈[{p[:,2].min():.3f},{p[:,2].max():.3f}]")


# ── helpers ───────────────────────────────────────────────────────────────────────────────
def _reset_cloth():
    """Restore cloth to initial flat state and randomize ball position."""
    for st in (s0, s1):
        q = st.particle_q.numpy();  q[:]  = _q0;  st.particle_q.assign(q)
        qd = st.particle_qd.numpy(); qd[:] = _qd0; st.particle_qd.assign(qd)
    if args.ball:
        dx = float(_rng.uniform(-0.10, 0.10))
        dy = float(_rng.uniform(-0.10, 0.10))
        bc = np.array([BALL_C[0]+dx, BALL_C[1]+dy, BALL_C[2]])
        bq = s0.body_q.numpy(); bq[bbody] = [*bc, 0,0,0,1]; s0.body_q.assign(bq)
        xb = UsdGeom.Xformable(stage.GetPrimAtPath("/World/ball_vis"))
        xb.ClearXformOpOrder(); xb.AddTranslateOp().Set(Gf.Vec3d(*bc.tolist()))


GRAB_RADIUS = 0.03   # cloth verts within this radius of the grab anchor get pinned (m)


def _find_held(pcd_idx, pcd_to_mesh):
    """Return indices + start positions of verts within GRAB_RADIUS of the predicted mesh vert.

    Uses pcd_idx → mesh vert as the anchor (exact, no projection noise) then
    collects a patch of neighbouring verts within GRAB_RADIUS.
    """
    p = verts()
    anchor = int(pcd_to_mesh[pcd_idx])
    anchor_pos = p[anchor]
    held = np.where(np.linalg.norm(p - anchor_pos, axis=1) < GRAB_RADIUS)[0]
    if len(held) == 0:
        held = np.array([anchor])
    return held, p[held].copy()


def _pin(held_all):
    """Clear ACTIVE on held verts so the solver skips integrating them."""
    flags = model.particle_flags.numpy()
    flags[held_all] = flags[held_all] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags.assign(flags)
    wp.synchronize()


def _unpin(held_all):
    flags = model.particle_flags.numpy()
    flags[held_all] = flags[held_all] | int(ParticleFlags.ACTIVE)
    model.particle_flags.assign(flags)
    wp.synchronize()


def _drive(held_groups, start_groups, offsets):
    """Write held verts to start + offset(s) into s0 BEFORE step.
    offsets may be a single (3,) array (broadcast to all groups)
    or a list of per-group (3,) arrays for independent arm targets."""
    q = s0.particle_q.numpy()
    if isinstance(offsets, np.ndarray):
        offsets = [offsets] * len(held_groups)
    for held, start, off in zip(held_groups, start_groups, offsets):
        q[held, :3] = start + off
    s0.particle_q.assign(q)
    wp.synchronize()


def _render():
    if args.gui:
        vmesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(verts()))
    world.step(render=args.gui)


def _ease(t):
    """Cosine ease-in-out, t in [0,1]."""
    return 0.5 - 0.5 * np.cos(np.pi * t)


# ── 5. RL helpers ────────────────────────────────────────────────────────────────────────────

def _capture_rl_state(tmp_npz):
    """Sample point cloud from mesh verts + save RGB. Returns pcd_to_mesh or None."""
    mesh_pts = verts()                                       # (22139, 3) exact positions
    centroid  = mesh_pts.mean(0).astype(np.float32)

    # RGB capture for VLM / visualisation
    # push live cloth into the USD vis mesh — headless _render() skips this, so without it
    # the camera renders the frozen spawn-pose mesh (the "static mesh" bug)
    vmesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh_pts))
    place_camera(mesh_pts[:, :2].mean(0))
    for _ in range(CAM_WARMUP):
        world.step(render=True); rep.orchestrator.step(pause_timeline=False)
    rraw = rgb_ann.get_data()
    rgb  = np.asarray(rraw.get("data", rraw) if isinstance(rraw, dict) else rraw)
    if rgb.ndim == 3 and rgb.shape[2] == 4: rgb = rgb[:, :, :3]
    from PIL import Image
    Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(_ROOT, "rl_capture_latest.png"))

    # point cloud: FPS directly on mesh vertices — no depth annotator needed
    fps         = furthest_point_sampling_idx(mesh_pts, n_samples=N_PCD)
    pcd_xyz     = (mesh_pts[fps] - centroid).astype(np.float32)
    pcd_to_mesh = fps.astype(np.int32)
    normals     = estimate_normals(pcd_xyz, k=args.normal_k)

    np.savez(tmp_npz,
             pcd_xyz     = pcd_xyz,
             normals     = normals,
             centroid    = centroid,
             pcd_to_mesh = pcd_to_mesh)
    return pcd_to_mesh


def _run_rl_infer(tmp_npz, tmp_json):
    """Call rl_infer.py subprocess. Returns action dict or None on failure."""
    mdl_path    = args.model  or os.path.join(_ROOT, "checkpoints", "uv_mapper_best.pth")
    policy_path = args.rl_policy or os.path.join(_ROOT, "checkpoints", "rl_policy.pth")
    cmd = ["conda", "run", "-n", args.infer_env, "--no-capture-output",
           "python", os.path.join(_ROOT, "workers", "rl_infer.py"),
           "--npz",    tmp_npz,
           "--out",    tmp_json,
           "--model",  mdl_path,
           "--policy", policy_path]
    ret = subprocess.run(cmd, cwd=_ROOT, env=os.environ)
    if ret.returncode != 0:
        print("[rl] rl_infer.py failed")
        return None
    with open(tmp_json) as fh:
        return json.load(fh)


_POST_RGB_PATH = os.path.join(_ROOT, "rl_reward_img.png")


def _save_post_rgb(path):
    """Capture overhead RGB after settle and save to path (for VLM scoring)."""
    mesh_pts = verts()
    vmesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh_pts))   # sync live cloth (headless render)
    place_camera(mesh_pts[:, :2].mean(0))
    for _ in range(CAM_WARMUP):
        world.step(render=True); rep.orchestrator.step(pause_timeline=False)
    rraw = rgb_ann.get_data()
    rgb  = np.asarray(rraw.get("data", rraw) if isinstance(rraw, dict) else rraw)
    if rgb.ndim == 3 and rgb.shape[2] == 4: rgb = rgb[:, :, :3]
    from PIL import Image
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def _execute_drag(action, pcd_to_mesh):
    """Pin grasp patch, drag to (start+dx, start+dy) at height z, release, settle."""
    held, start = _find_held(action["pcd_idx"], pcd_to_mesh)
    target_off  = np.array([action["dx"], action["dy"], action["z"]], dtype=np.float32)
    current_z   = start[:, 2].mean()
    _pin(held)

    # move to drag target over DRAG_FRAMES
    for i in range(args.rl_drag_fr):
        if not simulation_app.is_running(): break
        t   = _ease((i + 1) / args.rl_drag_fr)
        # lift z immediately to avoid table scrape, then translate xy
        off = np.array([target_off[0] * t,
                        target_off[1] * t,
                        current_z + (target_off[2] - current_z) * min(t * 3, 1.0)],
                       dtype=np.float32)
        _drive([held], [start], off)
        step(); _render()

    # hold 10 frames at target
    for _ in range(10):
        if not simulation_app.is_running(): break
        _drive([held], [start], target_off)
        step(); _render()

    _unpin(held)

    print(f"[rl] settling {args.rl_settle} frames ...")
    for i in range(args.rl_settle):
        if not simulation_app.is_running(): break
        step(); pump(i, "rl-settle ")

    _save_post_rgb(_POST_RGB_PATH)


_MIN_GRAB_DIST = 0.05  # metres


def _uv_to_pcd_idx(grab_u, grab_v, uv_pred, pcd_xyz, centroid, exclude_world_pos=None):
    """Find pcd index whose predicted UV is nearest to (grab_u, grab_v).
    If exclude_world_pos is given, enforces ≥ MIN_GRAB_DIST separation in world space."""
    dists = np.linalg.norm(uv_pred - np.array([grab_u, grab_v], dtype=np.float32), axis=1)
    for idx in np.argsort(dists):
        if exclude_world_pos is not None:
            world = pcd_xyz[idx] + centroid
            if np.linalg.norm(world - exclude_world_pos) < _MIN_GRAB_DIST:
                continue
        return int(idx)
    return int(np.argmin(dists))   # fallback: closest regardless


def _execute_drag_dual(a1, a2, pcd_to_mesh):
    """Simultaneous 2-arm drag: pin both patches, drive independently each frame."""
    held1, start1 = _find_held(a1["pcd_idx"], pcd_to_mesh)
    held2, start2 = _find_held(a2["pcd_idx"], pcd_to_mesh)
    held_all = np.concatenate([held1, held2])
    _pin(held_all)

    t1 = np.array([a1["dx"], a1["dy"], a1["dz"]], dtype=np.float32)
    t2 = np.array([a2["dx"], a2["dy"], a2["dz"]], dtype=np.float32)
    z1_cur = float(start1[:, 2].mean())
    z2_cur = float(start2[:, 2].mean())

    for i in range(args.rl_drag_fr):
        if not simulation_app.is_running(): break
        t = _ease((i + 1) / args.rl_drag_fr)
        off1 = np.array([t1[0]*t, t1[1]*t,
                         z1_cur + (t1[2] - z1_cur) * min(t*3, 1.0)], dtype=np.float32)
        off2 = np.array([t2[0]*t, t2[1]*t,
                         z2_cur + (t2[2] - z2_cur) * min(t*3, 1.0)], dtype=np.float32)
        _drive([held1, held2], [start1, start2], [off1, off2])
        step(); _render()

    for _ in range(10):
        if not simulation_app.is_running(): break
        _drive([held1, held2], [start1, start2], [t1, t2])
        step(); _render()

    _unpin(held_all)
    print(f"[rl] settling {args.rl_settle} frames ...")
    for i in range(args.rl_settle):
        if not simulation_app.is_running(): break
        step(); pump(i, "rl-settle ")

    _save_post_rgb(_POST_RGB_PATH)


def _run_uv_infer(state_npz, uv_out):
    """Run UV Mapper (no policy) → (N,2) UV predictions saved to uv_out. Returns path or None."""
    mdl_path = args.model or os.path.join(_ROOT, "checkpoints", "uv_mapper_best.pth")
    cmd = ["conda", "run", "-n", args.infer_env, "--no-capture-output",
           "python", os.path.join(_ROOT, "workers", "uv_infer.py"),
           "--npz",   state_npz,
           "--out",   uv_out,
           "--model", mdl_path]
    ret = subprocess.run(cmd, cwd=_ROOT, env=os.environ)
    if ret.returncode != 0 or not os.path.exists(uv_out):
        print("[vla] uv_infer.py failed")
        return None
    return uv_out


_UV_OVERLAY_PATH = os.path.join(_ROOT, "rl_uv_overlay.png")


def _uv_dot_color(u, v):
    return (int(u * 255), int(v * 255), 80)


def _flat_uv_panel(S):
    """Static RIGHT panel: the TRUE flat garment layout from the reference UV (PANEL_UV_ALL).
    Every mesh vert plotted at its ground-truth (u, 1-v), coloured by UV (R=u, G=v). Cached —
    it never changes, so this is the fixed canonical template Haiku compares against."""
    from PIL import Image, ImageDraw
    cache = getattr(_flat_uv_panel, "_cache", None)
    if cache is not None and cache.size[0] == S:
        return cache.copy()
    panel = Image.new("RGB", (S, S), (12, 12, 16))
    dp    = ImageDraw.Draw(panel)
    uv    = PANEL_UV_ALL                                       # (22139, 2) ground-truth UV
    for i in range(len(uv)):
        px = int(uv[i, 0] * (S - 1))
        py = int((1.0 - uv[i, 1]) * (S - 1))                  # v up
        dp.ellipse([px-1, py-1, px+1, py+1], fill=_uv_dot_color(uv[i, 0], uv[i, 1]))
    _flat_uv_panel._cache = panel
    return panel.copy()


def _render_uv_overlay(state_npz, uv_path, rgb_path, out_path=_UV_OVERLAY_PATH):
    """Two-panel image for Haiku.

    LEFT : the REAL camera render of the crumpled garment, with all N points overlaid and
           coloured by their PREDICTED UV (R=u, G=v). Shows 3D structure → WHERE to grab.
    RIGHT: the fixed TRUE flat garment layout (reference UV), coloured the same way → WHERE
           each bit of fabric belongs when flat. A dot's colour on the LEFT tells you where on
           the RIGHT template it goes. Colour code (R=u, G=v) is identical in both panels."""
    from PIL import Image, ImageDraw, ImageFont
    cam  = Image.open(rgb_path).convert("RGB")
    W, H = cam.size                                            # 640 x 480
    dcam = ImageDraw.Draw(cam)

    d        = np.load(state_npz)
    pcd_xyz  = d["pcd_xyz"].astype(np.float32)
    centroid = d["centroid"].astype(np.float32)
    uv       = np.load(uv_path).astype(np.float32)            # predicted UV for captured points
    world    = pcd_xyz + centroid
    cx, cy   = float(centroid[0]), float(centroid[1])          # camera sits above the centroid

    # LEFT: project ALL captured points into the camera image, colour by predicted UV
    dd  = np.maximum(CAM_Z - world[:, 2], 1e-4)
    upx = (world[:, 0] - cx) / dd * CAM_FX + CAM_CX
    vpx = -(world[:, 1] - cy) / dd * CAM_FY + CAM_CY
    for i in range(len(world)):
        px, py = int(upx[i]), int(vpx[i])
        if 0 <= px < W and 0 <= py < H:
            dcam.ellipse([px-1, py-1, px+1, py+1], fill=_uv_dot_color(uv[i, 0], uv[i, 1]))

    # RIGHT: fixed true flat UV template
    S   = H                                                    # square panel, 480
    uvp = _flat_uv_panel(S)
    dvp = ImageDraw.Draw(uvp)

    try:    font = ImageFont.load_default()
    except: font = None
    def _label(draw, xy, text, fill=(255, 255, 0)):
        try:    draw.text(xy, text, fill=fill, font=font, stroke_width=1, stroke_fill=(0, 0, 0))
        except TypeError:  draw.text(xy, text, fill=fill, font=font)

    for t in (0.0, 0.5, 1.0):                                  # UV axis ticks
        x = int(t * (S - 1)); _label(dvp, (min(x, S - 26), S - 13), f"u={t:.1f}")
        y = int((1.0 - t) * (S - 1)); _label(dvp, (2, min(max(y - 6, 0), S - 13)), f"v={t:.1f}")

    # compose side by side
    gap    = 12
    canvas = Image.new("RGB", (W + gap + S, H), (0, 0, 0))
    canvas.paste(cam, (0, 0))
    canvas.paste(uvp, (W + gap, 0))
    dc = ImageDraw.Draw(canvas)
    _label(dc, (6, 6),         "LEFT: overhead camera (crumpled) — colour = predicted UV (R=u,G=v)")
    _label(dc, (6, 20),        "grab raised/bunched fabric here")
    _label(dc, (W + gap + 6, 6), "RIGHT: TRUE flat UV layout (where fabric belongs)")
    canvas.save(out_path)
    return out_path


import tempfile as _tempfile
_UV_PRED_PATH = os.path.join(_tempfile.gettempdir(), "uv_pred.npy")   # last predicted UV (for IL logging)


def _run_vlm_action(state_npz):
    """UV-infer → render overlay → Haiku picks 2-arm grabs in UV space → resolve to pcd indices.
    Returns (a1, a2) dicts with pcd_idx, grab_u/v, grab_xyz, reasoning + drag; or (None, None).
    Leaves the predicted UV at _UV_PRED_PATH for the IL logger."""
    uv_out  = _UV_PRED_PATH
    tmp_out = os.path.join(_tempfile.gettempdir(), "vlm_action_out.json")

    if _run_uv_infer(state_npz, uv_out) is None:
        return None, None

    rgb_path = os.path.join(_ROOT, "rl_capture_latest.png")
    overlay  = _render_uv_overlay(state_npz, uv_out, rgb_path)
    print(f"[vla] UV overlay → {overlay}")

    cmd = ["python", os.path.join(_ROOT, "workers", "vlm_action.py"),
           "--image", overlay,
           "--out",   tmp_out]
    ret = subprocess.run(cmd, cwd=_ROOT, env=os.environ)
    if ret.returncode != 0 or not os.path.exists(tmp_out):
        print("[vla] vlm_action.py failed")
        return None, None
    with open(tmp_out) as fh:
        data = json.load(fh)

    d        = np.load(state_npz)
    pcd_xyz  = d["pcd_xyz"].astype(np.float32)
    centroid = d["centroid"].astype(np.float32)
    uv_pred  = np.load(uv_out).astype(np.float32)

    arm1_d, arm2_d = data["arm1"], data["arm2"]

    idx1   = _uv_to_pcd_idx(arm1_d["grab_u"], arm1_d["grab_v"], uv_pred, pcd_xyz, centroid)
    world1 = pcd_xyz[idx1] + centroid
    idx2   = _uv_to_pcd_idx(arm2_d["grab_u"], arm2_d["grab_v"], uv_pred, pcd_xyz, centroid,
                            exclude_world_pos=world1)

    def _arm(idx, src):
        return {"pcd_idx": idx,
                "grab_u": float(src["grab_u"]), "grab_v": float(src["grab_v"]),
                "grab_xyz": (pcd_xyz[idx] + centroid).tolist(),
                "reasoning": src.get("reasoning", ""),
                "dx": src["dx"], "dy": src["dy"], "dz": src["dz"]}

    a1 = _arm(idx1, arm1_d)
    a2 = _arm(idx2, arm2_d)
    print(f"[vla] arm1 pcd={idx1} uv=({uv_pred[idx1,0]:.2f},{uv_pred[idx1,1]:.2f})  "
          f"arm2 pcd={idx2} uv=({uv_pred[idx2,0]:.2f},{uv_pred[idx2,1]:.2f})  "
          f"sep={np.linalg.norm(pcd_xyz[idx2]-pcd_xyz[idx1]):.3f}m")
    return a1, a2


VLM_WEIGHT = 0.3


def _vlm_score(image_path):
    """Call vlm_reward.py subprocess. Returns float [0,1] or 0.0 on failure."""
    import tempfile
    out_json = os.path.join(tempfile.gettempdir(), "vlm_score.json")
    cmd = ["python", os.path.join(_ROOT, "workers", "vlm_reward.py"),
           "--image", image_path,
           "--out",   out_json]
    ret = subprocess.run(cmd, cwd=_ROOT, env=os.environ)
    if ret.returncode != 0 or not os.path.exists(out_json):
        print("[rl] vlm_reward.py failed — using score=0.0")
        return 0.0
    with open(out_json) as fh:
        data = json.load(fh)
    return float(data.get("score", 0.0))


def _flatness():
    """Cheap scalar flatness = -mean L2 of verts to the flat reference (higher = flatter).
    Logged before/after each drag so IL data can be filtered/weighted by improvement."""
    return -float(np.mean(np.linalg.norm(verts() - FLAT_REF_PTS, axis=1)))


def _compute_reward():
    """Per-vertex L2 to flat reference + z_var + spread + VLM flatness score."""
    p        = verts()                                    # (N, 3)
    pos_loss = -float(np.mean(np.linalg.norm(p - FLAT_REF_PTS, axis=1)))
    z_var    = -float(np.var(p[:, 2]))
    spread   =  float(np.std(p[:, :2]))
    vlm      = _vlm_score(_POST_RGB_PATH) if os.path.exists(_POST_RGB_PATH) else 0.0
    reward   = pos_loss + 0.1 * z_var + 0.1 * spread + VLM_WEIGHT * vlm
    print(f"[rl] reward={reward:.4f}  pos={pos_loss:.4f}  z_var={z_var:.4f}"
          f"  spread={spread:.4f}  vlm={vlm:.3f}")
    return reward


def _append_buffer(record, buf_path):
    buf = []
    if os.path.exists(buf_path):
        with open(buf_path) as fh:
            try: buf = json.load(fh)
            except json.JSONDecodeError: buf = []
    buf.append(record)
    with open(buf_path, "w") as fh:
        json.dump(buf, fh, indent=2)


def _training_step(buf_path):
    """Call rl_update.py in infer env to do one REINFORCE gradient step."""
    policy_path = args.rl_policy or os.path.join(_ROOT, "checkpoints", "rl_policy.pth")
    cmd = ["conda", "run", "-n", args.infer_env, "--no-capture-output",
           "python", os.path.join(_ROOT, "workers", "rl_update.py"),
           "--buffer", buf_path,
           "--k",      str(args.rl_k),
           "--policy", policy_path]
    print(f"[rl] running training step ...")
    subprocess.run(cmd, cwd=_ROOT)


# ── 6. Main loop ─────────────────────────────────────────────────────────────────────────
def _crumple():
    """Grab-drape-drop crumple (default) or ball drape (--ball). Assumes cloth is at rest pose."""
    if args.ball:
        print(f"[demo] draping {args.drape_frames} frames (ball)...")
        for i in range(args.drape_frames):
            if not simulation_app.is_running(): return
            step(); pump(i, "drape ")
        bq = s0.body_q.numpy(); bq[bbody] = [0,0,-5,0,0,0,1]; s0.body_q.assign(bq)
        if args.gui:
            xb2 = UsdGeom.Xformable(stage.GetPrimAtPath("/World/ball_vis"))
            xb2.ClearXformOpOrder(); xb2.AddTranslateOp().Set(Gf.Vec3d(0,0,-5))
    else:
        # pick random grab point near centroid
        centroid_xy = _q0[:, :2].mean(0)
        angle  = float(_rng.uniform(0, 2 * np.pi))
        reach  = float(_rng.uniform(0, args.grab_reach))
        gc_xy  = centroid_xy + reach * np.array([np.cos(angle), np.sin(angle)])
        grab_r = float(_rng.uniform(args.grab_radius_min, args.grab_radius_max))
        dists  = np.linalg.norm(_q0[:, :2] - gc_xy, axis=1)
        held   = np.where(dists <= grab_r)[0]
        if len(held) == 0:
            held = np.array([int(dists.argmin())])

        # lift all particles so grabbed verts sit at grab_height
        lift_dz = args.grab_height - float(_q0[held, 2].mean())
        for st in (s0, s1):
            q = st.particle_q.numpy(); q[:, 2] += lift_dz; st.particle_q.assign(q)

        # freeze grabbed verts
        flags = model.particle_flags.numpy()
        flags[held] &= ~int(ParticleFlags.ACTIVE)
        model.particle_flags.assign(flags)
        wp.synchronize()

        print(f"[demo] grab {len(held)} verts @ r={grab_r:.3f}m h={args.grab_height:.2f}m, "
              f"draping {args.drape_frames} frames...")
        for i in range(args.drape_frames):
            if not simulation_app.is_running(): return
            step(); pump(i, "drape ")

        # release
        flags = model.particle_flags.numpy()
        flags[held] |= int(ParticleFlags.ACTIVE)
        model.particle_flags.assign(flags)
        wp.synchronize()

    print(f"[demo] settling {args.settle} frames ...")
    for i in range(args.settle):
        if not simulation_app.is_running(): return
        step(); pump(i, "settle ")


# ── RL training loop (--rl flag) ─────────────────────────────────────────────────────────
if args.rl and simulation_app.is_running():
    buf_path = args.rl_buffer
    episode  = 0
    try:
        while simulation_app.is_running():
            episode += 1
            print(f"\n[rl] ══ episode {episode} ══════════════════════════════")
            _reset_cloth()
            _crumple()

            for turn in range(args.rl_turns):
                if not simulation_app.is_running():
                    break
                print(f"[rl] turn {turn+1}/{args.rl_turns}")

                tmp_npz  = f"/tmp/rl_ep{episode}_t{turn}.npz"
                tmp_json = f"/tmp/rl_action_ep{episode}_t{turn}.json"

                pcd_to_mesh = _capture_rl_state(tmp_npz)
                if pcd_to_mesh is None:
                    break

                action = _run_rl_infer(tmp_npz, tmp_json)
                if action is None:
                    os.unlink(tmp_npz)
                    break
                os.unlink(tmp_json)

                _execute_drag(action, pcd_to_mesh)

                reward = _compute_reward()
                print(f"[rl] ep {episode} turn {turn+1}  reward={reward:.4f}"
                      f"  (pos_loss component: "
                      f"{-float(np.mean(np.linalg.norm(verts() - FLAT_REF_PTS, axis=1))):.4f})")

                _append_buffer({
                    "state_npz":  tmp_npz,
                    "action_idx": action["pcd_idx"],
                    "raw_dx":     action["raw_dx"],
                    "raw_dy":     action["raw_dy"],
                    "raw_z":      action["raw_z"],
                    "dx":         action["dx"],
                    "dy":         action["dy"],
                    "z":          action["z"],
                    "log_prob":   action["log_prob"],
                    "reward":     reward,
                    "episode":    episode,
                    "turn":       turn,
                }, buf_path)

            if episode % args.rl_k == 0:
                _training_step(buf_path)

    except KeyboardInterrupt:
        pass

# ── VLA loop (default — Haiku drives 2 arms from UV+xyz state) ─────────────────────────────
if not args.rl and simulation_app.is_running():
    episode = 0
    try:
        while simulation_app.is_running():
            episode += 1
            print(f"\n[vla] ══ episode {episode} ══════════════════════════════")
            _reset_cloth()
            _crumple()

            for turn in range(args.rl_turns):
                if not simulation_app.is_running():
                    break
                print(f"[vla] turn {turn+1}/{args.rl_turns}")

                tmp_npz = f"/tmp/vla_ep{episode}_t{turn}.npz"

                pcd_to_mesh = _capture_rl_state(tmp_npz)
                if pcd_to_mesh is None:
                    break

                a1, a2 = _run_vlm_action(tmp_npz)
                if a1 is None:
                    print("[vla] vlm_action failed — skipping turn")
                    break

                flat_before = _flatness()
                _execute_drag_dual(a1, a2, pcd_to_mesh)
                flat_after  = _flatness()

                reward = _compute_reward()
                print(f"[vla] ep {episode} turn {turn+1}  reward={reward:.4f}"
                      f"  flat {flat_before:.4f}→{flat_after:.4f}")

                # ── log IL sample (state arrays + Haiku action) ─────────────────────────────
                if args.il:
                    try:
                        d  = np.load(tmp_npz)
                        uv = np.load(_UV_PRED_PATH)
                        state  = {"pcd_xyz": d["pcd_xyz"], "normals": d["normals"],
                                  "uv_pred": uv,           "centroid": d["centroid"]}
                        action = {
                            "arm1": il_dataset.make_arm(a1["grab_u"], a1["grab_v"],
                                        a1["dx"], a1["dy"], a1["dz"],
                                        grab_xyz=a1["grab_xyz"], grab_pcd_idx=a1["pcd_idx"],
                                        reasoning=a1.get("reasoning", "")),
                            "arm2": il_dataset.make_arm(a2["grab_u"], a2["grab_v"],
                                        a2["dx"], a2["dy"], a2["dz"],
                                        grab_xyz=a2["grab_xyz"], grab_pcd_idx=a2["pcd_idx"],
                                        reasoning=a2.get("reasoning", "")),
                        }
                        sid = il_dataset.record_sample(
                            IL_DIR, "haiku", state, action,
                            episode=episode, turn=turn, reward=reward,
                            reward_before=flat_before, reward_after=flat_after,
                            teacher_model="claude-haiku-4-5")
                        print(f"[il] logged {sid}  ({il_dataset.count(IL_DIR)} samples total)")
                    except Exception as e:
                        print(f"[il] WARNING: failed to log sample: {e}")

                # clean up per-turn npz (state is already copied into the IL store)
                try: os.unlink(tmp_npz)
                except: pass

    except KeyboardInterrupt:
        pass

simulation_app.close()
