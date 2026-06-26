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
parser.add_argument("--drape-frames", type=int,   default=40)
parser.add_argument("--settle",       type=int,   default=90)
parser.add_argument("--substeps",     type=int,   default=10)
parser.add_argument("--iters",        type=int,   default=4)
# ── garment material (calibrated Style3D real-fabric scale — ported from data_gen.py) ────────
parser.add_argument("--mass",     type=float, default=0.6,  help="TOTAL garment mass (kg) → density = mass/panel_area")
parser.add_argument("--density",  type=float, default=None, help="override fabric mass per area (kg/m^2); else from --mass")
parser.add_argument("--stretch-weft",  type=float, default=1.0e2, help="tri_aniso_ke weft (Style3D real-fabric scale)")
parser.add_argument("--stretch-warp",  type=float, default=1.0e2, help="tri_aniso_ke warp")
parser.add_argument("--stretch-shear", type=float, default=1.0e1, help="tri_aniso_ke shear (soft → natural drape)")
parser.add_argument("--damping",  type=float, default=None, help="tri_kd override (default: builder's 10.0)")
parser.add_argument("--bend-weft",  type=float, default=0.5e-5, help="edge_aniso_ke weft (Style3D units; LOW = fabric)")
parser.add_argument("--bend-warp",  type=float, default=0.5e-5, help="edge_aniso_ke warp")
parser.add_argument("--bend-shear", type=float, default=0.6e-6, help="edge_aniso_ke shear")
parser.add_argument("--bend-spec",  action="store_true", help="force literal spec bend 8000 into edge_aniso_ke (WARNING: rigid)")
parser.add_argument("--prad",       type=float, default=5.0e-3, help="particle radius (m) — body/ground contact only (NOT self-contact)")
parser.add_argument("--self-thick", type=float, default=0.7e-2, help="cloth self-contact radius (m); folded layers rest ~2x apart")
parser.add_argument("--self-stiff", type=float, default=0.8,    help="self-contact layer-separation spring stiffness multiplier")
parser.add_argument("--config",       type=str,   default=None)
parser.add_argument("--model",        type=str,   default=None)
parser.add_argument("--infer-env",    type=str,   default="infer")
parser.add_argument("--no-capture",   dest="capture", action="store_false")
parser.add_argument("--normal-k",     type=int,   default=30)
parser.add_argument("--gui",          action="store_true", help="open Isaac Sim window (default: headless)")
# ── grab-drape-drop crumple (default) ────────────────────────────────────────
parser.add_argument("--grab-reach",      type=float, default=0.35, help="max grab offset from centroid (m)")
parser.add_argument("--arm-box-half",    type=float, default=0.6,  help="half-width (m) of the FIXED world workspace box the arm may not leave: world XY clamped to [center ± this]. Anchored at --arm-box-cx/cy (the cloth spawn), it does NOT follow the cloth")
parser.add_argument("--arm-box-cx",      type=float, default=0.0,  help="world X centre of the workspace box (cloth spawn X)")
parser.add_argument("--arm-box-cy",      type=float, default=0.0,  help="world Y centre of the workspace box (cloth spawn Y)")
parser.add_argument("--arm-box-zmax",    type=float, default=0.5,  help="max height (m) above the table a waypoint may reach")
parser.add_argument("--recovery-step",   type=float, default=0.3,  help="when the policy's grab falls OUTSIDE the workspace box, the fallback grabs the nearest in-box cloth point and drags it this far (m) toward the box centre to pull the garment back into reach")
parser.add_argument("--max-drag-len",    type=float, default=0.8,  help="cap on a single drag's total path length (m). A random/aggressive multi-waypoint path is scaled down toward the grab so it can't stretch the cloth past what the solver handles (blow-up → black screen). Real arms can't yank that far either")
parser.add_argument("--grab-radius-min", type=float, default=0.01, help="min grab patch radius (m)")
parser.add_argument("--grab-radius-max", type=float, default=0.03, help="max grab patch radius (m)")
parser.add_argument("--grab-radius",     type=float, default=0.008, help="grasp PINCH radius (m): cloth verts within this of the grab point get pinned. ~0.008 ≈ robot fingertip pinch (~15 verts); 0.03 = broad hand grab (~200). Applies to manual + RL/VLA execution")
parser.add_argument("--grab-height",     type=float, default=0.50, help="hang height during drape (m)")
# ── RL training mode ──────────────────────────────────────────────────────────
parser.add_argument("--rl",           action="store_true", help="run RL training loop")
parser.add_argument("--rl-turns",     type=int,   default=4,  help="grasp turns per crumpled state")
parser.add_argument("--rl-k",         type=int,   default=6,  help="episodes between policy updates")
parser.add_argument("--rl-buffer",    type=str,   default="rl_buffer.json")
parser.add_argument("--rl-policy",    type=str,   default=None, help="RL policy checkpoint path")
parser.add_argument("--rl-student",   action="store_true", help="RL-train the StudentVLA trajectory policy via MULTI-STEP PPO+GAE: run whole smoothing SEQUENCES, reward each grab by its flatness improvement, and credit it with the discounted future (so a setup move that looks bad now is rewarded for what it unlocks). BC-init it first with workers/train_student.py")
parser.add_argument("--student-buffer", type=str, default="student_buffer.json", help="(--rl-student) PPO rollout buffer (one entry per grab, tagged with its trajectory)")
parser.add_argument("--student-policy", type=str, default=None, help="(--rl-student) StudentVLA checkpoint (default checkpoints/student_vla.pth)")
parser.add_argument("--rl-group-k",     type=int, default=4, help="(--rl-student) independent SEQUENCES branched from each crumpled state (best-of-N diversity from one start; the critic is the baseline). Cost is K×turns sim per state")
parser.add_argument("--rl-gamma",      type=float, default=0.0, help="(--rl-student) discount γ. DEFAULT 0 = SINGLE-STEP: each grab is credited only by its own immediate ΔΦ (no cross-turn credit) — the right setting for smoothing (near-greedy) and the pre-IL curriculum. Set 0.97 for MULTI-STEP credit assignment once BC/IL is baked in")
parser.add_argument("--rl-lambda",     type=float, default=0.95, help="(--rl-student) GAE λ — bias/variance trade-off of the advantage")
parser.add_argument("--rl-clip",       type=float, default=0.2,  help="(--rl-student) PPO clip ε")
parser.add_argument("--rl-epochs",     type=int,   default=4,    help="(--rl-student) PPO gradient epochs per update batch")
parser.add_argument("--rl-ent-weight", type=float, default=1e-3, help="(--rl-student) entropy bonus weight (exploration)")
parser.add_argument("--rl-vf-weight",  type=float, default=0.5,  help="(--rl-student) critic (value) loss weight")
parser.add_argument("--rl-det-critic", action="store_true", help="(--rl-student) use the EXACT potential-based critic V(s)=phi_target-Φ(s) from the stored 'phi' instead of the learned value head (which is random with no IL value-pretraining). Use for the from-scratch RL sanity run; switch off once you've BC/value-pretrained on demos")
parser.add_argument("--no-rot",        action="store_true", help="FREEZE wrist rotation: pure-translation drags, rotation excluded from the PPO objective. For the position-only sanity pass before rotation is trusted. Without it, rotation is a bounded swing-twist RL action (≤30° off vertical, free roll) learned alongside position")
parser.add_argument("--student-eval",   action="store_true", help="EVAL the StudentVLA: load the checkpoint, run GREEDY (deterministic) actions and watch — no sampling, no learning, no logging. Add --gui to see it live in Isaac; saves rl_flat_overlay.png each step")
parser.add_argument("--rl-settle",    type=int,   default=80,  help="settle frames after each drag")
parser.add_argument("--rl-drag-fr",   type=int,   default=60,  help="(RL mode) fixed frames to execute drag motion")
parser.add_argument("--drag-speed",   type=float, default=0.12, help="(VLA / --scripted) drag speed in m/s; frames derived from path length so speed is constant (slow → cloth doesn't tear)")
parser.add_argument("--vlm-actions",  action="store_true", help="(deprecated no-op) VLA is the default mode now")
parser.add_argument("--il-dir",       type=str, default=None, help="IL dataset dir (default <root>/il_dataset); VLA mode logs (state,action) here")
parser.add_argument("--no-il",        dest="il", action="store_false", help="disable IL data logging in VLA mode")
parser.add_argument("--manual",       action="store_true", help="interactive: click+drag the garment with the mouse (overhead view); scroll = raise/lower grab Z, r = re-crumple, q = quit")
parser.add_argument("--manual-z-step", type=float, default=0.02, help="(--manual) metres Z changes per mouse-wheel notch")
parser.add_argument("--scripted",     action="store_true", help="deterministic UV teacher: replay grasp_regions.json in priority order (grab UV patch → drag to its flat target), logging IL samples (source=scripted). No VLM, no learning. Add --gui to watch.")
parser.add_argument("--regions",      type=str, default=None, help="(--scripted) region config (default reference/grasp_regions.json from data/label_grasp_regions.py)")
parser.add_argument("--teacher-tol",     type=float, default=0.05, help="(--scripted) region 'done' tolerance (m): once the patch sits within this of its flat target, advance to the next region")
parser.add_argument("--teacher-retries", type=int,   default=2,    help="(--scripted) max grab attempts per region before moving on")
parser.add_argument("--teacher-lift",    type=float, default=0.22, help="(--scripted) apex height (m) of the lift-carry-place arc")
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
DENSITY = args.density if args.density is not None else (args.mass / panel_area if panel_area > 0 else 0.3)
if args.bend_spec:                               # user insists on the literal spec value
    args.bend_weft = args.bend_warp = args.bend_shear = 8.0e3

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
    tri_aniso_ke=wp.vec3(args.stretch_weft, args.stretch_warp, args.stretch_shear),
    edge_aniso_ke=wp.vec3(args.bend_weft, args.bend_warp, args.bend_shear),
    **({"tri_kd": args.damping} if args.damping is not None else {}),  # else builder default 10.0
)
builder.add_ground_plane()
if args.ball:
    bbody = builder.add_body(xform=wp.transform(p=wp.vec3(*BALL_C), q=wp.quat_identity()),
                             is_kinematic=True, label="ball")
    bcfg = newton.ModelBuilder.ShapeConfig(); bcfg.density, bcfg.mu = 0.0, 0.5
    builder.add_shape_sphere(bbody, radius=args.ball_radius, cfg=bcfg)

model = builder.finalize()
model.soft_contact_ke = 1.0e1; model.soft_contact_kd = 1.0e-6; model.soft_contact_mu = 0.8
model.set_gravity((0,0,-9.81))
solver = newton.solvers.SolverStyle3D(model=model, iterations=args.iters)
solver._precompute(builder)

# Style3D self-contact thickness lives on the solver's Collision handler (hard-coded 3mm radius →
# 6mm gap, too thin: front/back layers bleed in the overhead capture). Bump the radius + scale the
# layer-separation springs so folded layers rest ~2*self_thick apart. (calibrated in data_gen.py)
if getattr(solver, "collision", None) is not None:
    solver.collision.radius    = float(args.self_thick)
    solver.collision.stiff_vf *= args.self_stiff   # vertex-face layer-separation spring
    solver.collision.stiff_ee *= args.self_stiff   # edge-edge
    solver.collision.stiff_ef *= args.self_stiff   # edge-face (untangling)
    solver.collision.rebuild_bvh(model.particle_q)
    print(f"[demo] self-contact radius {args.self_thick*1e3:.0f}mm "
          f"(layer gap ~{2*args.self_thick*1e3:.0f}mm) stiff x{args.self_stiff:g}")
else:
    print("[demo] WARNING: solver.collision is None — self-contact disabled, UV bleeding likely")
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


def _snapshot_cloth():
    """Capture full cloth state so we can replay K different actions from the SAME crumple
    (per-state RLOO baseline). Returns particle q/qd for both solver states + active flags."""
    return (s0.particle_q.numpy().copy(),  s0.particle_qd.numpy().copy(),
            s1.particle_q.numpy().copy(),  s1.particle_qd.numpy().copy(),
            model.particle_flags.numpy().copy())


def _restore_cloth(snap):
    q0, qd0, q1, qd1, flags = snap
    s0.particle_q.assign(q0); s0.particle_qd.assign(qd0)
    s1.particle_q.assign(q1); s1.particle_qd.assign(qd1)
    model.particle_flags.assign(flags)
    wp.synchronize()


GRAB_RADIUS = args.grab_radius   # grasp pinch radius (m): verts within this of the anchor get pinned (robot fingertip pinch ≈ 0.01)


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


def _run_student_infer(state_npz, out_json, greedy=False):
    """Call student_infer.py (UV Mapper encode + StudentVLA sample). Returns action dict or None.
    greedy=True → deterministic action (argmax grasp + mean drag) for eval."""
    policy_path = args.student_policy or os.path.join(_ROOT, "checkpoints", "student_vla.pth")
    mdl_path    = args.model or os.path.join(_ROOT, "checkpoints", "uv_mapper_best.pth")
    cmd = ["conda", "run", "-n", args.infer_env, "--no-capture-output",
           "python", os.path.join(_ROOT, "workers", "student_infer.py"),
           "--npz", state_npz, "--out", out_json, "--policy", policy_path, "--model", mdl_path]
    if greedy:
        cmd.append("--greedy")
    if args.no_rot:
        cmd.append("--no-rot")
    ret = subprocess.run(cmd, cwd=_ROOT, env=os.environ)
    if ret.returncode != 0 or not os.path.exists(out_json):
        print("[student] student_infer.py failed"); return None
    with open(out_json) as fh:
        return json.load(fh)


def _run_student_update(buf_path, n):
    """Call student_update.py for one PPO+GAE update over the last n rollout steps."""
    policy_path = args.student_policy or os.path.join(_ROOT, "checkpoints", "student_vla.pth")
    cmd = ["conda", "run", "-n", args.infer_env, "--no-capture-output",
           "python", os.path.join(_ROOT, "workers", "student_update.py"),
           "--buffer", buf_path, "--n", str(n), "--policy", policy_path,
           "--gamma", str(args.rl_gamma), "--lam", str(args.rl_lambda),
           "--clip", str(args.rl_clip), "--epochs", str(args.rl_epochs),
           "--ent-weight", str(args.rl_ent_weight), "--vf-weight", str(args.rl_vf_weight)]
    if args.rl_det_critic:
        cmd.append("--deterministic-critic")
    print(f"[student] PPO+GAE update on last {n} rollout steps ...")
    subprocess.run(cmd, cwd=_ROOT, env=os.environ)


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


def _polyline_point(pts, s):
    """Point at arclength fraction s∈[0,1] along the polyline pts (M,3)."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L   = float(seg.sum())
    if L < 1e-9:
        return pts[-1].astype(np.float32).copy()
    target, acc = s * L, 0.0
    for i in range(len(seg)):
        if seg[i] < 1e-9:
            continue
        if acc + seg[i] >= target:
            t = (target - acc) / seg[i]
            return (pts[i] * (1 - t) + pts[i + 1] * t).astype(np.float32)
        acc += seg[i]
    return pts[-1].astype(np.float32).copy()


def _quat_to_R(q):
    """Unit quaternion [x,y,z,w] → 3×3 rotation matrix (numpy, execution side)."""
    x, y, z, w = (np.asarray(q, np.float64) / (np.linalg.norm(q) + 1e-9))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]], np.float32)


def _slerp(q0, q1, t):
    """SLERP (spherical linear interpolation) between quaternions: glide along the shortest arc on the
    rotation sphere at constant angular speed. Sign-aligned (q≡-q) so it never takes the long way."""
    q0 = np.asarray(q0, np.float64) / (np.linalg.norm(q0) + 1e-9)
    q1 = np.asarray(q1, np.float64) / (np.linalg.norm(q1) + 1e-9)
    d = float(np.dot(q0, q1))
    if d < 0.0:                                             # flip to the nearer hemisphere
        q1, d = -q1, -d
    if d > 0.9995:                                          # near-parallel → plain lerp (avoids /sin≈0)
        q = q0 + t * (q1 - q0)
        return q / (np.linalg.norm(q) + 1e-9)
    th0 = np.arccos(d); s0 = np.sin(th0)
    return (np.sin(th0 * (1 - t)) / s0) * q0 + (np.sin(th0 * t) / s0) * q1


def _polyline_quat(pts, quats, s):
    """Orientation at arclength fraction s∈[0,1] along the polyline pts (M,3), SLERPing the per-vertex
    quats — mirrors _polyline_point so rotation and position advance together."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L   = float(seg.sum())
    if L < 1e-9:
        return np.asarray(quats[-1], np.float64)
    target, acc = s * L, 0.0
    for i in range(len(seg)):
        if seg[i] < 1e-9:
            continue
        if acc + seg[i] >= target:
            return _slerp(quats[i], quats[i + 1], (target - acc) / seg[i])
        acc += seg[i]
    return np.asarray(quats[-1], np.float64)


ROT_PIVOT_OFFSET = 0.05    # wrist pivot sits 5 cm above the grasped patch, along the patch normal


def _execute_drag_path(arm, pcd_to_mesh, centroid):
    """Single-arm grab+drag: pin the grab patch, follow grab → path → release, applying the per-waypoint
    WRIST ROTATION about a pivot ROT_PIVOT_OFFSET above the patch along its mean normal — so rotating the
    wrist sweeps the patch through an arc (folding), not a spin in place.

    arm["release"]/arm["path"] are XY-rel-centroid, Z-absolute. arm["path_quat"]/["release_quat"] are the
    [x,y,z,w] wrist orientations per waypoint; absent (Haiku/scripted/manual) ⇒ identity ⇒ the transform
    collapses to the original pure-translation drag (unchanged behaviour)."""
    held, start = _find_held(arm["pcd_idx"], pcd_to_mesh)
    grab_world  = start.mean(0).astype(np.float32)          # grasp anchor (world)
    cx, cy = float(centroid[0]), float(centroid[1])
    to_world = lambda p: np.array([cx + p[0], cy + p[1], p[2]], dtype=np.float32)

    traj = np.array([grab_world] + [to_world(p) for p in arm["path"]] + [to_world(arm["release"])],
                    dtype=np.float32)

    # cap total drag length: a random/aggressive multi-waypoint path can sum to metres and tear the
    # cloth (solver blow-up → black screen). Scale every waypoint uniformly toward the grab so the
    # polyline length ≤ --max-drag-len; waypoint count (and quat alignment) is preserved.
    seg_len = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    if seg_len > args.max_drag_len:
        traj = (grab_world + (traj - grab_world) * (args.max_drag_len / seg_len)).astype(np.float32)
        print(f"[vla] drag path {seg_len:.2f}m → capped to {args.max_drag_len:g}m")

    # orientation keyframes aligned 1:1 with traj. Grab keyframe = first waypoint's orientation (no
    # reorientation on the grab→first-waypoint leg); the rest are the model's per-waypoint quats.
    ident = list(il_dataset.IDENTITY_QUAT)
    pq = arm.get("path_quat") or [ident] * len(arm["path"])
    rq = arm.get("release_quat") or ident
    wp_quats = [list(q) for q in pq] + [list(rq)]           # aligned with traj[1:]
    quats = [wp_quats[0]] + wp_quats                        # prepend grab keyframe → aligned with traj
    R0 = _quat_to_R(quats[0])

    # pivot 5 cm above the patch along its mean normal; the patch is rigid about it
    nrm = estimate_normals(verts(), k=args.normal_k)[held].mean(0)
    nrm = (nrm / (np.linalg.norm(nrm) + 1e-9)).astype(np.float32)
    rel = (start - (grab_world + ROT_PIVOT_OFFSET * nrm)).astype(np.float32)   # offsets from pivot at grasp

    def patch_target(frac):
        e = _ease(frac)
        T = _polyline_point(traj, e)
        R = _quat_to_R(_polyline_quat(traj, quats, e))
        pivot = T + ROT_PIVOT_OFFSET * nrm
        return (pivot + rel @ (R @ R0.T).T).astype(np.float32)                 # (H,3) world targets

    _pin(held)
    # constant drag speed: derive frame count from total path length (sim runs at 60 Hz)
    path_len = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    n_frames = int(np.clip(round(path_len / max(args.drag_speed, 1e-3) * 60.0), 20, 1500))
    print(f"[vla] drag {path_len:.3f}m @ {args.drag_speed:g} m/s → {n_frames} frames ({n_frames/60.0:.2f}s)")

    for i in range(n_frames):
        if not simulation_app.is_running(): break
        _drive([held], [start], [patch_target((i + 1) / n_frames) - start])
        step(); _render()

    rel_tgt = patch_target(1.0)                             # hold at release
    for _ in range(10):
        if not simulation_app.is_running(): break
        _drive([held], [start], [rel_tgt - start])
        step(); _render()

    _unpin(held)
    print(f"[vla] settling {args.rl_settle} frames ...")
    for i in range(args.rl_settle):
        if not simulation_app.is_running(): break
        step(); pump(i, "vla-settle ")

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

    # LEFT (target): the canonical flat silhouette + orientation the fabric must fill, centred on
    # the cloud centroid (= action-frame origin) and projected with the SAME camera model. The
    # offsets are centroid-relative, so cx,cy cancel; the outline lies on the table (z=_TABLE_Z).
    ddt = max(CAM_Z - _TABLE_Z, 1e-4)
    for poly in _flat_target_contour():
        pts = [(ox / ddt * CAM_FX + CAM_CX, -oy / ddt * CAM_FY + CAM_CY) for ox, oy in poly]
        if len(pts) > 1:
            dcam.line([(int(x), int(y)) for x, y in pts] + [(int(pts[0][0]), int(pts[0][1]))],
                      fill=(255, 255, 255), width=3)

    try:    font = ImageFont.load_default()
    except: font = None
    def _label(draw, xy, text, fill=(255, 255, 0)):
        try:    draw.text(xy, text, fill=fill, font=font, stroke_width=1, stroke_fill=(0, 0, 0))
        except TypeError:  draw.text(xy, text, fill=fill, font=font)

    # MAIN image = the overhead camera: crumpled garment (UV-coloured) with the white target outline
    # laid on the floor behind it. SIDE = a small flat-UV reference window (the colour key only).
    SIDE = 200
    uvp  = _flat_uv_panel(SIDE)
    dvp  = ImageDraw.Draw(uvp)
    for t in (0.0, 0.5, 1.0):                                  # UV axis ticks
        x = int(t * (SIDE - 1)); _label(dvp, (min(x, SIDE - 26), SIDE - 13), f"u={t:.1f}")
        y = int((1.0 - t) * (SIDE - 1)); _label(dvp, (2, min(max(y - 6, 0), SIDE - 13)), f"v={t:.1f}")

    gap    = 12
    canvas = Image.new("RGB", (W + gap + SIDE, H), (0, 0, 0))
    canvas.paste(cam, (0, 0))
    canvas.paste(uvp, (W + gap, 0))                            # small reference, top of the side strip
    dc = ImageDraw.Draw(canvas)
    _label(dc, (6, 6),  "MAIN: overhead camera — fabric coloured by predicted UV (R=u,G=v)")
    _label(dc, (6, 20), "WHITE outline on the floor = goal flat shape + orientation; fill it",
           fill=(255, 255, 255))
    _label(dc, (W + gap + 4, SIDE + 4), "ref: flat UV", fill=(200, 200, 200))
    _label(dc, (W + gap + 4, SIDE + 18), "(colour key)", fill=(200, 200, 200))
    canvas.save(out_path)
    return out_path


import tempfile as _tempfile
_UV_PRED_PATH = os.path.join(_tempfile.gettempdir(), "uv_pred.npy")   # last predicted UV (for IL logging)


def _run_vlm_action(state_npz):
    """UV-infer → render overlay → Haiku picks ONE grab (UV) + release + path → resolve grab to a
    pcd index. Returns (arm, centroid) where arm has pcd_idx, grab_u/v, release, path, grab_xyz,
    reasoning; or (None, None). Leaves the predicted UV at _UV_PRED_PATH for the IL logger."""
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

    src = data["arm1"]
    idx = _uv_to_pcd_idx(src["grab_u"], src["grab_v"], uv_pred, pcd_xyz, centroid)
    arm = {"pcd_idx":  idx,
           "grab_u":   float(src["grab_u"]), "grab_v": float(src["grab_v"]),
           "release":  [float(c) for c in src["release"]],
           "path":     [[float(c) for c in p] for p in src["path"]],
           "grab_xyz": (pcd_xyz[idx] + centroid).tolist(),
           "reasoning": src.get("reasoning", "")}
    r = arm["release"]
    print(f"[vla] grab pcd={idx} uv=({uv_pred[idx,0]:.2f},{uv_pred[idx,1]:.2f})  "
          f"release=({r[0]:.2f},{r[1]:.2f},{r[2]:.2f})  +{len(arm['path'])} waypoints")
    return arm, centroid


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


# ── flat-template reward (StudentVLA / --rl-student) ───────────────────────────────────────────
# Goal: "get the garment as close to the canonical FLAT layout as possible." Built from the flat
# template (FLAT_REF_PTS) + ground-truth per-vertex UV (PANEL_UV_ALL). Translation-invariant (flatten
# anywhere on the table) but ORIENTATION-AWARE: shape/iou are measured rotation-invariantly (pure
# flatness), and a separate `orient` term penalises the garment's rotation off the reference pose.
FLAT_W_SHAPE  = 1.0    # weight: 2D-aligned per-vertex XY distance to flat (shape + UV match)  (main)
FLAT_W_FLAT   = 0.5    # weight: max(0, mean(z) - table) — net lift above flat       (lie-flat term)
FLAT_W_COV    = 0.5    # weight: top-down outline IoU vs the flat footprint          (silhouette term)
FLAT_W_ORIENT = 0.3    # weight: garment rotation (rad) off the reference pose       (orientation term)
FLAT_GRID    = 96      # raster resolution for the outline-coverage IoU
_TABLE_Z     = float(FLAT_REF_PTS[:, 2].mean())   # table height of the flat template


def _align_xy(P, Q):
    """Best-fit rigid transform (rotation + translation) in the table plane mapping P→Q (Kabsch,
    exact: P,Q share vertex correspondence). Z is left untouched. Returns (aligned copy of P, angle),
    where angle (rad, 0..π) is the ROTATION the fit had to apply = how mis-oriented P is from Q. The
    aligned copy is orientation-INVARIANT (rotation removed), so shape/iou measure pure flatness; the
    returned angle is the separate, tunable orientation signal."""
    Pxy, Qxy = P[:, :2], Q[:, :2]
    pc, qc   = Pxy.mean(0), Qxy.mean(0)
    H = (Pxy - pc).T @ (Qxy - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.array([[1, 0], [0, d]], np.float64) @ U.T
    out = P.copy()
    out[:, :2] = (Pxy - pc) @ R.T + qc
    angle = float(abs(np.arctan2(R[1, 0], R[0, 0])))                  # rotation magnitude (rad)
    return out, angle


def _footprint_mask(xy, lo, cell, G):
    """Rasterise an XY point set into a (G,G) occupancy mask."""
    ij = np.floor((xy - lo) / cell).astype(np.int32)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < G) & (ij[:, 1] >= 0) & (ij[:, 1] < G)
    m  = np.zeros((G, G), bool)
    m[ij[ok, 1], ij[ok, 0]] = True
    return m


# precompute the flat template's raster frame + footprint once
_FLAT_LO   = FLAT_REF_PTS[:, :2].min(0) - 0.05
_FLAT_HI   = FLAT_REF_PTS[:, :2].max(0) + 0.05
_FLAT_CELL = float((_FLAT_HI - _FLAT_LO).max()) / FLAT_GRID
_FLAT_MASK = _footprint_mask(FLAT_REF_PTS[:, :2], _FLAT_LO, _FLAT_CELL, FLAT_GRID)


def _flat_target_contour():
    """Outline polygons of the canonical FLAT garment silhouette, as XY offsets (metres) from the
    template's own centre. Drawn on the LEFT camera panel (centred on the cloud centroid) so Haiku
    sees the exact shape + ORIENTATION the fabric must fill. Fixed orientation ⇒ rotation-variant
    target: the garment must be turned into this pose, not just any flat one. Cached."""
    cache = getattr(_flat_target_contour, "_cache", None)
    if cache is not None:
        return cache
    polys = []
    try:
        import cv2
        cnts, _ = cv2.findContours(_FLAT_MASK.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tcen = FLAT_REF_PTS[:, :2].mean(0)
        for c in cnts:
            if len(c) < 3:
                continue
            ij = c[:, 0, :].astype(np.float64)                 # (P,2) = (col=x, row=y)
            xy = _FLAT_LO + (ij + 0.5) * _FLAT_CELL            # template world XY
            polys.append((xy - tcen).astype(np.float32))       # centre-relative offsets
    except Exception:
        pass
    _flat_target_contour._cache = polys
    return polys


def _flat_reward(verbose=True):
    """How close the garment is to the base FLAT 2D shape. Returns (reward, components dict).
      shape : mean per-vertex 2D (XY) distance after rigid alignment to FLAT_REF_PTS. Each vertex
              carries its UV, so this is exactly 'every UV-fabric region sitting in its flat 2D
              place' = the base-flat-shape + UV-distribution match. (the core term)
      height: max(0, mean(z) - table) — net lift ABOVE the flat reference (one-sided: only punished
              when the garment sits higher than flat; being lower, which can't really happen on a
              table, is never rewarded). Light flatness term.
      iou   : top-down outline IoU vs the flat footprint (overall 2D silhouette / no fold-under).
      orient: |rotation| (rad) the Kabsch fit applied = how far the garment is turned from the
              reference pose. Penalised so the smoothed garment ends in the canonical orientation.
    reward = -(W_SHAPE*shape + W_FLAT*height + W_ORIENT*orient) + W_COV*iou."""
    p  = verts()
    pa, orient = _align_xy(p, FLAT_REF_PTS)                                            # orient = rad off reference pose
    shape  = float(np.mean(np.linalg.norm(pa[:, :2] - FLAT_REF_PTS[:, :2], axis=1)))   # 2D shape / UV match (rot-invariant)
    height = float(max(0.0, p[:, 2].mean() - _TABLE_Z))                                # net lift ABOVE flat (one-sided)
    cur    = _footprint_mask(pa[:, :2], _FLAT_LO, _FLAT_CELL, FLAT_GRID)
    inter  = np.logical_and(cur, _FLAT_MASK).sum()
    union  = np.logical_or(cur, _FLAT_MASK).sum()
    iou    = float(inter / union) if union else 0.0
    reward = -(FLAT_W_SHAPE * shape + FLAT_W_FLAT * height + FLAT_W_ORIENT * orient) + FLAT_W_COV * iou
    if verbose:
        print(f"[flat] reward={reward:.4f}  shape={shape*100:.1f}cm  height={height*100:.1f}cm  "
              f"orient={np.degrees(orient):.0f}deg  iou={iou:.3f}")
    return reward, {"shape": shape, "height": height, "orient": orient, "iou": iou}


def _render_flat_overlay(path, S=360):
    """Diagnostic PNG: LEFT = flat target (UV-coloured), RIGHT = current garment 2D-aligned to the
    template (UV-coloured) with the flat OUTLINE overlaid in white. Watch RIGHT converge onto LEFT."""
    from PIL import Image, ImageDraw
    span = float((_FLAT_HI - _FLAT_LO).max())
    def _to_px(xy):
        q = (xy - _FLAT_LO) / span
        return (q[:, 0] * (S - 1)).astype(np.int32), ((1 - q[:, 1]) * (S - 1)).astype(np.int32)
    def _panel(xy, draw_outline):
        img = Image.new("RGB", (S, S), (12, 12, 16)); d = ImageDraw.Draw(img)
        px, py = _to_px(xy)
        for i in range(0, len(xy), 3):                       # subsample for speed
            if 0 <= px[i] < S and 0 <= py[i] < S:
                d.point((px[i], py[i]), fill=_uv_dot_color(PANEL_UV_ALL[i, 0], PANEL_UV_ALL[i, 1]))
        if draw_outline:                                     # flat footprint contour in white
            try:
                import cv2
                cnts, _ = cv2.findContours(_FLAT_MASK.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    pts = [((int(x) + 0.5) / FLAT_GRID * S, (1 - (int(y) + 0.5) / FLAT_GRID) * S)
                           for x, y in c[:, 0, :]]
                    if len(pts) > 1: d.line(pts + [pts[0]], fill=(255, 255, 255), width=1)
            except Exception: pass
        return img
    pa, _ = _align_xy(verts(), FLAT_REF_PTS)
    canvas = Image.new("RGB", (2 * S + 12, S), (0, 0, 0))
    canvas.paste(_panel(FLAT_REF_PTS[:, :2], False), (0, 0))
    canvas.paste(_panel(pa[:, :2], True),            (S + 12, 0))
    canvas.save(path)
    return path


def _clamp_wp(p, centroid):
    """Safety-clamp a sampled waypoint into the FIXED world workspace box (the arm's physical reach):
    world XY within [center ± arm_box_half], Z in [0, arm_box_zmax]. The box is anchored at the spawn
    (--arm-box-cx/cy) and does NOT move with the cloth. p is centroid-relative XY + absolute Z, so we
    add the centroid to get world, clamp there, then subtract it back to keep the pipeline's
    centroid-relative frame unchanged."""
    cx, cy = float(centroid[0]), float(centroid[1])
    h, bx, by = args.arm_box_half, args.arm_box_cx, args.arm_box_cy
    wx = float(np.clip(cx + p[0], bx - h, bx + h))     # clamp in WORLD
    wy = float(np.clip(cy + p[1], by - h, by + h))
    return [wx - cx, wy - cy, float(np.clip(p[2], 0.0, args.arm_box_zmax))]


def _in_box(world_xy):
    """True where a world XY lies inside the fixed workspace box."""
    h, bx, by = args.arm_box_half, args.arm_box_cx, args.arm_box_cy
    return (np.abs(world_xy[..., 0] - bx) <= h) & (np.abs(world_xy[..., 1] - by) <= h)


def _student_arm(act, pcd_xyz, centroid):
    """Resolve a student action dict → an executable arm for _execute_drag_path. The policy picks a
    point INDEX directly (grasp head is a Categorical over the same cloud head_RL captured), so use it
    as-is; fall back to nearest predicted-UV only for legacy/odd indices.

    The arm may only act INSIDE the fixed workspace box (--arm-box-*). Drag waypoints are clamped into
    it. The grab must also be in-box: if the policy's grab point has drifted OUTSIDE, we DON'T grab
    there — instead a RECOVERY move grabs the nearest in-box cloth point and drags it toward the box
    centre to pull the garment back into reach (arm["recovery"]=True; head_RL skips logging it to the
    PPO buffer, so this safety override never trains the policy off its own samples)."""
    idx = int(act.get("grab_idx", -1))
    if not (0 <= idx < len(pcd_xyz)):
        uv_pred = np.load(act["uv_pred_path"]).astype(np.float32)
        idx     = _uv_to_pcd_idx(act["grab_u"], act["grab_v"], uv_pred, pcd_xyz, centroid)

    world  = pcd_xyz + centroid                                   # (N,3) cloth points in world
    in_box = _in_box(world[:, :2])

    if in_box[idx]:                                               # normal: policy grab is reachable
        return {"pcd_idx": idx, "grab_u": act["grab_u"], "grab_v": act["grab_v"],
                "release": _clamp_wp(act["release"], centroid),
                "path": [_clamp_wp(p, centroid) for p in act["path"]],
                "path_quat": act.get("path_quat"), "release_quat": act.get("release_quat"),
                "grab_xyz": world[idx].tolist(), "recovery": False}

    # RECOVERY: grab the in-box cloth point nearest the (unreachable) chosen grab, then drag it a
    # capped step toward the box centre. Pure translation (identity rotation), no path waypoints.
    ctr = np.array([args.arm_box_cx, args.arm_box_cy], np.float32)
    if in_box.any():
        cand = np.where(in_box)[0]
        ridx = int(cand[np.argmin(np.linalg.norm(world[cand, :2] - world[idx, :2], axis=1))])
    else:                                                         # whole cloth outside box (extreme)
        ridx = int(np.argmin(np.linalg.norm(world[:, :2] - ctr, axis=1)))
    to_ctr = ctr - world[ridx, :2]
    dist   = float(np.linalg.norm(to_ctr))
    tgt_xy = world[ridx, :2] + (to_ctr / (dist + 1e-9)) * min(dist, args.recovery_step)
    release = [float(tgt_xy[0] - centroid[0]), float(tgt_xy[1] - centroid[1]), float(_TABLE_Z)]
    print(f"[student] grab outside box → RECOVERY: grab {idx}→{ridx}, drag {min(dist, args.recovery_step):.2f}m toward centre")
    return {"pcd_idx": ridx, "grab_u": act["grab_u"], "grab_v": act["grab_v"],
            "release": release, "path": [], "path_quat": [], "release_quat": list(il_dataset.IDENTITY_QUAT),
            "grab_xyz": world[ridx].tolist(), "recovery": True}


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


# ── manual mouse-drag mode (--manual) ────────────────────────────────────────────────────
def _project_verts(p, cx, cy):
    """World (N,3) → overhead-camera pixels (N,2). Same model as _render_uv_overlay."""
    dd  = np.maximum(CAM_Z - p[:, 2], 1e-4)
    upx = (p[:, 0] - cx) / dd * CAM_FX + CAM_CX
    vpx = -(p[:, 1] - cy) / dd * CAM_FY + CAM_CY
    return np.stack([upx, vpx], axis=1)


def _pixel_to_world_xy(px, py, z, cx, cy):
    """Inverse projection at a KNOWN height z → world (x,y). The overhead cam points
    straight down, so the image plane is world-XY and holding z fixed = pure XY drag."""
    dd = CAM_Z - z
    wx = (px - CAM_CX) / CAM_FX * dd + cx
    wy = -(py - CAM_CY) / CAM_FY * dd + cy
    return float(wx), float(wy)


def _capture_overhead(warmup=1):
    """Sync live cloth into the vis mesh and grab one overhead RGB frame (uint8, no alpha)."""
    vmesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(verts()))
    for _ in range(warmup):
        world.step(render=True); rep.orchestrator.step(pause_timeline=False)
    rraw = rgb_ann.get_data()
    rgb  = np.asarray(rraw.get("data", rraw) if isinstance(rraw, dict) else rraw)
    if rgb.ndim == 3 and rgb.shape[2] == 4: rgb = rgb[:, :, :3]
    return rgb.astype(np.uint8)


def _manual_loop():
    """Click a point on the overhead view to grab it, drag with the mouse (Z locked at pickup
    height, wheel raises/lowers), release to drop and settle. Each drag is logged as an IL
    sample (source='manual') so human demos feed the same dataset as the Haiku teacher."""
    import matplotlib
    matplotlib.use("TkAgg")          # interactive window (fold's cv2 is headless → no imshow)
    import matplotlib.pyplot as plt

    def _capture_state():
        npz = "/tmp/manual_state.npz"
        _capture_rl_state(npz)                       # places cam at cloth centroid, writes state
        d   = np.load(npz)
        uvp = None
        if args.il and _run_uv_infer(npz, _UV_PRED_PATH):
            try: uvp = np.load(_UV_PRED_PATH).astype(np.float32)
            except Exception: uvp = None
        return npz, d["pcd_xyz"].astype(np.float32), d["centroid"].astype(np.float32), uvp

    tmp_npz, pcd_xyz, centroid, uv_pred = _capture_state()
    cx, cy = float(centroid[0]), float(centroid[1])  # fixed cam centre (matches _capture_rl_state)

    M = {"down": False, "px": (0, 0), "held": None, "start": None, "z": 0.0,
         "grab_world": None, "grab_pcd": -1, "path": [], "release": False}

    def _begin_grab(px, py):
        p   = verts()
        d2  = np.sum((_project_verts(p, cx, cy) - np.array([px, py]))**2, axis=1)
        near = np.where(d2 < 14.0**2)[0]
        anchor = int(near[np.argmax(p[near, 2])]) if len(near) else int(d2.argmin())  # topmost layer
        held = np.where(np.linalg.norm(p - p[anchor], axis=1) < GRAB_RADIUS)[0]
        if len(held) == 0: held = np.array([anchor])
        world_xyz = (pcd_xyz + centroid)
        M.update(down=True, held=held, start=p[held].copy(), z=float(p[anchor, 2]),
                 grab_world=p[held].mean(0).astype(np.float32), path=[],
                 grab_pcd=int(np.argmin(np.linalg.norm(world_xyz - p[anchor], axis=1))))
        _pin(held)
        print(f"[manual] grab vert={anchor}  {len(held)} verts  z={M['z']:.3f}")

    def _log_manual():
        if not (args.il and uv_pred is not None and M["grab_pcd"] >= 0):
            return
        i  = M["grab_pcd"]
        # full traj grab → mouse waypoints → release, resampled to PATH_LEN points spread
        # evenly by arclength (so the 5 saved points span the whole drag, not just its start).
        traj = np.array([M["grab_world"]] + M["path"], dtype=np.float32)
        if len(traj) == 1:
            traj = np.vstack([traj, traj])
        to_frame = lambda w: [float(w[0] - cx), float(w[1] - cy), float(w[2])]
        release  = to_frame(traj[-1])
        path     = [to_frame(_polyline_point(traj, (j + 1) / (il_dataset.PATH_LEN + 1)))
                    for j in range(il_dataset.PATH_LEN)]
        try:
            d = np.load(tmp_npz)
            state  = {"pcd_xyz": d["pcd_xyz"], "normals": d["normals"],
                      "uv_pred": uv_pred,      "centroid": d["centroid"]}
            action = {"arm1": il_dataset.make_arm(
                          float(uv_pred[i, 0]), float(uv_pred[i, 1]), release, path,
                          grab_quat=il_dataset.IDENTITY_QUAT,   # mouse = vertical-down approach
                          grab_xyz=M["grab_world"], grab_pcd_idx=i, reasoning="manual mouse drag")}
            sid = il_dataset.record_sample(IL_DIR, "manual", state, action)
            print(f"[il] logged {sid}  ({il_dataset.count(IL_DIR)} samples total)")
        except Exception as e:
            print(f"[il] WARNING: failed to log sample: {e}")

    def _end_grab():
        _unpin(M["held"])
        print(f"[manual] release → settling {args.rl_settle} frames ...")
        for _ in range(args.rl_settle):
            if not simulation_app.is_running(): break
            step()
        _log_manual()
        M["down"] = False

    M["recrumple"] = M["quit"] = False

    def on_press(e):
        if e.button == 1 and not M["down"] and e.xdata is not None:
            _begin_grab(e.xdata, e.ydata)
    def on_move(e):
        if e.xdata is not None: M["px"] = (e.xdata, e.ydata)
    def on_release(e):
        if e.button == 1 and M["down"]:
            if e.xdata is not None: M["px"] = (e.xdata, e.ydata)
            M["release"] = True
    def on_scroll(e):
        if M["down"]: M["z"] += args.manual_z_step * (1 if e.button == "up" else -1)
    def on_key(e):
        if e.key == "q": M["quit"] = True
        elif e.key == "r" and not M["down"]: M["recrumple"] = True

    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(_capture_overhead(1))
    ax.set_title("LMB drag = grab+move   wheel = Z   r = re-crumple   q = quit")
    marker, = ax.plot([], [], "o", mfc="none", mec="lime", ms=14, mew=2)
    txt = ax.text(8, 24, "", color="yellow", fontsize=10)
    for ev, cb in (("button_press_event", on_press), ("motion_notify_event", on_move),
                   ("button_release_event", on_release), ("scroll_event", on_scroll),
                   ("key_press_event", on_key)):
        fig.canvas.mpl_connect(ev, cb)
    print("[manual] LMB drag = grab+move | wheel = raise/lower Z | r = re-crumple | q = quit")
    try:
        while simulation_app.is_running() and not M["quit"] and plt.fignum_exists(fig.number):
            if M["release"]:
                M["release"] = False
                _end_grab()
            elif M["down"]:
                tx, ty = _pixel_to_world_xy(M["px"][0], M["px"][1], M["z"], cx, cy)
                tgt = np.array([tx, ty, M["z"]], dtype=np.float32)
                _drive([M["held"]], [M["start"]], [tgt - M["grab_world"]])
                if not M["path"] or np.linalg.norm(tgt - M["path"][-1]) > 0.01:
                    M["path"].append(tgt.copy())
            step()

            im.set_data(_capture_overhead(1))
            if M["down"]:
                marker.set_data([M["px"][0]], [M["px"][1]]); txt.set_text(f"z={M['z']:.3f}")
            else:
                marker.set_data([], []); txt.set_text("")
            fig.canvas.draw_idle(); fig.canvas.flush_events()
            plt.pause(0.001)

            if M["recrumple"]:
                M["recrumple"] = False
                print("[manual] re-crumpling ...")
                _reset_cloth(); _crumple()
                tmp_npz, pcd_xyz, centroid, uv_pred = _capture_state()
                cx, cy = float(centroid[0]), float(centroid[1])
                im.set_data(_capture_overhead(1))
    except KeyboardInterrupt:
        pass
    finally:
        plt.close("all")


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

# ── manual mouse-drag loop (--manual) ─────────────────────────────────────────────────────
if args.manual and simulation_app.is_running():
    _reset_cloth(); _crumple()
    _manual_loop()

# ── VLA loop (default — Haiku drives 2 arms from UV+xyz state) ─────────────────────────────
if not args.rl and not args.manual and not args.rl_student and not args.student_eval and not args.scripted and simulation_app.is_running():
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

                a1, centroid = _run_vlm_action(tmp_npz)
                if a1 is None:
                    print("[vla] vlm_action failed — skipping turn")
                    break

                flat_before = _flatness()
                _execute_drag_path(a1, pcd_to_mesh, centroid)
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
                                        a1["release"], a1["path"],
                                        grab_xyz=a1["grab_xyz"], grab_pcd_idx=a1["pcd_idx"],
                                        reasoning=a1.get("reasoning", "")),
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

# ── RL-student loop (--rl-student — MULTI-STEP PPO+GAE on the StudentVLA policy) ─────────────
# A grab is NOT scored by its own immediate result — it is scored by the discounted future of the
# whole smoothing SEQUENCE it belongs to, so a tension-building move that lowers flatness now is
# rewarded for the larger gain it unlocks two grabs later. Per crumpled state we branch K full
# sequences (best-of-N diversity from one start); each sequence runs --rl-turns grabs, and every
# grab logs (state, action, per-step flatness improvement r_t, trajectory id, done). student_update
# forms the GAE advantage + discounted return per trajectory and does a clipped PPO update; the
# critic head V(s) is the baseline. One buffer entry == one grab.
if args.rl_student and simulation_app.is_running():
    buf_path = args.student_buffer
    K        = max(1, args.rl_group_k)          # sequences branched from each crumple
    T        = args.rl_turns                    # grabs per sequence
    episode  = 0
    try:
        while simulation_app.is_running():
            episode += 1
            print(f"\n[student] ══ episode {episode}  ({K} branch × {T} turns, PPO+GAE) ══")
            _reset_cloth()
            _crumple()
            snap = _snapshot_cloth()            # every branch replays from this identical crumple

            for k in range(K):
                if not simulation_app.is_running():
                    break
                _restore_cloth(snap)
                traj          = f"{episode}_{k}"
                phi_prev, _   = _flat_reward(verbose=False)       # Φ(s_0) for this branch
                ret           = 0.0
                for t in range(T):
                    if not simulation_app.is_running():
                        break
                    tmp_npz  = f"/tmp/student_{traj}_t{t}.npz"     # one state per grab (cloth evolves)
                    tmp_json = f"/tmp/student_act_{traj}_t{t}.json"
                    pcd_to_mesh = _capture_rl_state(tmp_npz)
                    if pcd_to_mesh is None:
                        break
                    act = _run_student_infer(tmp_npz, tmp_json)    # stochastic
                    if os.path.exists(tmp_json): os.unlink(tmp_json)
                    if act is None:
                        break

                    # diagnostic: two-panel UV overlay (predicted UV on the live cloud + flat template),
                    # same render the Haiku path used → rl_uv_overlay.png in the repo root
                    try:
                        ovl = _render_uv_overlay(tmp_npz, act["uv_pred_path"],
                                                 os.path.join(_ROOT, "rl_capture_latest.png"))
                        print(f"[student] UV overlay → {ovl}")
                    except Exception as e:
                        print(f"[student] UV overlay skipped: {e}")

                    d        = np.load(tmp_npz)
                    pcd_xyz  = d["pcd_xyz"].astype(np.float32)
                    centroid = d["centroid"].astype(np.float32)
                    arm = _student_arm(act, pcd_xyz, centroid)
                    _execute_drag_path(arm, pcd_to_mesh, centroid)

                    phi, comps = _flat_reward()
                    r_t  = phi - phi_prev                          # per-step flatness improvement
                    done = (t == T - 1)                            # artificial horizon → bootstrap stops
                    phi_prev = phi
                    ret += r_t
                    _render_flat_overlay(os.path.join(_ROOT, "rl_flat_overlay.png"))
                    print(f"[student] {traj} turn {t+1}/{T}  Φ={phi:.4f}  r={r_t:+.4f}")

                    # recovery moves are a safety override, NOT a policy sample → don't train on them
                    # (skip the buffer; clean up their temp files student_update would otherwise delete)
                    if arm.get("recovery"):
                        for f in (tmp_npz, act["uv_pred_path"]):
                            try: os.unlink(f)
                            except Exception: pass
                        continue

                    _append_buffer({
                        "state_npz":    tmp_npz,
                        "uv_pred_path": act["uv_pred_path"],
                        "grab_idx":     act["grab_idx"],
                        "waypoints":    act["waypoints"],
                        "active":       act["active"],
                        "log_prob":     act["log_prob"],     # behaviour log_prob (PPO ratio denominator)
                        "wp_rot3":      act.get("wp_rot3"),  # raw swing-twist sample (None if --no-rot)
                        "reward":       r_t,                 # per-step flatness improvement
                        "phi":          phi,
                        "shape":        comps["shape"],
                        "height":       comps["height"],
                        "orient":       comps["orient"],
                        "iou":          comps["iou"],
                        "traj":         traj,                # GAE/returns are computed within a trajectory
                        "t":            t,
                        "done":         done,
                        "episode":      episode,
                        "branch":       k,
                    }, buf_path)

                print(f"[student] {traj} return ΣΔΦ={ret:+.4f}")

            if episode % args.rl_k == 0:
                _run_student_update(buf_path, args.rl_k * K * T)  # entries (grabs) since last update

    except KeyboardInterrupt:
        pass

# ── student EVAL loop (--student-eval — watch the trained policy, greedy, no learning) ──────
if args.student_eval and simulation_app.is_running():
    episode = 0
    try:
        while simulation_app.is_running():
            episode += 1
            print(f"\n[eval] ══ episode {episode} (greedy) ══════════════════════════")
            _reset_cloth()
            _crumple()
            r0, _ = _flat_reward(verbose=False)
            print(f"[eval] start reward={r0:.4f}")

            for turn in range(args.rl_turns):                 # sequential greedy refinement
                if not simulation_app.is_running():
                    break
                tmp_npz  = f"/tmp/student_eval_ep{episode}_t{turn}.npz"
                tmp_json = f"/tmp/student_eval_act.json"

                pcd_to_mesh = _capture_rl_state(tmp_npz)
                if pcd_to_mesh is None:
                    break
                act = _run_student_infer(tmp_npz, tmp_json, greedy=True)
                if act is None:
                    break

                d        = np.load(tmp_npz)
                pcd_xyz  = d["pcd_xyz"].astype(np.float32)
                centroid = d["centroid"].astype(np.float32)
                arm = _student_arm(act, pcd_xyz, centroid)
                _execute_drag_path(arm, pcd_to_mesh, centroid)

                reward, comps = _flat_reward()
                _render_flat_overlay(os.path.join(_ROOT, "rl_flat_overlay.png"))
                print(f"[eval] ep {episode} turn {turn+1}/{args.rl_turns}  reward={reward:.4f}  "
                      f"shape={comps['shape']*100:.1f}cm  height={comps['height']*100:.1f}cm  iou={comps['iou']:.3f}")

                for fpath in (tmp_npz, tmp_json, act["uv_pred_path"]):
                    try: os.unlink(fpath)
                    except Exception: pass

    except KeyboardInterrupt:
        pass

# ── deterministic UV teacher (--scripted) ───────────────────────────────────────────────────
# Replays grasp_regions.json (priority order) on every crumple: for each region, grab the visible
# point whose PREDICTED UV is nearest the region centre and drag it to the region's flat target;
# advance once that patch sits within --teacher-tol of its target. No VLM, no learning — a strong
# scripted expert (UniGarmentManip-style correspondence + a human grab-order prior) that logs IL
# samples (source="scripted") to feed BC, with the same reward (_flat_reward) the student RL uses.
def _teacher_arc(grab_xy, release, lift):
    """PATH_LEN waypoints: lift off the grab, carry over, descend to the release. grab_xy/release
    are centroid-relative XY (Z absolute), matching _execute_drag_path's action frame."""
    gx, gy = float(grab_xy[0]), float(grab_xy[1])
    rx, ry, rz = float(release[0]), float(release[1]), float(release[2])
    key = np.array([[gx, gy, lift],
                    [(gx + rx) / 2.0, (gy + ry) / 2.0, lift],
                    [rx, ry, max(lift * 0.45, rz + 0.06)],
                    [rx, ry, rz]], dtype=np.float32)
    return [[float(c) for c in _polyline_point(key, (j + 1) / (il_dataset.PATH_LEN + 1))]
            for j in range(il_dataset.PATH_LEN)]


if args.scripted and simulation_app.is_running():
    import grasp_regions as _gr
    regions = _gr.load(args.regions) if args.regions else _gr.load()
    print(f"[scripted] {len(regions)} regions: " + " → ".join(r["name"] for r in regions))
    Z_TABLE = float(FLAT_REF_PTS[:, 2].mean())
    episode = 0
    try:
        while simulation_app.is_running():
            episode += 1
            print(f"\n[scripted] ══ episode {episode} ══════════════════════════════")
            _reset_cloth(); _crumple()
            before, _ = _flat_reward(verbose=False)

            for ri, reg in enumerate(regions):
                if not simulation_app.is_running():
                    break
                uc  = np.array(reg["uv_center"], dtype=np.float32)
                rad = float(reg["uv_radius"])
                tgt = np.array(reg["target_off"][:2], dtype=np.float32)
                name = reg["name"]

                for attempt in range(max(1, args.teacher_retries)):
                    if not simulation_app.is_running():
                        break
                    tmp_npz = f"/tmp/scripted_ep{episode}_r{ri}_{attempt}.npz"
                    pcd_to_mesh = _capture_rl_state(tmp_npz)
                    if pcd_to_mesh is None:
                        break
                    if _run_uv_infer(tmp_npz, _UV_PRED_PATH) is None:
                        try: os.unlink(tmp_npz)
                        except: pass
                        break
                    d        = np.load(tmp_npz)
                    pcd_xyz  = d["pcd_xyz"].astype(np.float32)   # centroid-relative
                    centroid = d["centroid"].astype(np.float32)
                    uv_pred  = np.load(_UV_PRED_PATH).astype(np.float32)

                    # save the garment coloured by predicted UV after every inference (latest frame)
                    try:
                        ov = _render_uv_overlay(tmp_npz, _UV_PRED_PATH,
                                                os.path.join(_ROOT, "rl_capture_latest.png"))
                        print(f"[scripted] UV overlay → {ov}")
                    except Exception as e:
                        print(f"[scripted] WARNING: UV overlay failed: {e}")

                    du   = np.linalg.norm(uv_pred - uc, axis=1)
                    cand = np.where(du <= rad)[0]
                    if len(cand) == 0:                        # this UV zone isn't visible — ignore it
                        print(f"[scripted] region {ri+1}/{len(regions)} '{name}' not visible "
                              f"(nearest uv {du.min():.2f} > r {rad:.2f}) — skip to next")
                        try: os.unlink(tmp_npz)
                        except: pass
                        break

                    # region 'done'? — its current patch centre already within tol of the flat target
                    cur_err = float(np.linalg.norm(pcd_xyz[cand, :2].mean(0) - tgt))
                    if cur_err <= args.teacher_tol:
                        print(f"[scripted] region {ri+1}/{len(regions)} '{name}' OK "
                              f"(err {cur_err*100:.1f}cm ≤ {args.teacher_tol*100:.0f}cm) — skip to next")
                        try: os.unlink(tmp_npz)
                        except: pass
                        break

                    # grab the candidate that is FURTHEST from its flat target (most useful pull)
                    disp = np.linalg.norm(pcd_xyz[cand, :2] - tgt, axis=1)
                    gi   = int(cand[int(disp.argmax())])
                    grab_world = (pcd_xyz[gi] + centroid).astype(np.float32)
                    release    = [float(tgt[0]), float(tgt[1]), Z_TABLE]
                    path       = _teacher_arc(pcd_xyz[gi, :2], release, args.teacher_lift)
                    arm = {"pcd_idx": gi, "grab_u": float(uv_pred[gi, 0]), "grab_v": float(uv_pred[gi, 1]),
                           "grab_xyz": grab_world, "release": release, "path": path}
                    print(f"[scripted] region {ri+1}/{len(regions)} '{name}' attempt {attempt+1} "
                          f"err {cur_err*100:.1f}cm → grab uv=({arm['grab_u']:.2f},{arm['grab_v']:.2f})")
                    _execute_drag_path(arm, pcd_to_mesh, centroid)

                    if args.il:
                        try:
                            state  = {"pcd_xyz": pcd_xyz, "normals": d["normals"],
                                      "uv_pred": uv_pred, "centroid": centroid}
                            action = {"arm1": il_dataset.make_arm(
                                          arm["grab_u"], arm["grab_v"], release, path,
                                          grab_quat=il_dataset.IDENTITY_QUAT,
                                          grab_xyz=grab_world, grab_pcd_idx=gi,
                                          reasoning=f"scripted:{name}")}
                            sid = il_dataset.record_sample(IL_DIR, "scripted", state, action,
                                      episode=episode, turn=ri, teacher_model="uv-correspondence")
                            print(f"[il] logged {sid}  ({il_dataset.count(IL_DIR)} samples total)")
                        except Exception as e:
                            print(f"[il] WARNING: failed to log sample: {e}")
                    try: os.unlink(tmp_npz)
                    except: pass

            reward, comps = _flat_reward()
            _render_flat_overlay(os.path.join(_ROOT, "rl_flat_overlay.png"))
            print(f"[scripted] ep {episode} done: flat reward {before:.3f} → {reward:.3f}")
    except KeyboardInterrupt:
        pass

simulation_app.close()
