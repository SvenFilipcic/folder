"""
head_parallel.py — VECTORIZED N-cloth StudentVLA RL (PPO+GAE), one Isaac Sim, one solver.

A scalable sibling of `head_RL.py --rl-student`. head_RL runs ONE cloth and pays a `conda run`
cold-start every grab (import Torch + load two checkpoints, GPU idle); this script:

  • spawns N cloths in a SINGLE Newton model (N× style3d.add_cloth_mesh at a tiled grid of XY
    offsets) → one solver, one CUDA graph, one step() advances ALL of them;
  • drives all N grabs CONCURRENTLY (one step() per drag frame, not N sequential drags);
  • captures via a per-env overhead DEPTH camera (matches data_gen's training distribution — partial
    occluded cloud, NOT the full mesh) in ONE render pass → N annotator reads;
  • talks to a PERSISTENT inference server (workers/infer_server.py, env `infer`) over a raw
    localhost socket (socket_ipc.py): batched UV-Mapper + policy.sample for all N envs in one call,
    and centralized PPO+GAE updates — no per-turn process spawn, weights stay resident.

Each env works in its OWN LOCAL FRAME (tile offset subtracted at the sim boundary), so every reward /
workspace-box / arm helper is reused VERBATIM from head_RL — the offset only enters when placing a
camera or writing particle positions.

Launch (two terminals):
    conda run -n infer --no-capture-output python workers/infer_server.py --port 5557
    conda run -n fold  --no-capture-output python head_parallel.py --n-envs 10 --rl-det-critic --no-rot

head_RL.py is untouched and keeps working on its own.
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
# ── parallel / server ────────────────────────────────────────────────────────
parser.add_argument("--n-envs",      type=int,   default=10,  help="number of cloths simulated in parallel in one Isaac Sim")
parser.add_argument("--tile-pitch",  type=float, default=3.0, help="grid spacing (m) between tiled cloths — big enough that cloths + their cameras never interfere")
parser.add_argument("--server-host", type=str,   default="127.0.0.1")
parser.add_argument("--server-port", type=int,   default=5557)
parser.add_argument("--gui",         action="store_true", help="open the Isaac Sim window (default headless)")
# ── garment material (calibrated Style3D real-fabric scale — identical to head_RL) ────────────
parser.add_argument("--mass",     type=float, default=0.6)
parser.add_argument("--density",  type=float, default=None)
parser.add_argument("--stretch-weft",  type=float, default=1.0e2)
parser.add_argument("--stretch-warp",  type=float, default=1.0e2)
parser.add_argument("--stretch-shear", type=float, default=1.0e1)
parser.add_argument("--damping",  type=float, default=None)
parser.add_argument("--bend-weft",  type=float, default=0.5e-5)
parser.add_argument("--bend-warp",  type=float, default=0.5e-5)
parser.add_argument("--bend-shear", type=float, default=0.6e-6)
parser.add_argument("--bend-spec",  action="store_true")
parser.add_argument("--prad",       type=float, default=5.0e-3)
parser.add_argument("--self-thick", type=float, default=0.7e-2)
parser.add_argument("--self-stiff", type=float, default=0.8)
parser.add_argument("--substeps",   type=int,   default=10)
parser.add_argument("--iters",      type=int,   default=4)
parser.add_argument("--model",      type=str,   default=None, help="UV Mapper checkpoint (server-side default checkpoints/uv_mapper_best.pth)")
parser.add_argument("--normal-k",   type=int,   default=30)
# ── crumple (grab-drape-drop) ─────────────────────────────────────────────────
parser.add_argument("--grab-reach",      type=float, default=0.35)
parser.add_argument("--grab-radius-min", type=float, default=0.01)
parser.add_argument("--grab-radius-max", type=float, default=0.03)
parser.add_argument("--grab-radius",     type=float, default=0.008)
parser.add_argument("--grab-height",     type=float, default=0.50)
parser.add_argument("--drape-frames",    type=int,   default=40)
parser.add_argument("--settle",          type=int,   default=90)
# ── workspace box / drag execution ────────────────────────────────────────────
parser.add_argument("--arm-box-half",  type=float, default=0.6)
parser.add_argument("--arm-box-cx",    type=float, default=0.0)
parser.add_argument("--arm-box-cy",    type=float, default=0.0)
parser.add_argument("--arm-box-zmax",  type=float, default=0.5)
parser.add_argument("--recovery-step", type=float, default=0.3)
parser.add_argument("--max-drag-len",  type=float, default=0.8)
parser.add_argument("--drag-speed",    type=float, default=0.12)
parser.add_argument("--rl-settle",     type=int,   default=80)
# ── RL-student (PPO+GAE) ──────────────────────────────────────────────────────
parser.add_argument("--rl-turns",      type=int,   default=4,  help="grabs per smoothing sequence")
parser.add_argument("--rl-group-k",    type=int,   default=4,  help="sequences branched per crumpled state")
parser.add_argument("--rl-k",          type=int,   default=6,  help="episodes between PPO updates")
parser.add_argument("--rl-gamma",      type=float, default=0.0)
parser.add_argument("--rl-lambda",     type=float, default=0.95)
parser.add_argument("--rl-clip",       type=float, default=0.2)
parser.add_argument("--rl-epochs",     type=int,   default=4)
parser.add_argument("--rl-minibatch",  type=int,   default=16)
parser.add_argument("--rl-ent-weight", type=float, default=1e-3)
parser.add_argument("--rl-vf-weight",  type=float, default=0.5)
parser.add_argument("--rl-det-critic", action="store_true")
parser.add_argument("--no-rot",        action="store_true")
parser.add_argument("--phi-target",    type=float, default=0.5)
args = parser.parse_args()

_ROOT = os.path.dirname(os.path.abspath(__file__))
N     = max(1, args.n_envs)

# ── Isaac bootstrap (same flags as head_RL) ───────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": not args.gui, "multi_gpu": False,
                                "renderer": "RaytracedLighting"})

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
import socket_ipc
apply_style3d_nan_patch()

MESH = os.path.join(_ROOT, "assets", "garments", "majca2.usdc")
CAM_Z, CAM_W, CAM_H, CAM_WARMUP, N_PCD = 1.0, 640, 480, 5, 4096
CAM_FX = 24.0 / 36.0 * CAM_W
CAM_FY = 24.0 / 24.0 * CAM_H
CAM_CX, CAM_CY = CAM_W / 2.0, CAM_H / 2.0
ROT_PIVOT_OFFSET = 0.05
GRAB_RADIUS = args.grab_radius


# ── geometry (identical to head_RL) ───────────────────────────────────────────
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

V, F = load_usd_mesh(MESH)
V *= 0.01
ext = V.max(0) - V.min(0); thin = int(ext.argmin())
order = [i for i in (0, 1, 2) if i != thin] + [thin]
V = V[:, order]; V[:, :2] -= V[:, :2].mean(0); V[:, 2] -= V[:, 2].min()
V[:, 2] += 0.02
NV = len(V)                                           # particles per cloth

_panel = np.load(os.path.join(_ROOT, "reference", "majca_panel_xatlas.npz"))
vmapping, uv_indices, uvs = _panel["vmapping"], _panel["indices"], _panel["uvs"].astype(np.float64)
tri3d = vmapping[uv_indices]
T3 = V[tri3d]
area3d = np.abs(0.5 * np.cross(T3[:, 1] - T3[:, 0], T3[:, 2] - T3[:, 0])[:, 2]).sum()
Tu = uvs[uv_indices]; e1 = Tu[:, 1] - Tu[:, 0]; e2 = Tu[:, 2] - Tu[:, 0]
areaUV = np.abs(0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])).sum()
uv_scale = float(np.sqrt(area3d / areaUV)) if areaUV > 0 else 1.0
panel = (uvs * uv_scale).astype(np.float32)
F = tri3d.reshape(-1).astype(np.int32)
panel_area = float(area3d)
DENSITY = args.density if args.density is not None else (args.mass / panel_area if panel_area > 0 else 0.3)
if args.bend_spec:
    args.bend_weft = args.bend_warp = args.bend_shear = 8.0e3

_g = np.load(os.path.join(_ROOT, "reference", "majca_mesh_graph.npz"))
PANEL_UV_ALL = _g["node_uv"].astype(np.float32)
_flat_ref    = np.load(os.path.join(_ROOT, "reference", "majca_flat_reference_uv.npz"))
FLAT_REF_PTS = _flat_ref["points"].astype(np.float32)


# ── tiled grid of N cloths ────────────────────────────────────────────────────
_cols = int(np.ceil(np.sqrt(N)))
def _tile_xy(e):
    r, c = divmod(e, _cols)
    return np.array([c * args.tile_pitch, r * args.tile_pitch, 0.0], np.float64)
TILE = np.stack([_tile_xy(e) for e in range(N)]).astype(np.float32)   # (N,3) XY offsets, z=0

builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
newton.solvers.SolverStyle3D.register_custom_attributes(builder)
for e in range(N):
    style3d.add_cloth_mesh(
        builder,
        pos=wp.vec3(*_tile_xy(e).tolist()), rot=wp.quat_identity(), vel=wp.vec3(0, 0, 0),
        vertices=V.tolist(), indices=F.tolist(),
        panel_verts=panel.tolist(), panel_indices=uv_indices.reshape(-1).tolist(),
        density=DENSITY, scale=1.0, particle_radius=args.prad,
        tri_aniso_ke=wp.vec3(args.stretch_weft, args.stretch_warp, args.stretch_shear),
        edge_aniso_ke=wp.vec3(args.bend_weft, args.bend_warp, args.bend_shear),
        **({"tri_kd": args.damping} if args.damping is not None else {}),
    )
builder.add_ground_plane()

model = builder.finalize()
model.soft_contact_ke = 1.0e1; model.soft_contact_kd = 1.0e-6; model.soft_contact_mu = 0.8
model.set_gravity((0, 0, -9.81))
solver = newton.solvers.SolverStyle3D(model=model, iterations=args.iters)
solver._precompute(builder)
if getattr(solver, "collision", None) is not None:
    solver.collision.radius    = float(args.self_thick)
    solver.collision.stiff_vf *= args.self_stiff
    solver.collision.stiff_ee *= args.self_stiff
    solver.collision.stiff_ef *= args.self_stiff
    solver.collision.rebuild_bvh(model.particle_q)
print(f"[parallel] {N} cloths × {NV} verts = {N*NV} particles; grid {_cols}×{int(np.ceil(N/_cols))} "
      f"@ {args.tile_pitch}m pitch", flush=True)

s0, s1 = model.state(), model.state()
_q0_all  = s0.particle_q.numpy().copy()              # (N*NV,3) flat spawn (tiled)
_qd0_all = s0.particle_qd.numpy().copy()
_rng = np.random.default_rng()
control, contacts = model.control(), model.contacts()
sim_dt = (1.0 / 60.0) / args.substeps

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
    print("[parallel] CUDA graph captured", flush=True)

def step():
    if _graph: wp.capture_launch(_graph)
    else: simulate()

def env_slice(e):
    return slice(e * NV, (e + 1) * NV)

def verts_world(e):
    return s0.particle_q.numpy()[env_slice(e), :3].astype(np.float32)

def verts_local(e):
    return (verts_world(e) - TILE[e]).astype(np.float32)


# ── Isaac scene: per-env vis mesh + per-env overhead depth camera ─────────────
world = World(physics_dt=1 / 60.0, backend="numpy")
world.scene.add_ground_plane(size=max(25.0, 2 * args.tile_pitch * _cols), color=np.array([0.5, 0.5, 0.5]))
stage = world.scene.stage
UsdGeom.Imageable(stage.GetPrimAtPath("/World/groundPlane")).MakeInvisible()

for path, az in (("/World/SunF", 0.0), ("/World/SunB", 180.0)):
    light = UsdLux.DistantLight.Define(stage, path)
    light.CreateIntensityAttr(8000.0); light.CreateAngleAttr(0.5)
    xf = UsdGeom.Xformable(light.GetPrim()); xf.ClearXformOpOrder()
    xf.AddRotateZOp().Set(az); xf.AddRotateXOp().Set(60.0)

vmeshes, depth_anns, rgb_anns, cam_paths = [], [], [], []
for e in range(N):
    vm = UsdGeom.Mesh.Define(stage, f"/World/garment_vis_{e}")
    vm.GetFaceVertexIndicesAttr().Set(F.tolist())
    vm.GetFaceVertexCountsAttr().Set([3] * (len(F) // 3))
    vm.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy((V + TILE[e]).astype(np.float32)))
    vm.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.20, 0.45, 0.85)]))
    vmeshes.append(vm)

    cpath = f"/World/DepthCamera_{e}"
    cam = UsdGeom.Camera.Define(stage, cpath)
    cam.GetFocalLengthAttr().Set(24.0)
    cam.GetHorizontalApertureAttr().Set(36.0)
    cam.GetVerticalApertureAttr().Set(24.0)
    cam.GetClippingRangeAttr().Set((0.01, 10.0))
    rp = rep.create.render_product(cpath, (CAM_W, CAM_H))
    da = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"); da.attach(rp)
    ra = rep.AnnotatorRegistry.get_annotator("rgb");                      ra.attach(rp)
    depth_anns.append(da); rgb_anns.append(ra); cam_paths.append(cpath)

def place_camera(e, cxy_world):
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(cam_paths[e])); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(cxy_world[0]), float(cxy_world[1]), float(CAM_Z)))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    return np.array([float(cxy_world[0]), float(cxy_world[1]), float(CAM_Z)], dtype=np.float64)

world.reset()
for _ in range(CAM_WARMUP):
    world.step(render=True); rep.orchestrator.step(pause_timeline=False)


# ── render / sim-step helpers ─────────────────────────────────────────────────
def _sync_vis():
    """Push live cloth into every USD vis mesh (headless render reads these — the 'static mesh' bug)."""
    q = s0.particle_q.numpy()
    for e in range(N):
        vmeshes[e].GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(q[env_slice(e), :3].astype(np.float32)))

def _render():
    if args.gui:
        _sync_vis()
    world.step(render=args.gui)

def _ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


# ── pin / drive (operate on GLOBAL particle indices, batched across envs) ──────
def _pin(held_global):
    flags = model.particle_flags.numpy()
    flags[held_global] = flags[held_global] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags.assign(flags); wp.synchronize()

def _unpin(held_global):
    flags = model.particle_flags.numpy()
    flags[held_global] = flags[held_global] | int(ParticleFlags.ACTIVE)
    model.particle_flags.assign(flags); wp.synchronize()

def _drive_batch(targets):
    """targets: list of (held_global, world_xyz (H,3)). One q read/assign for all envs."""
    q = s0.particle_q.numpy()
    for held, world_pos in targets:
        q[held, :3] = world_pos
    s0.particle_q.assign(q); wp.synchronize()


# ── capture: ONE render pass → per-env back-projected DEPTH cloud ──────────────
def capture_all(active_envs):
    """Place every active env's camera, render once, back-project each depth frame to a partial cloud
    in that env's LOCAL frame. Returns {e: {pcd_xyz,normals,centroid,pcd_to_mesh}} (skips envs with
    too few visible points)."""
    _sync_vis()
    for e in active_envs:
        place_camera(e, verts_world(e)[:, :2].mean(0))
    for _ in range(CAM_WARMUP):
        world.step(render=True); rep.orchestrator.step(pause_timeline=False)

    out = {}
    for e in active_envs:
        mesh_local = verts_local(e)                       # NN target in local frame
        centroid   = mesh_local.mean(0).astype(np.float32)
        raw   = depth_anns[e].get_data()
        depth = np.asarray(raw.get("data", raw) if isinstance(raw, dict) else raw).squeeze()
        if e == 0:
            rraw = rgb_anns[0].get_data()
            rgb  = np.asarray(rraw.get("data", rraw) if isinstance(rraw, dict) else rraw)
            if rgb.ndim == 3 and rgb.shape[-1] == 4: rgb = rgb[..., :3]
            try:
                from PIL import Image
                Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(_ROOT, "parallel_capture_env0.png"))
            except Exception:
                pass

        cam_pos = np.array([*verts_world(e)[:, :2].mean(0), CAM_Z], np.float64)
        valid   = np.isfinite(depth) & (depth > 0) & (depth < CAM_Z + 0.05)
        vs, us  = np.where(valid)
        dd = depth[vs, us].astype(np.float64)
        X  = (us - CAM_CX) / CAM_FX * dd
        Y  = -(vs - CAM_CY) / CAM_FY * dd
        # world → local by subtracting the tile offset (XY); z is shared
        pts_world = np.stack([X + cam_pos[0], Y + cam_pos[1], -dd + cam_pos[2]], axis=1)
        pts_local = pts_world - TILE[e]

        nn_d, nn_i = cKDTree(mesh_local).query(pts_local, k=1, workers=-1)
        seen = nn_d < 0.05
        vis_pts, vis_idx = pts_local[seen], nn_i[seen]
        if len(vis_pts) < N_PCD:
            print(f"[parallel] env{e} capture: {len(vis_pts)} < {N_PCD} visible — skip", flush=True)
            continue
        fps = furthest_point_sampling_idx(vis_pts, n_samples=N_PCD)
        out[e] = {
            "pcd_xyz":     (vis_pts[fps] - centroid).astype(np.float32),
            "normals":     estimate_normals((vis_pts[fps] - centroid).astype(np.float32), k=args.normal_k),
            "centroid":    centroid,
            "pcd_to_mesh": vis_idx[fps].astype(np.int32),    # LOCAL vert indices into this env's slice
        }
    return out


# ── reward Φ (reused from head_RL, on the LOCAL slice — translation-invariant) ─
FLAT_W_SHAPE, FLAT_W_FLAT, FLAT_W_COV, FLAT_W_ORIENT, FLAT_W_OOB = 1.0, 0.5, 0.5, 0.3, 0.5
FLAT_GRID = 96
_TABLE_Z  = float(FLAT_REF_PTS[:, 2].mean())

def _align_xy(P, Q):
    Pxy, Qxy = P[:, :2], Q[:, :2]
    pc, qc   = Pxy.mean(0), Qxy.mean(0)
    H = (Pxy - pc).T @ (Qxy - qc)
    U, _, Vt_ = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt_.T @ U.T))
    R = Vt_.T @ np.array([[1, 0], [0, d]], np.float64) @ U.T
    out = P.copy()
    out[:, :2] = (Pxy - pc) @ R.T + qc
    return out, float(abs(np.arctan2(R[1, 0], R[0, 0])))

def _footprint_mask(xy, lo, cell, G):
    ij = np.floor((xy - lo) / cell).astype(np.int32)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < G) & (ij[:, 1] >= 0) & (ij[:, 1] < G)
    m  = np.zeros((G, G), bool)
    m[ij[ok, 1], ij[ok, 0]] = True
    return m

_FLAT_LO   = FLAT_REF_PTS[:, :2].min(0) - 0.05
_FLAT_HI   = FLAT_REF_PTS[:, :2].max(0) + 0.05
_FLAT_CELL = float((_FLAT_HI - _FLAT_LO).max()) / FLAT_GRID
_FLAT_MASK = _footprint_mask(FLAT_REF_PTS[:, :2], _FLAT_LO, _FLAT_CELL, FLAT_GRID)

def flat_reward(e):
    p = verts_local(e)
    pa, orient = _align_xy(p, FLAT_REF_PTS)
    shape  = float(np.mean(np.linalg.norm(pa[:, :2] - FLAT_REF_PTS[:, :2], axis=1)))
    height = float(max(0.0, p[:, 2].mean() - _TABLE_Z))
    cur    = _footprint_mask(pa[:, :2], _FLAT_LO, _FLAT_CELL, FLAT_GRID)
    inter  = np.logical_and(cur, _FLAT_MASK).sum()
    union  = np.logical_or(cur, _FLAT_MASK).sum()
    iou    = float(inter / union) if union else 0.0
    reward = -(FLAT_W_SHAPE * shape + FLAT_W_FLAT * height + FLAT_W_ORIENT * orient) + FLAT_W_COV * iou
    return reward, {"shape": shape, "height": height, "orient": orient, "iou": iou}


# ── workspace box helpers (LOCAL frame, identical to head_RL) ──────────────────
def _clamp_wp(p, centroid):
    cx, cy = float(centroid[0]), float(centroid[1])
    h, bx, by = args.arm_box_half, args.arm_box_cx, args.arm_box_cy
    wx = float(np.clip(cx + p[0], bx - h, bx + h))
    wy = float(np.clip(cy + p[1], by - h, by + h))
    return [wx - cx, wy - cy, float(np.clip(p[2], 0.0, args.arm_box_zmax))]

def _in_box(world_xy):
    h, bx, by = args.arm_box_half, args.arm_box_cx, args.arm_box_cy
    return (np.abs(world_xy[..., 0] - bx) <= h) & (np.abs(world_xy[..., 1] - by) <= h)

def _oob_penalty(waypoints, active, centroid):
    wp_ = np.asarray(waypoints, np.float32)
    ac  = np.asarray(active, np.float32) > 0.5
    if not ac.any():
        return 0.0
    wp_ = wp_[ac]
    h, bx, by = args.arm_box_half, args.arm_box_cx, args.arm_box_cy
    wx, wy = centroid[0] + wp_[:, 0], centroid[1] + wp_[:, 1]
    dx = np.clip(np.abs(wx - bx) - h, 0.0, None)
    dy = np.clip(np.abs(wy - by) - h, 0.0, None)
    dz = np.clip(wp_[:, 2] - args.arm_box_zmax, 0.0, None) + np.clip(-wp_[:, 2], 0.0, None)
    return float((dx + dy + dz).sum())

def student_arm(act, pcd_xyz, centroid):
    """Resolve a policy action → executable arm (LOCAL frame). Same logic as head_RL._student_arm."""
    idx = int(act.get("grab_idx", -1))
    if not (0 <= idx < len(pcd_xyz)):
        idx = 0
    world  = pcd_xyz + centroid
    in_box = _in_box(world[:, :2])
    if in_box[idx]:
        return {"pcd_idx": idx, "release": _clamp_wp(act["release"], centroid),
                "path": [_clamp_wp(p, centroid) for p in act["path"]],
                "path_quat": act.get("path_quat"), "release_quat": act.get("release_quat"),
                "recovery": False}
    ctr = np.array([args.arm_box_cx, args.arm_box_cy], np.float32)
    if in_box.any():
        cand = np.where(in_box)[0]
        ridx = int(cand[np.argmin(np.linalg.norm(world[cand, :2] - world[idx, :2], axis=1))])
    else:
        ridx = int(np.argmin(np.linalg.norm(world[:, :2] - ctr, axis=1)))
    to_ctr = ctr - world[ridx, :2]
    dist   = float(np.linalg.norm(to_ctr))
    tgt_xy = world[ridx, :2] + (to_ctr / (dist + 1e-9)) * min(dist, args.recovery_step)
    release = [float(tgt_xy[0] - centroid[0]), float(tgt_xy[1] - centroid[1]), float(_TABLE_Z)]
    return {"pcd_idx": ridx, "release": release, "path": [], "path_quat": [],
            "release_quat": list(il_dataset.IDENTITY_QUAT), "recovery": True}


# ── rotation helpers (quaternion → matrix, slerp, polyline) — from head_RL ─────
def _quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)]], np.float32)

def _slerp(q0, q1, t):
    q0 = np.asarray(q0, np.float64); q1 = np.asarray(q1, np.float64)
    d  = float(np.dot(q0, q1))
    if d < 0: q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        th0 = np.arccos(np.clip(d, -1, 1)); th = th0 * t
        q = (q0 * np.cos(th)) + ((q1 - q0 * d) / (np.sin(th0) + 1e-9)) * np.sin(th)
    return (q / (np.linalg.norm(q) + 1e-12)).astype(np.float32)

def _polyline_point(pts, s):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    tot = seg.sum()
    if tot < 1e-9: return pts[0].copy()
    d = s * tot; acc = 0.0
    for i in range(len(seg)):
        if acc + seg[i] >= d:
            f = (d - acc) / (seg[i] + 1e-12)
            return (pts[i] + f * (pts[i+1] - pts[i])).astype(np.float32)
        acc += seg[i]
    return pts[-1].copy()

def _polyline_quat(pts, quats, s):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    tot = seg.sum()
    if tot < 1e-9: return np.asarray(quats[0], np.float32)
    d = s * tot; acc = 0.0
    for i in range(len(seg)):
        if acc + seg[i] >= d:
            f = (d - acc) / (seg[i] + 1e-12)
            return _slerp(quats[i], quats[i+1], f)
        acc += seg[i]
    return np.asarray(quats[-1], np.float32)


def build_drag(e, arm, pcd_to_mesh, centroid):
    """Precompute an env's concurrent drag: held verts (global), per-frac world-target closure, n_frames.
    Pure translation when quats are identity (recovery / --no-rot). Reuses head_RL._execute_drag_path math
    but in this env's LOCAL frame, converting to WORLD (adds tile offset) only for the particle targets."""
    p_local = verts_local(e)
    anchor  = int(pcd_to_mesh[arm["pcd_idx"]])
    held_local = np.where(np.linalg.norm(p_local - p_local[anchor], axis=1) < GRAB_RADIUS)[0]
    if len(held_local) == 0:
        held_local = np.array([anchor])
    held_global = held_local + e * NV
    start_world = (p_local[held_local] + TILE[e]).astype(np.float32)
    grab_world  = start_world.mean(0).astype(np.float32)

    cx, cy = float(centroid[0]), float(centroid[1])
    tx, ty = float(TILE[e][0]), float(TILE[e][1])
    to_world = lambda q: np.array([cx + q[0] + tx, cy + q[1] + ty, q[2]], np.float32)
    traj = np.array([grab_world] + [to_world(q) for q in arm["path"]] + [to_world(arm["release"])], np.float32)

    seg_len = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    if seg_len > args.max_drag_len:
        traj = (grab_world + (traj - grab_world) * (args.max_drag_len / seg_len)).astype(np.float32)

    ident = list(il_dataset.IDENTITY_QUAT)
    pq = arm.get("path_quat") or [ident] * len(arm["path"])
    rq = arm.get("release_quat") or ident
    wp_quats = [list(q) for q in pq] + [list(rq)]
    quats = [wp_quats[0]] + wp_quats
    R0 = _quat_to_R(quats[0])

    nrm = estimate_normals(p_local, k=args.normal_k)[held_local].mean(0)
    nrm = (nrm / (np.linalg.norm(nrm) + 1e-9)).astype(np.float32)
    rel = (start_world - (grab_world + ROT_PIVOT_OFFSET * nrm)).astype(np.float32)

    def target(frac):
        ee = _ease(frac)
        T = _polyline_point(traj, ee)
        R = _quat_to_R(_polyline_quat(traj, quats, ee))
        pivot = T + ROT_PIVOT_OFFSET * nrm
        return (pivot + rel @ (R @ R0.T).T).astype(np.float32)

    path_len = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    n_frames = int(np.clip(round(path_len / max(args.drag_speed, 1e-3) * 60.0), 20, 1500))
    return {"held": held_global, "target": target, "n_frames": n_frames}


# ── snapshot / restore (per-env slice) ────────────────────────────────────────
def reset_all():
    for st in (s0, s1):
        q = st.particle_q.numpy();  q[:]  = _q0_all;  st.particle_q.assign(q)
        qd = st.particle_qd.numpy(); qd[:] = _qd0_all; st.particle_qd.assign(qd)
    f = model.particle_flags.numpy()
    f[:] = f[:] | int(ParticleFlags.ACTIVE)
    model.particle_flags.assign(f); wp.synchronize()

def snapshot_all():
    return (s0.particle_q.numpy().copy(), s0.particle_qd.numpy().copy(),
            s1.particle_q.numpy().copy(), s1.particle_qd.numpy().copy(),
            model.particle_flags.numpy().copy())

def restore_all(snap):
    q0, qd0, q1, qd1, flags = snap
    s0.particle_q.assign(q0); s0.particle_qd.assign(qd0)
    s1.particle_q.assign(q1); s1.particle_qd.assign(qd1)
    model.particle_flags.assign(flags); wp.synchronize()


# ── concurrent crumple (all envs at once, one shared step loop) ────────────────
def crumple_all():
    """Per-env random grab-drape-drop, all envs draping simultaneously under one step() loop."""
    q  = s0.particle_q.numpy()
    qd = s1.particle_q.numpy()
    flags = model.particle_flags.numpy()
    held_all = []
    for e in range(N):
        sl   = env_slice(e)
        base = _q0_all[sl]                                       # tiled flat spawn for this env
        centroid_xy = base[:, :2].mean(0)
        angle = float(_rng.uniform(0, 2 * np.pi))
        reach = float(_rng.uniform(0, args.grab_reach))
        gc_xy = centroid_xy + reach * np.array([np.cos(angle), np.sin(angle)])
        grab_r = float(_rng.uniform(args.grab_radius_min, args.grab_radius_max))
        dists  = np.linalg.norm(base[:, :2] - gc_xy, axis=1)
        held_local = np.where(dists <= grab_r)[0]
        if len(held_local) == 0:
            held_local = np.array([int(dists.argmin())])
        lift_dz = args.grab_height - float(base[held_local, 2].mean())
        q[sl, 2]  += lift_dz                                     # lift this env's cloth
        qd[sl, 2] += lift_dz
        hg = held_local + e * NV
        held_all.append(hg)
        flags[hg] &= ~int(ParticleFlags.ACTIVE)
    s0.particle_q.assign(q); s1.particle_q.assign(qd)
    model.particle_flags.assign(flags); wp.synchronize()

    for _ in range(args.drape_frames):
        if not simulation_app.is_running(): return
        step(); _render()

    flags = model.particle_flags.numpy()
    for hg in held_all:
        flags[hg] |= int(ParticleFlags.ACTIVE)
    model.particle_flags.assign(flags); wp.synchronize()

    for _ in range(args.settle):
        if not simulation_app.is_running(): return
        step(); _render()


# ── server client ─────────────────────────────────────────────────────────────
_sock = socket_ipc.connect(args.server_host, args.server_port)
print(f"[parallel] connected to inference server {args.server_host}:{args.server_port}", flush=True)

def server(msg):
    socket_ipc.send_msg(_sock, msg)
    rep_ = socket_ipc.recv_msg(_sock)
    if isinstance(rep_, dict) and rep_.get("error"):
        raise RuntimeError(f"server error: {rep_['error']}")
    return rep_

def infer_batch(states_by_env):
    """states_by_env: ordered dict {e: state}. Returns {e: action} (uv_pred kept per action)."""
    envs   = list(states_by_env.keys())
    states = [states_by_env[e] for e in envs]
    rep_   = server({"op": "infer", "states": states, "greedy": False, "no_rot": args.no_rot})
    return {e: a for e, a in zip(envs, rep_["actions"])}


# ── PPO+GAE rollout: K branches × T turns, all N envs concurrent per turn ──────
EPISODES_PER_UPDATE = args.rl_k
K = max(1, args.rl_group_k)
T = args.rl_turns

buffer = []         # in-memory rollout entries → server "update"
episode = 0
try:
    while simulation_app.is_running():
        episode += 1
        print(f"\n[parallel] ══ episode {episode}  ({N} envs × {K} branch × {T} turns) ══", flush=True)
        reset_all()
        crumple_all()
        snap = snapshot_all()

        for k in range(K):
            if not simulation_app.is_running(): break
            restore_all(snap)
            phi_prev = {e: flat_reward(e)[0] for e in range(N)}      # Φ(s_0) per env
            traj_id  = {e: f"{episode}_{e}_{k}" for e in range(N)}    # one trajectory per (env,branch)

            for t in range(T):
                if not simulation_app.is_running(): break

                # 1) capture all envs (one render pass) → batched inference
                caps = capture_all(list(range(N)))
                if not caps:
                    break
                acts = infer_batch({e: {"pcd_xyz": caps[e]["pcd_xyz"], "normals": caps[e]["normals"],
                                        "centroid": caps[e]["centroid"]} for e in caps})

                # 2) resolve arms + build concurrent drags
                drags, meta = [], {}
                held_to_pin = []
                for e in caps:
                    act      = acts[e]
                    centroid = caps[e]["centroid"]
                    arm = student_arm(act, caps[e]["pcd_xyz"], centroid)
                    dg  = build_drag(e, arm, caps[e]["pcd_to_mesh"], centroid)
                    drags.append((e, dg))
                    held_to_pin.append(dg["held"])
                    meta[e] = {"act": act, "centroid": centroid, "arm": arm}

                if not drags:
                    break
                _pin(np.concatenate(held_to_pin))

                # 3) drive ALL envs' patches concurrently — one step() per frame
                max_fr = max(dg["n_frames"] for _, dg in drags)
                for i in range(max_fr):
                    if not simulation_app.is_running(): break
                    tgts = []
                    for e, dg in drags:
                        frac = min(1.0, (i + 1) / dg["n_frames"])
                        tgts.append((dg["held"], dg["target"](frac)))
                    _drive_batch(tgts); step(); _render()
                # hold at release
                for _ in range(10):
                    if not simulation_app.is_running(): break
                    _drive_batch([(dg["held"], dg["target"](1.0)) for _, dg in drags])
                    step(); _render()

                _unpin(np.concatenate(held_to_pin))
                for _ in range(args.rl_settle):
                    if not simulation_app.is_running(): break
                    step(); _render()

                # 4) per-env reward + buffer entry
                done = (t == T - 1)
                for e in caps:
                    phi, comps = flat_reward(e)
                    act = meta[e]["act"]; centroid = meta[e]["centroid"]
                    oob = _oob_penalty(act["waypoints"], act["active"], centroid)
                    r_t = (phi - phi_prev[e]) - FLAT_W_OOB * oob
                    phi_prev[e] = phi
                    if meta[e]["arm"].get("recovery"):
                        continue                                     # safety override → don't train on it
                    buffer.append({
                        "pcd_xyz":   caps[e]["pcd_xyz"], "normals": caps[e]["normals"],
                        "centroid":  centroid,           "uv_pred": act["uv_pred"],
                        "grab_idx":  act["grab_idx"],    "waypoints": act["waypoints"],
                        "active":    act["active"],      "wp_rot3":  act.get("wp_rot3"),
                        "log_prob":  act["log_prob"],    "reward":   r_t, "phi": phi,
                        "traj":      traj_id[e],         "t": t,     "done": done,
                    })
                phis = [phi_prev[e] for e in caps]
                print(f"[parallel] ep{episode} k{k} turn {t+1}/{T}  Φ μ={np.mean(phis):+.4f} "
                      f"[{np.min(phis):+.3f},{np.max(phis):+.3f}]  buffer={len(buffer)}", flush=True)

        # ── PPO update via the server every rl_k episodes ──
        if episode % EPISODES_PER_UPDATE == 0 and buffer:
            print(f"[parallel] PPO+GAE update on {len(buffer)} grabs ...", flush=True)
            res = server({"op": "update", "entries": buffer,
                          "gamma": args.rl_gamma, "lam": args.rl_lambda, "clip": args.rl_clip,
                          "epochs": args.rl_epochs, "minibatch": args.rl_minibatch,
                          "ent_weight": args.rl_ent_weight, "vf_weight": args.rl_vf_weight,
                          "det_critic": args.rl_det_critic, "phi_target": args.phi_target})
            print(f"[parallel] update L={res['L']:.4f} reward μ={res['reward_mean']:.4f} "
                  f"ret μ={res['ret_mean']:.4f} ({res['n_traj']} trajs × {res['n']} grabs)", flush=True)
            server({"op": "save"})
            buffer = []

except KeyboardInterrupt:
    pass
finally:
    try: _sock.close()
    except Exception: pass
    simulation_app.close()
