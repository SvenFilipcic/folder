"""
data_gen_physx.py  —  PhysX5 PBD particle cloth → training npz dataset.

Cloth crumpling: random tilt (20-70°) + drop from height — no ball needed.
Same capture pipeline as data_gen.py (overhead depth, FPS→4096, panel UV, shadow, normals).

Default stiffness matches spec card:
  stretch 10000  |  bend 8000  |  shear 1.5  |  damping 0.2  |  mass 0.2 kg

    OMNI_KIT_ACCEPT_EULA=YES conda run -n fold python data/data_gen_physx.py --gui
    OMNI_KIT_ACCEPT_EULA=YES conda run -n fold python data/data_gen_physx.py --samples 20
"""

import os, sys, argparse
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--gui",          action="store_true")
parser.add_argument("--samples",      type=int,   default=1)
parser.add_argument("--settle",       type=int,   default=150,    help="settle steps after drop")
parser.add_argument("--drop-h-min",   type=float, default=0.4,    help="min drop height (m)")
parser.add_argument("--drop-h-max",   type=float, default=0.8,    help="max drop height (m)")
parser.add_argument("--tilt-min",     type=float, default=20.0,   help="min tilt angle (deg)")
parser.add_argument("--tilt-max",     type=float, default=70.0,   help="max tilt angle (deg)")
# PhysX cloth stiffness
parser.add_argument("--stretch",  type=float, default=10000.0)
parser.add_argument("--bend",     type=float, default=8000.0)
parser.add_argument("--shear",    type=float, default=11.5)
parser.add_argument("--damping",  type=float, default=0.2)
parser.add_argument("--mass",     type=float, default=0.2,        help="total garment mass (kg)")
parser.add_argument("--iters",    type=int,   default=16,         help="solver position iterations")
parser.add_argument("--friction", type=float, default=0.95)
parser.add_argument("--contact-offset", type=float, default=0.005)
# capture / output
parser.add_argument("--no-capture",  dest="capture", action="store_false")
parser.add_argument("--outdir",  type=str, default="data/majca_physx")
parser.add_argument("--prefix",  type=str, default="majca")
parser.add_argument("--idx",     type=int, default=-1,            help="start index (-1=auto)")
parser.add_argument("--normal-k", type=int, default=30)
args = parser.parse_args()
GUI = args.gui

# ── Isaac FIRST ───────────────────────────────────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": not GUI, "multi_gpu": False})

import numpy as np
import torch
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import ClothPrim, SingleClothPrim, SingleParticleSystem
from isaacsim.core.api.materials.particle_material import ParticleMaterial
from pxr import UsdGeom, UsdLux, Gf
import omni.replicator.core as rep
from scipy.spatial import cKDTree
from utils.pointcloud import furthest_point_sampling_idx

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH   = os.path.join(_ROOT, "assets", "garments", "majca2.usdc")
DEVICE = "cuda:0"

DRESS_SCALE = 0.01   # USD is in cm → m
DRESS_Z     = 0.6    # initial spawn height for settling to rest
CAM_Z, CAM_W, CAM_H, CAM_WARMUP, N_PCD = 1.0, 640, 480, 5, 4096
CAM_FX = 24.0 / 36.0 * CAM_W
CAM_FY = 24.0 / 24.0 * CAM_H
CAM_CX, CAM_CY = CAM_W / 2.0, CAM_H / 2.0

_rng = np.random.default_rng()


# ── helpers ───────────────────────────────────────────────────────────────────────────────
def srgb_to_linear(rgb):
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float64)
    return np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

def rgb_to_lab(rgb):
    lin = srgb_to_linear(rgb)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = (lin @ M.T) / np.array([0.95047, 1.0, 1.08883])
    d = 6.0 / 29.0
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3*d**2) + 4.0/29.0)
    return np.stack([116*f[:,1]-16, 500*(f[:,0]-f[:,1]), 200*(f[:,1]-f[:,2])], axis=1).astype(np.float32)

def compute_shadow(rgb):
    L = rgb_to_lab(np.clip(rgb, 0.0, 1.0))[:, 0]
    return np.clip(0.5 + (L - np.median(L)) / 100.0, 0.0, 1.0).astype(np.float32)[:, None]

def estimate_normals(pts, k=30):
    k = min(k, len(pts))
    _, idx = cKDTree(pts).query(pts, k=k, workers=-1)
    nbr = pts[idx]; c = nbr - nbr.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / k
    _, vecs = np.linalg.eigh(cov)
    n = vecs[:, :, 0]; n[n[:, 2] < 0] *= -1
    return n.astype(np.float32)


# ── UV labels ────────────────────────────────────────────────────────────────────────────
if args.capture:
    _g = np.load(os.path.join(_ROOT, "reference", "majca_mesh_graph.npz"))
    PANEL_ID_ALL = _g["node_panel"].astype(np.int32)
    PANEL_UV_ALL = _g["node_uv"].astype(np.float32)
    OUTDIR   = args.outdir if os.path.isabs(args.outdir) else os.path.join(_ROOT, args.outdir)
    PART_DIR = os.path.join(OUTDIR, "partial")
    FULL_DIR = os.path.join(OUTDIR, "full")
    print(f"[physx] capture ON → {OUTDIR}/{{partial,full}}")


# ── World + PhysX GPU ────────────────────────────────────────────────────────────────────
world = World(physics_dt=1/120.0, backend="torch", device=DEVICE)
world.scene.add_ground_plane(size=25.0, color=np.array([0.5, 0.5, 0.5]))
phys = world.get_physics_context()
phys.enable_gpu_dynamics(True)
phys.set_broadphase_type("GPU")
phys.enable_stablization(True)
stage = world.scene.stage
UsdGeom.Imageable(stage.GetPrimAtPath("/World/groundPlane")).MakeInvisible()

for path, az in (("/World/SunF", 0.0), ("/World/SunB", 180.0)):
    light = UsdLux.DistantLight.Define(stage, path)
    light.CreateIntensityAttr(8000.0); light.CreateAngleAttr(0.5)
    xf = UsdGeom.Xformable(light.GetPrim()); xf.ClearXformOpOrder()
    xf.AddRotateZOp().Set(az); xf.AddRotateXOp().Set(60.0)


# ── PhysX particle cloth ─────────────────────────────────────────────────────────────────
UsdGeom.Xform.Define(stage, "/World/Garment")
particle_material = ParticleMaterial(
    prim_path="/World/Garment/particleMaterial",
    friction=args.friction)
particle_system = SingleParticleSystem(
    prim_path="/World/Garment/particleSystem",
    simulation_owner=world.get_physics_context().prim_path,
    particle_contact_offset=args.contact_offset,
    enable_ccd=True,
    global_self_collision_enabled=True,
    non_particle_collision_enabled=True,
    solver_position_iteration_count=args.iters,
)
add_reference_to_stage(usd_path=MESH, prim_path="/World/Garment/garment")

gxf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Garment/garment"))
gxf.ClearXformOpOrder()
gxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, DRESS_Z))
gxf.AddScaleOp().Set(Gf.Vec3f(DRESS_SCALE, DRESS_SCALE, DRESS_SCALE))

mesh_prim_path = None
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if prim.GetTypeName() == "Mesh" and "groundPlane" not in p:
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if pts and len(pts) > 0:
            mesh_prim_path = p; break
if mesh_prim_path is None:
    simulation_app.close(); raise RuntimeError("no garment mesh prim found")

print(f"[physx] mesh: {mesh_prim_path}  "
      f"stretch={args.stretch} bend={args.bend} shear={args.shear} "
      f"damping={args.damping} mass={args.mass}kg iters={args.iters}")

SingleClothPrim(
    prim_path=mesh_prim_path,
    particle_system=particle_system,
    particle_material=particle_material,
    stretch_stiffness=args.stretch,
    bend_stiffness=args.bend,
    shear_stiffness=args.shear,
    spring_damping=args.damping,
)


# ── Initial settle to rest pose ───────────────────────────────────────────────────────────
world.reset()
cloth = ClothPrim(prim_paths_expr=mesh_prim_path)
_psv = getattr(world, "physics_sim_view", None) or getattr(world, "_physics_sim_view", None)
cloth.initialize(_psv)

def get_particles():
    pts = cloth.get_world_positions()
    if hasattr(pts, "cpu"): pts = pts.cpu().numpy()
    return pts.squeeze(0) if pts.ndim == 3 else pts

def set_particles(pts):
    cloth.set_world_positions(
        torch.as_tensor(pts, dtype=torch.float32, device=DEVICE).unsqueeze(0))

def zero_velocities():
    cloth.set_velocities(torch.zeros_like(cloth.get_velocities()))

# scale to target total mass
try:
    pv = cloth._physics_view
    m  = pv.get_masses()
    scale = args.mass / float(m[0].sum()) if float(m[0].sum()) > 0 else 1.0
    m[0] *= scale
    pv.set_masses(m, torch.tensor([0], dtype=torch.long, device=DEVICE))
    print(f"[physx] mass scaled ×{scale:.3f} → {args.mass:.3f} kg")
except Exception as e:
    print(f"[physx] mass scaling skipped ({e})")

print(f"[physx] initial settle {args.settle} steps...")
for i in range(args.settle):
    world.step(render=GUI)
    if i % 50 == 0:
        z = get_particles()[:, 2]
        print(f"  settle {i:3d}  z∈[{z.min():.3f},{z.max():.3f}]")

_rest_pts = get_particles().copy()
print(f"[physx] rest: {len(_rest_pts)} particles  z∈[{_rest_pts[:,2].min():.3f},{_rest_pts[:,2].max():.3f}]")

if args.capture and len(_rest_pts) != len(PANEL_ID_ALL):
    print(f"[physx] WARNING: particle count {len(_rest_pts)} != label count {len(PANEL_ID_ALL)}")


# ── Overhead camera ───────────────────────────────────────────────────────────────────────
cam_path = "/World/DepthCamera"
cam = UsdGeom.Camera.Define(stage, cam_path)
cam.GetFocalLengthAttr().Set(24.0)
cam.GetHorizontalApertureAttr().Set(36.0)
cam.GetVerticalApertureAttr().Set(24.0)
cam.GetClippingRangeAttr().Set((0.01, 10.0))
rp        = rep.create.render_product(cam_path, (CAM_W, CAM_H))
depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"); depth_ann.attach(rp)
rgb_ann   = rep.AnnotatorRegistry.get_annotator("rgb");                      rgb_ann.attach(rp)

def place_camera(cxy):
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path)); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(cxy[0]), float(cxy[1]), float(CAM_Z)))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    return np.array([float(cxy[0]), float(cxy[1]), float(CAM_Z)], dtype=np.float64)

for _ in range(CAM_WARMUP):
    world.step(render=True); rep.orchestrator.step(pause_timeline=False)


# ── Crumple: tilt + drop ──────────────────────────────────────────────────────────────────
def _crumple_and_settle(tag=""):
    pts = _rest_pts.copy()
    c   = pts.mean(axis=0)

    # random tilt: rotate around a horizontal axis by tilt-min..tilt-max degrees
    angle = _rng.uniform(np.radians(args.tilt_min), np.radians(args.tilt_max))
    axis  = np.array([_rng.uniform(-1, 1), _rng.uniform(-1, 1), 0.0])
    axis /= np.linalg.norm(axis)
    R     = Rotation.from_rotvec(axis * angle).as_matrix()
    pts   = (pts - c) @ R.T + c

    # lift centroid to random drop height above floor
    drop_h = _rng.uniform(args.drop_h_min, args.drop_h_max)
    pts[:, 2] += drop_h - pts[:, 2].mean()

    set_particles(pts)
    zero_velocities()

    print(f"[physx] {tag}drop h={drop_h:.2f}m tilt={np.degrees(angle):.0f}° — settling {args.settle} steps...")
    for i in range(args.settle):
        world.step(render=GUI)
        if i % 50 == 0:
            z = get_particles()[:, 2]
            print(f"  {tag}settle {i:3d}  z∈[{z.min():.3f},{z.max():.3f}]")


# ── Multi-sample loop ────────────────────────────────────────────────────────────────────
_crumple_and_settle()   # sample 0

for _si in range(args.samples):
    if _si > 0:
        _crumple_and_settle(f"[{_si+1}/{args.samples}] ")

    mesh_pts = get_particles()
    ok = np.isfinite(mesh_pts).all()
    print(f"[physx] sample {_si+1}/{args.samples}  {'OK' if ok else 'DIVERGED'}  "
          f"z∈[{mesh_pts[:,2].min():.3f},{mesh_pts[:,2].max():.3f}]")

    if args.capture and ok:
        centroid = mesh_pts.mean(0).astype(np.float32)
        cam_pos  = place_camera(mesh_pts[:, :2].mean(0))
        for _ in range(CAM_WARMUP):
            world.step(render=True); rep.orchestrator.step(pause_timeline=False)

        raw   = depth_ann.get_data()
        depth = np.asarray(raw.get("data", raw) if isinstance(raw, dict) else raw).squeeze()
        rraw  = rgb_ann.get_data()
        rgb   = np.asarray(rraw.get("data", rraw) if isinstance(rraw, dict) else rraw)
        if rgb.ndim == 3 and rgb.shape[-1] >= 3: rgb = rgb[..., :3]

        valid  = np.isfinite(depth) & (depth > 0) & (depth < CAM_Z + 0.05)
        vs, us = np.where(valid); dd = depth[vs, us].astype(np.float64)
        X  = (us - CAM_CX) / CAM_FX * dd
        Y  = -(vs - CAM_CY) / CAM_FY * dd
        pts_world = np.stack([X+cam_pos[0], Y+cam_pos[1], -dd+cam_pos[2]], axis=1)
        rgb_v = rgb[vs, us].astype(np.float32) / 255.0

        nn_d, nn_i = cKDTree(mesh_pts).query(pts_world, k=1, workers=-1)
        seen = nn_d < 0.05
        vis_pts, vis_idx, vis_rgb = pts_world[seen], nn_i[seen], rgb_v[seen]
        print(f"[physx] capture: {len(vis_pts)} visible pts ({100*len(vis_pts)/len(mesh_pts):.0f}%)")

        if len(vis_pts) < N_PCD:
            print(f"[physx] ✗ < {N_PCD} pts — skipping"); continue

        fps     = furthest_point_sampling_idx(vis_pts, n_samples=N_PCD)
        pcd_xyz = (vis_pts[fps] - centroid).astype(np.float32)
        idx     = vis_idx[fps]

        if len(mesh_pts) == len(PANEL_ID_ALL):
            panel_id = PANEL_ID_ALL[idx].astype(np.int32)
            panel_uv = PANEL_UV_ALL[idx].astype(np.float32)
        else:
            _ref = np.load(os.path.join(_ROOT, "reference", "majca_mesh_graph.npz"))
            _ref_pts = _ref.get("node_pos", mesh_pts).astype(np.float32)
            _, _ri   = cKDTree(_ref_pts).query(mesh_pts[idx], k=1, workers=-1)
            panel_id = PANEL_ID_ALL[_ri].astype(np.int32)
            panel_uv = PANEL_UV_ALL[_ri].astype(np.float32)

        os.makedirs(PART_DIR, exist_ok=True); os.makedirs(FULL_DIR, exist_ok=True)
        n = args.idx + _si if args.idx >= 0 else len(
            [f for f in os.listdir(PART_DIR) if f.startswith(args.prefix) and f.endswith(".npz")])
        fname = f"{args.prefix}_{n:04d}.npz"

        np.savez(os.path.join(PART_DIR, fname),
                 pcd_points = pcd_xyz,
                 shadow     = compute_shadow(vis_rgb[fps]),
                 panel_id   = panel_id,
                 panel_uv   = panel_uv,
                 centroid   = centroid,
                 normals    = estimate_normals(pcd_xyz, k=args.normal_k))
        np.savez(os.path.join(FULL_DIR, fname),
                 full_points = (mesh_pts - centroid).astype(np.float32),
                 panel_id    = PANEL_ID_ALL.astype(np.int32) if len(mesh_pts) == len(PANEL_ID_ALL) else panel_id,
                 panel_uv    = PANEL_UV_ALL.astype(np.float32) if len(mesh_pts) == len(PANEL_UV_ALL) else panel_uv,
                 centroid    = centroid)
        print(f"[physx] ✓ saved {fname}")

if GUI:
    print("[physx] keep-alive — Ctrl-C or close window to exit")
    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        pass
simulation_app.close()
