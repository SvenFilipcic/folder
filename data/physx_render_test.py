"""
physx_render_test.py  —  PhysX PBD cloth variant of newton_render_test.py.

Same render/capture bridge (overhead RGB+depth camera, back-project, KDTree-match), but
the cloth PHYSICS is NVIDIA PhysX particle cloth (ParticleSystem + ClothPrim) instead of
Newton — the exact solver folder6000/data/data_gen_rgb.py used (it gave the good folds).
Ported to Isaac Sim 6: omni.isaac.core → isaacsim.core, and the single-prim ClothPrim lost
get/set_world_positions so we drive the cloth through the *view* class isaacsim.core.prims.ClothPrim.

Unlike Newton there is NO manual particle→USD push and NO world-scale (S) trick: PhysX
simulates the real USD mesh in place and Kit renders it directly. We read particle world
positions from the physics tensor view (get_world_positions) for the depth back-projection.

    OMNI_KIT_ACCEPT_EULA=YES conda run -n fold python data/physx_render_test.py            # headless
    OMNI_KIT_ACCEPT_EULA=YES conda run -n fold python data/physx_render_test.py --gui --ball  # watch a drape
"""

import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--gui",      action="store_true", help="open the Isaac window")
parser.add_argument("--usd",      type=str, default="majca2", help="garment USD (bare name → assets/garments/<name>.usdc)")
parser.add_argument("--settle",   type=int, default=150, help="initial settle steps to reach the rest shape")
parser.add_argument("--mass-scale", type=float, default=0.1, help="scale every particle mass (<1 = lighter cloth, more folding)")
parser.add_argument("--stretch",  type=float, default=1.0e5, help="stretch_stiffness (stiff = no rubbery stretch)")
parser.add_argument("--bend",     type=float, default=200.0, help="bend_stiffness (higher = larger, smoother folds)")
parser.add_argument("--shear",    type=float, default=5.0,   help="shear_stiffness")
parser.add_argument("--damping",  type=float, default=2.0,   help="spring_damping")
parser.add_argument("--contact-offset", type=float, default=0.005, help="particle_contact_offset (cloth half-thickness)")
parser.add_argument("--iters",    type=int, default=32, help="solver_position_iteration_count")
parser.add_argument("--friction", type=float, default=0.95, help="particle material friction")
# ball-drape (mirrors data_gen_rgb reset_garment); ball directly below the shirt, despawn fast
parser.add_argument("--ball",     action="store_true", help="drape over an invisible collision ball, then despawn it")
parser.add_argument("--ball-radius", type=float, default=0.08, help="ball radius (m)")
parser.add_argument("--ball-offset", type=float, default=0.0, help="ball horizontal offset from shirt centre (m); 0 = directly below")
parser.add_argument("--ball-lift", type=float, default=0.50, help="height the ball floats above the floor (m)")
parser.add_argument("--drape",    type=int, default=30,  help="frames to fall + drape over the ball before despawning it")
parser.add_argument("--drape-settle", type=int, default=140, help="frames to settle after the ball despawns")
parser.add_argument("--out",      type=str, default="physx_render_preview.npz")
args = parser.parse_args()
GUI = args.gui

# ── 1. Isaac FIRST ───────────────────────────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": not GUI, "multi_gpu": False})

import numpy as np
import torch
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import ClothPrim, SingleClothPrim, SingleParticleSystem
from isaacsim.core.api.materials.particle_material import ParticleMaterial
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Gf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_u = args.usd
GARMENT_USD = _u if os.path.sep in _u else os.path.join(
    _ROOT, "assets", "garments", _u + (".usdc" if not _u.endswith((".usd", ".usdc", ".usda")) else ""))
if not os.path.exists(GARMENT_USD):
    raise FileNotFoundError(f"--usd not found: {GARMENT_USD}")

DRESS_SCALE = 0.01                 # cm → m (same as data_gen DRESS_SCALE)
DRESS_Z     = 0.6                  # spawn height for the initial settle
CAM_Z, CAM_W, CAM_H, CAM_WARMUP = 1.0, 640, 480, 5
DEVICE = "cuda:0"

# ── 2. World + PhysX GPU dynamics (same context flags as data_gen) ────────────────────
world = World(physics_dt=1 / 120.0, backend="torch", device=DEVICE)
world.scene.add_ground_plane(size=25.0, color=np.array([0.5, 0.5, 0.5]))
phys = world.get_physics_context()
phys.enable_gpu_dynamics(True)
phys.set_broadphase_type("GPU")
phys.enable_stablization(True)
stage = world.scene.stage
UsdGeom.Imageable(stage.GetPrimAtPath("/World/groundPlane")).MakeInvisible()

# ── lights: two opposite "suns" (parallel rays → hard shadows that reveal folds) ──────
for path, az in (("/World/SunF", 0.0), ("/World/SunB", 180.0)):
    light = UsdLux.DistantLight.Define(stage, path)
    light.CreateIntensityAttr(8000.0)
    light.CreateAngleAttr(0.5)
    xf = UsdGeom.Xformable(light.GetPrim()); xf.ClearXformOpOrder()
    xf.AddRotateZOp().Set(az); xf.AddRotateXOp().Set(60.0)

# ── 3. PhysX particle cloth (ParticleSystem + ClothPrim), mirrors data_gen GARMENT ────
UsdGeom.Xform.Define(stage, "/World/Garment")
particle_material = ParticleMaterial(prim_path="/World/Garment/particleMaterial", friction=args.friction)
particle_system = SingleParticleSystem(
    prim_path="/World/Garment/particleSystem",
    simulation_owner=world.get_physics_context().prim_path,
    particle_contact_offset=args.contact_offset,
    enable_ccd=True,
    global_self_collision_enabled=True,
    non_particle_collision_enabled=True,
    solver_position_iteration_count=args.iters,
)
add_reference_to_stage(usd_path=GARMENT_USD, prim_path="/World/Garment/garment")

# scale + lift the referenced garment so it settles to its rest shape above the floor
gxf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Garment/garment")); gxf.ClearXformOpOrder()
gxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, DRESS_Z))
gxf.AddScaleOp().Set(Gf.Vec3f(DRESS_SCALE, DRESS_SCALE, DRESS_SCALE))

# auto-detect the garment mesh prim (first non-ground Mesh with points)
mesh_prim_path = None
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if prim.GetTypeName() == "Mesh" and "groundPlane" not in p:
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if pts and len(pts) > 0:
            mesh_prim_path = p; break
if mesh_prim_path is None:
    simulation_app.close(); raise RuntimeError("no garment mesh prim found in USD")
print(f"[physx] garment mesh prim: {mesh_prim_path}  ({GARMENT_USD})")

# AUTHOR the cloth with the single-prim class (applies PhysxParticleClothAPI + stiffness +
# binds the particle system/material). The view class is read-only over existing cloths, so
# we still need this to actually create the particle cloth on the mesh.
_cloth_single = SingleClothPrim(
    prim_path=mesh_prim_path,
    particle_system=particle_system,
    particle_material=particle_material,
    stretch_stiffness=args.stretch,
    bend_stiffness=args.bend,
    shear_stiffness=args.shear,
    spring_damping=args.damping,
)

# ── 4. initial settle → rest shape ────────────────────────────────────────────────────
world.reset()
# now wrap the authored cloth in the VIEW class for tensor get/set_world_positions
cloth = ClothPrim(prim_paths_expr=mesh_prim_path)
_psv = getattr(world, "physics_sim_view", None) or getattr(world, "_physics_sim_view", None)
cloth.initialize(_psv)

if args.mass_scale != 1.0:                       # lighter cloth → gentler fall, more folding
    # ClothPrim.set_particle_masses is broken in this build (calls a missing get_masses), so
    # write the physics tensor view's mass buffer directly (same workaround as data_gen_rgb).
    try:
        pv = cloth._physics_view
        m = pv.get_masses(); m[0] *= args.mass_scale
        pv.set_masses(m, torch.tensor([0], dtype=torch.long, device=DEVICE))
        print(f"[physx] scaled particle mass ×{args.mass_scale} → max {pv.get_masses()[0].max():.3g}")
    except Exception as e:
        print(f"[physx] mass-scale skipped ({type(e).__name__}: {e}) — running at default mass")


def get_particles():
    pts = cloth.get_world_positions()
    if hasattr(pts, "cpu"):
        pts = pts.cpu().numpy()
    return pts.squeeze(0) if pts.ndim == 3 else pts


def set_particles(pts):
    cloth.set_world_positions(torch.as_tensor(pts, dtype=torch.float32, device=DEVICE).unsqueeze(0))


def zero_velocities():
    v = cloth.get_velocities()
    cloth.set_velocities(torch.zeros_like(v))


print(f"[physx] initial settle {args.settle} steps...")
for i in range(args.settle):
    world.step(render=GUI)
    if i % 30 == 0:
        z = get_particles()[:, 2]
        print(f"  settle {i:3d}  z∈[{z.min():.3f},{z.max():.3f}]")

initial_particles = get_particles()
print(f"[physx] particle count: {len(initial_particles)}  "
      f"bounds min={initial_particles.min(0)} max={initial_particles.max(0)}")

# ── ball collider: invisible sphere the cloth drapes over, then collision disabled ────
BALL_PATH = "/World/DropBall"
ball_geom = UsdGeom.Sphere.Define(stage, BALL_PATH)
ball_geom.GetRadiusAttr().Set(1.0)                       # unit sphere, scaled via xform
ball_xf = UsdGeom.Xformable(ball_geom.GetPrim()); ball_xf.ClearXformOpOrder()
ball_t, ball_s = ball_xf.AddTranslateOp(), ball_xf.AddScaleOp()
ball_t.Set(Gf.Vec3d(0.0, 0.0, -10.0)); ball_s.Set(Gf.Vec3f(0.05, 0.05, 0.05))
UsdPhysics.CollisionAPI.Apply(ball_geom.GetPrim())
ball_col = UsdPhysics.CollisionAPI(ball_geom.GetPrim())
ball_col.GetCollisionEnabledAttr().Set(False)
UsdGeom.Imageable(ball_geom.GetPrim()).MakeInvisible()   # never seen by the camera


def set_ball(center_xy, radius, enabled=True):
    ball_t.Set(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), float(radius + args.ball_lift)))
    ball_s.Set(Gf.Vec3f(float(radius), float(radius), float(radius)))
    ball_col.GetCollisionEnabledAttr().Set(bool(enabled))


def disable_ball():
    ball_col.GetCollisionEnabledAttr().Set(False)
    ball_t.Set(Gf.Vec3d(0.0, 0.0, -10.0))


# ── 5. drape sequence (mirrors reset_garment ball-drape branch) ───────────────────────
if args.ball:
    c = initial_particles.mean(0)
    cxy = (float(c[0]) + args.ball_offset, float(c[1]))
    r = args.ball_radius
    drop_h = args.ball_lift + 2.0 * r + 0.10           # centroid start height above floor
    pts = initial_particles.copy()
    pts[:, 2] += drop_h - pts[:, 2].min()              # lift the shirt above the ball top
    set_ball(cxy, r, enabled=True)
    set_particles(pts); zero_velocities()
    print(f"[physx] BALL-DRAPE: {args.drape} drape steps over r={r}m ball at {cxy}...")
    for i in range(args.drape):
        world.step(render=GUI)
    print("[physx] despawning ball → settle...")
    disable_ball(); zero_velocities()
    for i in range(args.drape_settle):
        world.step(render=GUI)
        if i % 30 == 0:
            z = get_particles()[:, 2]
            print(f"  settle {i:3d}  z∈[{z.min():.3f},{z.max():.3f}]")

mesh_pts = get_particles()
if not np.isfinite(mesh_pts).all():
    print("[physx] ✗ cloth diverged (NaN) — lower --stretch/--mass-scale or raise --iters")
    simulation_app.close(); raise SystemExit(1)

# ── 6. overhead depth + rgb capture, back-project, KDTree-match to particles ───────────
import omni.replicator.core as rep

cam_path = "/World/DepthCamera"
cam = UsdGeom.Camera.Define(stage, cam_path)
cam.GetFocalLengthAttr().Set(24.0)
cam.GetHorizontalApertureAttr().Set(36.0)
cam.GetVerticalApertureAttr().Set(24.0)
cam.GetClippingRangeAttr().Set((0.01, 10.0))
CAM_FX, CAM_FY = 24.0 / 36.0 * CAM_W, 24.0 / 24.0 * CAM_H
CAM_CX, CAM_CY = CAM_W / 2.0, CAM_H / 2.0

cxy = mesh_pts[:, :2].mean(0)
cam_xf = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path)); cam_xf.ClearXformOpOrder()
cam_xf.AddTranslateOp().Set(Gf.Vec3d(float(cxy[0]), float(cxy[1]), float(CAM_Z)))
cam_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
cam_pos_w = np.array([float(cxy[0]), float(cxy[1]), float(CAM_Z)])

rp = rep.create.render_product(cam_path, (CAM_W, CAM_H))
depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"); depth_ann.attach(rp)
rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb"); rgb_ann.attach(rp)
for _ in range(CAM_WARMUP):
    world.step(render=True); rep.orchestrator.step(pause_timeline=False)

raw = depth_ann.get_data()
depth = np.asarray(raw.get("data", raw) if isinstance(raw, dict) else raw).squeeze()
raw_rgb = rgb_ann.get_data()
rgb_img = np.asarray(raw_rgb.get("data", raw_rgb) if isinstance(raw_rgb, dict) else raw_rgb)
if rgb_img.ndim == 3 and rgb_img.shape[-1] >= 3:
    rgb_img = rgb_img[..., :3]
print(f"[physx] depth {depth.shape}  rgb {rgb_img.shape}")

valid = np.isfinite(depth) & (depth > 0) & (depth < CAM_Z + 0.05)
vs, us = np.where(valid)
d = depth[vs, us].astype(np.float64)
X = (us - CAM_CX) / CAM_FX * d
Y = -(vs - CAM_CY) / CAM_FY * d
pts_world = np.stack([X + cam_pos_w[0], Y + cam_pos_w[1], -d + cam_pos_w[2]], axis=1)

from scipy.spatial import cKDTree
nn_d, _ = cKDTree(mesh_pts).query(pts_world, k=1, workers=-1)
vis = nn_d < 0.05
pct = 100 * vis.sum() / len(mesh_pts)
print(f"[physx] valid depth px: {valid.sum()} | matched-to-cloth: {vis.sum()} "
      f"({pct:.0f}% of {len(mesh_pts)} particles)")

ARGS_OUT = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
np.savez(ARGS_OUT,
         cloud=pts_world[vis].astype(np.float32),
         rgb=(rgb_img[vs, us][vis].astype(np.float32) / 255.0),
         mesh_pts=mesh_pts, depth=depth.astype(np.float32))
print(f"[physx] saved preview → {ARGS_OUT}")
print("[physx] ✓ BRIDGE WORKS" if vis.sum() > 1000 else "[physx] ✗ too few visible — check camera/scale")

if GUI:
    # keep simulating + rendering so the window stays live (otherwise it looks "frozen" —
    # capture is done, nothing was stepping it). Close the window or Ctrl-C to exit.
    print("[physx] keep-alive: stepping until you close the window (Ctrl-C to exit)...")
    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        pass
simulation_app.close()
