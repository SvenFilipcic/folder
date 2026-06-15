"""
newton_render_test.py  —  de-risk the Newton-physics ↔ Isaac-render BRIDGE.

The plan for the Isaac 5 data_gen port: Newton owns the cloth PHYSICS (VBD solver,
warp 1.14), Isaac Sim 5 owns the PIXELS (overhead RGB + depth camera). This script
proves the bridge end to end on the majca garment:

  1. Isaac SimulationApp first (so warp/newton attach to Isaac's CUDA context),
  2. build a Newton VBD cloth from majca_22139v.obj (same setup as env/newton_test.py),
  3. create a plain visual USD mesh (same verts/faces) + an overhead camera + suns,
  4. each frame: step Newton → copy particle_q into the USD mesh's points → Isaac renders,
  5. after settle: grab distance_to_image_plane + rgb, back-project to a world cloud,
     KDTree-match to the Newton particles, report visible %, and dump a debug npz/PNG.

If visible% is healthy and the saved preview shows the shirt, the physics-agnostic
data_gen pipeline (crumple, FPS→4096, panel UV, save) ports on top unchanged.

    conda run -n folder python data/newton_render_test.py            # headless, saves preview
    conda run -n folder python data/newton_render_test.py --gui      # watch it
"""

import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--gui",      action="store_true", help="open the Isaac window")
parser.add_argument("--settle",   type=int,   default=200, help="newton settle steps before capture")
parser.add_argument("--substeps", type=int,   default=32,  help="newton substeps per frame")
parser.add_argument("--iters",    type=int,   default=20,  help="VBD iterations (stiff cloth needs more)")
parser.add_argument("--spawn",    type=float, default=0.05, help="spawn height above floor (m)")
parser.add_argument("--ke",       type=float, default=1.0e3, help="stretch stiffness (tri_ke)")
parser.add_argument("--bend",     type=float, default=5.0,   help="bend stiffness (edge_ke) — high explodes light cloth")
parser.add_argument("--kd",       type=float, default=0.5,   help="stretch/bend damping")
parser.add_argument("--mass",     type=float, default=0.2,   help="total cloth mass (kg) → density")
parser.add_argument("--cke",      type=float, default=5.0e1, help="ground contact stiffness")
parser.add_argument("--margin",   type=float, default=0.02,  help="soft_contact_margin (m, pre-scale)")
parser.add_argument("--S",        type=float, default=10.0,  help="world scale: sim at S× metres (Newton tolerances like ~tens-of-units; 1m too small). Render divides by S.")
parser.add_argument("--ball",     action="store_true", help="drape the shirt over an invisible collision ball, then despawn it (mirrors data_gen_rgb reset_garment)")
parser.add_argument("--ball-radius", type=float, default=0.08, help="ball radius (m, pre-scale)")
parser.add_argument("--ball-offset", type=float, default=0.0, help="ball horizontal offset from shirt centre (m); 0 = directly below the shirt")
parser.add_argument("--ball-lift", type=float, default=0.50, help="height the ball floats above the floor (m); shirt free-falls this far to the ground after the ball despawns (data_gen DRAPE_LIFT)")
parser.add_argument("--drape",    type=int,   default=30,  help="frames to fall + drape over the ball before despawning it")
parser.add_argument("--drape-settle", type=int, default=140, help="frames to settle after the ball despawns")
parser.add_argument("--rck",      type=float, default=1.0e4, help="VBD rigid_contact_k_start — stiffness for cloth↔sphere (AVBD) contact; default 100 is too soft")
parser.add_argument("--self-contact", action="store_true", help="enable VBD particle self-contact = keep non-connected verts apart (cloth thickness / no self-pass-through). Needs a light mesh.")
parser.add_argument("--scr",      type=float, default=0.01, help="particle_self_contact_radius (m, pre-scale) — cloth half-thickness; keep < particle spacing (~0.012 m on lowpoly)")
parser.add_argument("--scm",      type=float, default=0.02, help="particle_self_contact_margin (m, pre-scale) — detection band, ~2× radius")
parser.add_argument("--sc-interval", type=int, default=5, help="rebuild self-contact BVH every N substeps (rollers uses 5 → faster; 0 = every substep)")
parser.add_argument("--sc-rest-excl", type=float, default=0.03, help="particle_rest_shape_contact_exclusion_radius (m, pre-scale) — exclude verts already this close in the REST pose (sleeves/sides/seams) so self-contact doesn't lock the shirt rigid. 0 = exclude nothing (freezes a garment)")
parser.add_argument("--out",      type=str,   default="newton_render_preview.npz")
args = parser.parse_args()
GUI = args.gui

# ── 1. Isaac FIRST (kit + CUDA context), THEN warp/newton attach to it ──────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": not GUI, "multi_gpu": False})

import numpy as np
import warp as wp
import newton
import carb, carb.settings
from isaacsim.core.api import World
from pxr import Usd, UsdGeom, UsdLux, Gf, Vt
import omni.replicator.core as rep

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ultra-lowpoly majca (~1.8k verts) — lightest mesh → fastest + self-contact-feasible. Swap
# this file for a re-decimated .usdc/.obj and the loader (dispatched on extension) just works.
MESH  = os.path.join(_ROOT, "assets", "garments", "majca_ultralp.usdc")

CAM_Z, CAM_W, CAM_H, CAM_WARMUP = 1.0, 640, 480, 5


def load_obj(path):
    """majca OBJ → (V,3) float verts, flat (M*3,) int tri indices. Handles 'f a//na ...'."""
    verts, faces = [], []
    with open(path) as f:
        for ln in f:
            if ln.startswith("v "):
                verts.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith("f "):
                fi = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
                for i in range(1, len(fi) - 1):
                    faces += [fi[0], fi[i], fi[i + 1]]
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def load_usd_mesh(path):
    """First UsdGeom.Mesh in the stage → (V,3) float verts, flat (M*3,) int tri indices.
    majca USD points are in cm (same as the OBJ), triangulated already (faceVertexCounts=3)."""
    stage = Usd.Stage.Open(path)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            m   = UsdGeom.Mesh(prim)
            V   = np.array(m.GetPointsAttr().Get(), dtype=np.float32)
            cnt = np.array(m.GetFaceVertexCountsAttr().Get())
            idx = np.array(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            faces = []                                  # fan-triangulate any n-gons (cnt may be >3)
            o = 0
            for c in cnt:
                for i in range(1, c - 1):
                    faces += [idx[o], idx[o + i], idx[o + i + 1]]
                o += c
            return V, np.array(faces, dtype=np.int32)
    raise ValueError(f"no UsdGeom.Mesh found in {path}")


def load_mesh(path):
    return load_obj(path) if path.lower().endswith(".obj") else load_usd_mesh(path)


# ── geometry: majca mesh → metres, laid flat just above the floor (same as newton_test) ──
V, F = load_mesh(MESH)
V *= 0.01                                  # cm → m (DRESS_SCALE)
ext   = V.max(0) - V.min(0)
thin  = int(ext.argmin())
order = [i for i in (0, 1, 2) if i != thin] + [thin]   # thinnest axis → z (lay flat)
V = V[:, order]
V[:, :2] -= V[:, :2].mean(0)
V[:, 2]  -= V[:, 2].min()
# ball mode: float the ball high (ball_lift) and spawn the shirt just above its top, so it
# drapes gently, then (after despawn) free-falls ball_lift to the floor — like data_gen_rgb.
spawn_h = (args.ball_lift + 2.0 * args.ball_radius + 0.10) if args.ball else args.spawn
V[:, 2] += spawn_h
Fr   = F.reshape(-1, 3)
area = 0.5 * np.linalg.norm(np.cross(V[Fr[:, 1]] - V[Fr[:, 0]],
                                     V[Fr[:, 2]] - V[Fr[:, 0]]), axis=1).sum()
S = args.S
density = args.mass / area / (S * S)       # keep particle masses constant under scaling
Vs = V * S                                  # build the Newton cloth at S× metres
print(f"[bridge] majca: {len(V)} verts, {len(F)//3} tris | S={S:g} density {density:.3f}")

# Isaac swallows stdout + simulation_app.close() hard-exits (exit 0) → instrument a
# status FILE written before every close path, else M0 gives no signal (isaac6.md gotcha).
import json as _json
_STATUS_PATH = os.path.join(_ROOT, "m0_status.json")
ARGS_OUT = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
_ST = {"n_verts": int(len(V)), "n_tris": int(len(F) // 3), "density": float(density),
       "params": {k: getattr(args, k) for k in ("ke", "bend", "kd", "mass", "iters",
                                                 "substeps", "settle", "cke", "spawn")},
       "settle_z": [], "verdict": "incomplete"}
def dump_status(verdict):
    _ST["verdict"] = verdict
    with open(_STATUS_PATH, "w") as _f:
        _json.dump(_ST, _f, indent=2, default=str)
    print(f"[bridge] status → {_STATUS_PATH}: {verdict}")

# ── 2. Newton VBD cloth at world-scale S (mirrors data/_cloth_debug.py) ──────────────
builder = newton.ModelBuilder(gravity=-9.81 * S)
gcfg = builder.default_shape_cfg.copy()
gcfg.ke, gcfg.kd, gcfg.kf, gcfg.mu = args.cke, 1.0, args.cke * 0.5, 1.0
builder.add_ground_plane(cfg=gcfg)
builder.add_cloth_mesh(
    pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
    vertices=[wp.vec3(*p) for p in Vs], indices=F.tolist(), density=density,
    tri_ke=args.ke, tri_ka=args.ke, tri_kd=args.kd,
    edge_ke=args.bend, edge_kd=args.kd, particle_radius=0.005 * S,
)
# ── ball collider (mirrors data_gen_rgb's drop-ball): kinematic sphere the shirt drapes
#    over, then we "despawn" it by parking it far below the floor (Newton can't toggle a
#    shape's collision mid-run). density=0 → unaffected by gravity. Off-centre → asym folds.
BALL_BODY = -1
ball_r_s  = args.ball_radius * S
ball_cz_s = (args.ball_lift + args.ball_radius) * S    # ball floats ball_lift above the floor
ball_xy_s = np.array([args.ball_offset, 0.0]) * S      # off-centre under the shirt
ball_park_s = [0.0, 0.0, -10.0 * S, 0.0, 0.0, 0.0, 1.0]   # (px,py,pz, qx,qy,qz,qw)
if args.ball:
    BALL_BODY = builder.add_body(                       # is_kinematic → fixed collider that
        xform=wp.transform(p=wp.vec3(float(ball_xy_s[0]), float(ball_xy_s[1]), float(ball_cz_s)),
                           q=wp.quat_identity()), is_kinematic=True, label="drop_ball")
    bcfg = newton.ModelBuilder.ShapeConfig()
    bcfg.density, bcfg.ke, bcfg.kd, bcfg.mu = 0.0, args.rck, 1.0, 0.5
    builder.add_shape_sphere(BALL_BODY, radius=ball_r_s, cfg=bcfg)

builder.color(include_bending=True)
model = builder.finalize()
model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = args.cke, 1.0, 1.0
# cloth↔sphere contact rides VBD's AVBD rigid path → rigid_contact_k_start (NOT soft_contact_ke).
solver = newton.solvers.SolverVBD(
    model=model, iterations=args.iters,
    particle_enable_self_contact=args.self_contact,         # keep non-connected verts apart
    particle_self_contact_radius=args.scr * S, particle_self_contact_margin=args.scm * S,
    particle_collision_detection_interval=args.sc_interval, # rebuild BVH every N substeps
    particle_rest_shape_contact_exclusion_radius=args.sc_rest_excl * S,  # skip rest-touching pairs
    rigid_contact_k_start=args.rck)
if args.self_contact:
    print(f"[bridge] self-contact ON: radius={args.scr*S:.3f} margin={args.scm*S:.3f} "
          f"rest_excl={args.sc_rest_excl*S:.3f} interval={args.sc_interval} (units, S={S:g})")
pipeline = newton.CollisionPipeline(model, broad_phase="sap", soft_contact_margin=args.margin * S)
s0, s1 = model.state(), model.state()
control, contacts = model.control(), pipeline.contacts()
sim_dt = (1.0 / 60.0) / args.substeps


def _simulate():
    global s0, s1
    for _ in range(args.substeps):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, sim_dt)
        s0, s1 = s1, s0          # swap so s0 holds the latest state


def park_ball():
    """Despawn the ball: move its kinematic body far below the floor (collision effectively
    gone). Write into both states' body_q in place so eager stepping reads the parked pose."""
    for st in (s0, s1):
        bq = st.body_q.numpy(); bq[BALL_BODY] = ball_park_s
        st.body_q.assign(bq)


def zero_cloth_velocities():
    for st in (s0, s1):
        qd = st.particle_qd.numpy(); qd[:] = 0.0
        st.particle_qd.assign(qd)


# CUDA-graph capture of the substep loop kills per-kernel launch overhead (~2.9× on the
# 4070). The s0/s1 swap only replays consistently for an EVEN substep count. Ball mode is
# fine WITH capture because park_ball writes body_q IN PLACE (.assign) — the graph keeps
# reading the same (updated) array. (poker_cards disables capture only because it REASSIGNS.)
_graph = None
if wp.get_device().is_cuda and args.substeps % 2 == 0:
    _simulate()                          # warmup: compile kernels before capture
    wp.synchronize()
    with wp.ScopedCapture() as _cap:
        _simulate()
    _graph = _cap.graph
    print(f"[bridge] CUDA graph captured (substeps={args.substeps})")
else:
    print(f"[bridge] eager stepping (substeps={args.substeps} odd or CPU)")


def newton_step():
    if _graph is not None:
        wp.capture_launch(_graph)
    else:
        _simulate()


def newton_verts():
    # particle_q is at S× metres; divide by S so the USD mesh / camera stay in metres
    return (s0.particle_q.numpy()[:, :3] / S).astype(np.float32)


# ── 3. Isaac stage: World (renderer), a VISUAL USD mesh tracking Newton, suns, camera ──
world = World(physics_dt=1 / 60.0, backend="numpy")
world.scene.add_ground_plane(size=25.0, color=np.array([0.5, 0.5, 0.5]))
stage = world.scene.stage
UsdGeom.Imageable(stage.GetPrimAtPath("/World/groundPlane")).MakeInvisible()

CLOTH = "/World/garment_vis"
vmesh = UsdGeom.Mesh.Define(stage, CLOTH)
vmesh.GetFaceVertexIndicesAttr().Set(F.tolist())
vmesh.GetFaceVertexCountsAttr().Set([3] * (len(F) // 3))
vmesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
vmesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.20, 0.45, 0.85)]))

# visual sphere so the drape is legible in the GUI (the physics ball itself is invisible).
# Positioned/hidden in metres; hidden once the ball despawns.
ball_vis = None
if args.ball:
    ball_vis = UsdGeom.Sphere.Define(stage, "/World/drop_ball_vis")
    ball_vis.GetRadiusAttr().Set(float(args.ball_radius))
    bxf = UsdGeom.Xformable(ball_vis.GetPrim()); bxf.ClearXformOpOrder()
    bxf.AddTranslateOp().Set(Gf.Vec3d(float(ball_xy_s[0] / S), float(ball_xy_s[1] / S),
                                      float(ball_cz_s / S)))
    ball_vis.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.85, 0.30, 0.30)]))

for path, az in (("/World/SunF", 0.0), ("/World/SunB", 180.0)):
    light = UsdLux.DistantLight.Define(stage, path)
    light.CreateIntensityAttr(8000.0)
    light.CreateAngleAttr(0.5)
    xf = UsdGeom.Xformable(light.GetPrim()); xf.ClearXformOpOrder()
    xf.AddRotateZOp().Set(az); xf.AddRotateXOp().Set(60.0)

cam_path = "/World/DepthCamera"
cam = UsdGeom.Camera.Define(stage, cam_path)
cam.GetFocalLengthAttr().Set(24.0)
cam.GetHorizontalApertureAttr().Set(36.0)
cam.GetVerticalApertureAttr().Set(24.0)
cam.GetClippingRangeAttr().Set((0.01, 10.0))

CAM_FX = 24.0 / 36.0 * CAM_W
CAM_FY = 24.0 / 24.0 * CAM_H
CAM_CX, CAM_CY = CAM_W / 2.0, CAM_H / 2.0

render_product = rep.create.render_product(cam_path, (CAM_W, CAM_H))
depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"); depth_ann.attach(render_product)
rgb_ann   = rep.AnnotatorRegistry.get_annotator("rgb");                     rgb_ann.attach(render_product)


def place_camera(cxy):
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path)); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(cxy[0]), float(cxy[1]), float(CAM_Z)))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    return np.array([float(cxy[0]), float(cxy[1]), float(CAM_Z)])


world.reset()
print("[bridge] warming renderer...")
for _ in range(CAM_WARMUP):
    world.step(render=True); rep.orchestrator.step(pause_timeline=False)

# ── 4. settle Newton, pushing verts into the visual mesh each frame ─────────────────
def _pump(i, tag=""):
    s0_verts = newton_verts()
    vmesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(s0_verts))
    world.step(render=GUI)
    if i % 30 == 0:
        zmin, zmax = s0_verts[:, 2].min(), s0_verts[:, 2].max()
        _ST["settle_z"].append([i, float(zmin), float(zmax),
                                [float(x) for x in np.ptp(s0_verts[:, :2], 0)]])
        print(f"  {tag}frame {i:3d}  z∈[{zmin:.3f},{zmax:.3f}]  xy_extent={np.ptp(s0_verts[:,:2],0)}")

if args.ball:
    print(f"[bridge] BALL-DRAPE: {args.drape} drape frames over r={args.ball_radius}m ball...")
    for i in range(args.drape):
        newton_step(); _pump(i, "drape ")
    print("[bridge] despawning ball → settle...")
    park_ball(); zero_cloth_velocities()
    if ball_vis is not None:
        UsdGeom.Imageable(ball_vis.GetPrim()).MakeInvisible()
    for i in range(args.drape_settle):
        newton_step(); _pump(i, "settle ")
else:
    print(f"[bridge] settling {args.settle} Newton frames...")
    for i in range(args.settle):
        newton_step(); _pump(i)

mesh_pts = newton_verts()
if not np.isfinite(mesh_pts).all():
    print("[bridge] ✗ Newton cloth diverged (NaN/inf) — tune --bend/--iters/--substeps")
    dump_status("DIVERGED")
    simulation_app.close(); raise SystemExit(1)

# ── 5. capture overhead depth + rgb, back-project, match to Newton particles ─────────
cam_pos_w = place_camera(mesh_pts[:, :2].mean(0))
for _ in range(CAM_WARMUP):
    world.step(render=True); rep.orchestrator.step(pause_timeline=False)

raw   = depth_ann.get_data()
depth = np.asarray(raw.get("data", raw) if isinstance(raw, dict) else raw).squeeze()
raw_rgb = rgb_ann.get_data()
rgb_img = np.asarray(raw_rgb.get("data", raw_rgb) if isinstance(raw_rgb, dict) else raw_rgb)
if rgb_img.ndim == 3 and rgb_img.shape[-1] >= 3:
    rgb_img = rgb_img[..., :3]

print(f"[bridge] depth {depth.shape}  rgb {rgb_img.shape}")
_ST["depth_shape"] = list(depth.shape); _ST["rgb_shape"] = list(rgb_img.shape)
_ST["mesh_z"] = [float(mesh_pts[:, 2].min()), float(mesh_pts[:, 2].max())]
valid = np.isfinite(depth) & (depth > 0) & (depth < CAM_Z + 0.05)
vs, us = np.where(valid)
d = depth[vs, us].astype(np.float64)
X = (us - CAM_CX) / CAM_FX * d
Y = -(vs - CAM_CY) / CAM_FY * d
pts_world = np.stack([X + cam_pos_w[0], Y + cam_pos_w[1], -d + cam_pos_w[2]], axis=1)

from scipy.spatial import cKDTree
nn_d, nn_i = cKDTree(mesh_pts).query(pts_world, k=1, workers=-1)
vis = nn_d < 0.05
pct = 100 * vis.sum() / len(mesh_pts)
print(f"[bridge] valid depth px: {valid.sum()} | matched-to-cloth: {vis.sum()} "
      f"({pct:.0f}% of {len(mesh_pts)} particles)")

np.savez(ARGS_OUT,
         cloud=pts_world[vis].astype(np.float32),
         rgb=(rgb_img[vs, us][vis].astype(np.float32) / 255.0),
         mesh_pts=mesh_pts, depth=depth.astype(np.float32))
print(f"[bridge] saved preview → {ARGS_OUT}")
print("[bridge] ✓ BRIDGE WORKS" if vis.sum() > 1000 else "[bridge] ✗ too few visible — check camera/scale")

_ST["valid_depth_px"] = int(valid.sum())
_ST["matched_particles"] = int(vis.sum())
_ST["visible_pct"] = float(pct)
_ST["preview_npz"] = ARGS_OUT
dump_status("BRIDGE_WORKS" if vis.sum() > 1000 else "TOO_FEW_VISIBLE")

if GUI and sys.stdin.isatty():
    input("[bridge] Enter to close...")   # only block when run in a real terminal (else EOFError)
simulation_app.close()
