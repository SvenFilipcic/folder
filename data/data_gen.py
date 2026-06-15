"""
data_gen.py  —  Newton Style3D garment cloth → training npz dataset.

Simulates a garment drop (optionally over a crumple ball), captures an overhead depth
point cloud, and saves partial + full npz files ready for UV Mapper training.

    OMNI_KIT_ACCEPT_EULA=YES conda run -n fold python data/data_gen.py --gui
    OMNI_KIT_ACCEPT_EULA=YES conda run -n fold python data/data_gen.py --ball --samples 20
"""

import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--gui",          action="store_true", help="open the Isaac window")
parser.add_argument("--samples",      type=int, default=1,  help="number of npz samples per Isaac boot")
parser.add_argument("--drape-frames", type=int, default=50, help="frames to drape over ball before removing it (--ball only)")
parser.add_argument("--settle",       type=int, default=120, help="settle frames after ball removed (or flat settle if no ball)")
parser.add_argument("--substeps", type=int, default=10,  help="Style3D substeps per frame (even for CUDA graph)")
parser.add_argument("--iters",    type=int, default=4,   help="Style3D solver iterations")
parser.add_argument("--mass",     type=float, default=0.2, help="TOTAL garment mass (kg) → density = mass/panel_area")
parser.add_argument("--density",  type=float, default=None, help="override: fabric mass per area (kg/m^2); else derived from --mass")
# NOTE: Style3D's tri_aniso_ke is a continuum coefficient; the example's 1e2 ALREADY = real
# woven fabric. The spec card's 10000/8000/1.5 are mass-spring units and DON'T transfer (1e4
# underconverges → sluggish). Defaults below are the calibrated Style3D real-fabric scale.
parser.add_argument("--stretch-weft", type=float, default=2.0e1, help="tri_aniso_ke weft (Style3D real-fabric scale)")
parser.add_argument("--stretch-warp", type=float, default=2.0e1, help="tri_aniso_ke warp")
parser.add_argument("--stretch-shear", type=float, default=2.0e0, help="tri_aniso_ke shear (soft → natural drape)")
parser.add_argument("--damping",  type=float, default=None, help="tri_kd override (default: builder's 10.0)")
# Bending: Style3D's edge_aniso_ke is a cotangent-dihedral coefficient, NOT N/m. The example's
# real-fabric value is ~2e-5. A mass-spring 'Bend 8000' here = rigid plank. Default to the
# realistic Style3D range; use --bend-spec to force the literal 8000 (see note in chat).
parser.add_argument("--bend-weft", type=float, default=2.0e-5, help="edge_aniso_ke weft (Style3D units; LOW = fabric)")
parser.add_argument("--bend-warp", type=float, default=2.0e-5, help="edge_aniso_ke warp")
parser.add_argument("--bend-shear", type=float, default=5.0e-6, help="edge_aniso_ke shear")
parser.add_argument("--bend-spec", action="store_true", help="force the literal spec bend 8000 into edge_aniso_ke (WARNING: rigid)")
parser.add_argument("--prad",     type=float, default=9.0e-3, help="particle radius (m)")
# grab-drape-drop crumple (default): freeze a patch near centroid, hang, release
parser.add_argument("--grab-reach",      type=float, default=0.20, help="max grab-point offset from cloth centroid (m)")
parser.add_argument("--grab-radius-min", type=float, default=0.03, help="min grab patch radius (m)")
parser.add_argument("--grab-radius-max", type=float, default=0.05, help="max grab patch radius (m)")
parser.add_argument("--grab-height",     type=float, default=0.50, help="height to hold grab point during drape (m)")
# ball (legacy — static collider the cloth drapes over)
parser.add_argument("--ball",     action="store_true", help="use ball-drape crumple instead of grab-drape-drop")
parser.add_argument("--ball-radius", type=float, default=0.08, help="ball radius (m)")
parser.add_argument("--ball-lift", type=float, default=0.35, help="height the ball floats above the floor (m)")
parser.add_argument("--ball-offset", type=float, default=0.0, help="ball horizontal offset from cloth centre (m); 0 = directly below")
parser.add_argument("--out",      type=str, default="style3d_grasp_preview.npz")
# capture → training data (overhead depth point cloud + shadow + 2-panel UV labels)
parser.add_argument("--no-capture", dest="capture", action="store_false",
                    help="skip the depth capture + dataset npz save (pure viewer / preview only)")
parser.add_argument("--outdir",   type=str, default="data/majca", help="dataset dir → writes partial/ + full/")
parser.add_argument("--prefix",   type=str, default="majca", help="npz filename prefix")
parser.add_argument("--idx",      type=int, default=-1, help="sample index for the filename (-1 = auto-next)")
parser.add_argument("--normal-k", type=int, default=30, help="kNN for normal estimation")
args = parser.parse_args()
GUI = args.gui

# ── 1. Isaac FIRST ────────────────────────────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": not GUI, "multi_gpu": False})

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
apply_style3d_nan_patch()   # 0/0 NaN guard for pinned-vs-pinned self-contacts

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH = os.path.join(_ROOT, "assets", "garments", "majca2.usdc")   # full-res garment

# ── overhead depth camera + capture constants ──────────────────────────────────────────
CAM_Z, CAM_W, CAM_H, CAM_WARMUP, N_PCD = 1.0, 640, 480, 5, 4096
CAM_FX = 24.0 / 36.0 * CAM_W          # focal_len/aperture * pixels (matches the USD camera below)
CAM_FY = 24.0 / 24.0 * CAM_H
CAM_CX, CAM_CY = CAM_W / 2.0, CAM_H / 2.0


def srgb_to_linear(rgb):
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float64)
    return np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)


def rgb_to_lab(rgb):
    """(N,3) sRGB[0,1] → (N,3) CIELAB (D65). Pure numpy (no skimage in the Isaac runtime)."""
    lin = srgb_to_linear(rgb)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = (lin @ M.T) / np.array([0.95047, 1.0, 1.08883])
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d ** 2) + 4.0 / 29.0)
    return np.stack([116.0 * f[:, 1] - 16.0,
                     500.0 * (f[:, 0] - f[:, 1]),
                     200.0 * (f[:, 1] - f[:, 2])], axis=1).astype(np.float32)


SHADOW_SCALE = 100.0   # L* spread mapped across shadow's [0,1] (data_gen_rgb value)


def compute_shadow(rgb):
    """(N,3) sRGB[0,1] → shadow (N,1)[0,1]: lightness L*, median-centred + fixed-scaled.
    Median-centring makes it exposure-invariant (survives per-sample light randomisation)."""
    L = rgb_to_lab(np.clip(rgb, 0.0, 1.0))[:, 0]
    return np.clip(0.5 + (L - np.median(L)) / SHADOW_SCALE, 0.0, 1.0).astype(np.float32)[:, None]


def estimate_normals(pts, k=30):
    """(N,3) → (N,3) unit normals via local-PCA over kNN (scipy, no open3d), oriented +z.
    Same method as data/precompute_normals.py / data_gen_rgb's open3d version."""
    k = min(k, len(pts))
    _, idx = cKDTree(pts).query(pts, k=k, workers=-1)        # (N,k)
    nbr = pts[idx]                                            # (N,k,3)
    c = nbr - nbr.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / k                # (N,3,3)
    _, vecs = np.linalg.eigh(cov)                            # ascending eigenvalues
    n = vecs[:, :, 0]                                        # smallest → surface normal
    n[n[:, 2] < 0] *= -1                                     # orient toward the overhead camera (+z)
    return n.astype(np.float32)


def load_usd_mesh(path):
    """First UsdGeom.Mesh → (V cm, flat tri indices, per-vertex UV or None)."""
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
                    faces += [idx[o], idx[o + i], idx[o + i + 1]]
                o += c
            return V, np.array(faces, dtype=np.int32)
    raise ValueError(f"no UsdGeom.Mesh in {path}")


# ── geometry: cm → m, lay flat just above the floor (real scale, NO S trick) ──────────
V, F = load_usd_mesh(MESH)
V *= 0.01
ext = V.max(0) - V.min(0); thin = int(ext.argmin())
order = [i for i in (0, 1, 2) if i != thin] + [thin]
V = V[:, order]
V[:, :2] -= V[:, :2].mean(0)
V[:, 2] -= V[:, 2].min()
_spawn = (args.ball_lift + 2.0 * args.ball_radius + 0.10) if args.ball else 0.02
V[:, 2] += _spawn                              # above the ball (if any), else 2 cm above floor

# ── sewing pattern: xatlas unwrap (reference/majca_panel_xatlas.npz). Style3D needs a valid
# 2D panel (consistent winding, seams) for its anisotropic stretch/bending rest — the shirt's
# flattened XY is NOT valid (front/back overlap, inverted halves). vmapping preserves the
# particle↔vertex correspondence: particle i = original mesh vertex i (= panel_uv[i]).
_panel = np.load(os.path.join(_ROOT, "reference", "majca_panel_xatlas.npz"))
vmapping, uv_indices, uvs = _panel["vmapping"], _panel["indices"], _panel["uvs"].astype(np.float64)
tri3d = vmapping[uv_indices]                    # (T,3) triangles into the 22139 particles
# rescale UVs so total panel area ≈ total 3D area (so the cloth isn't born pre-stretched)
T3 = V[tri3d]; area3d = np.abs(0.5 * np.cross(T3[:, 1] - T3[:, 0], T3[:, 2] - T3[:, 0])[:, 2]).sum() \
    if T3.shape[-1] == 3 else 0.0
Tu = uvs[uv_indices]; e1 = Tu[:, 1] - Tu[:, 0]; e2 = Tu[:, 2] - Tu[:, 0]
areaUV = np.abs(0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])).sum()   # 2D cross = scalar z
uv_scale = float(np.sqrt(area3d / areaUV)) if areaUV > 0 else 1.0
panel = (uvs * uv_scale).astype(np.float32)     # (Nuv,2) panel verts in metres
F = tri3d.reshape(-1).astype(np.int32)          # use the unwrap's 3D triangles for the cloth+vis
panel_area = float(area3d)                       # total panel area ≈ 3D area (rescaled above), m^2
DENSITY = args.density if args.density is not None else (args.mass / panel_area if panel_area > 0 else 0.3)
if args.bend_spec:                               # user insists on the literal spec value
    args.bend_weft = args.bend_warp = args.bend_shear = 8.0e3
print(f"[s3d] garment: {len(V)} verts, {len(tri3d)} tris | panel {panel.shape} uv_scale={uv_scale:.3f} "
      f"| panel_area={panel_area:.3f} m^2 → density={DENSITY:.3f} kg/m^2 (mass {DENSITY*panel_area:.3f} kg)")
print(f"[s3d] stretch(weft,warp,shear)=({args.stretch_weft:g},{args.stretch_warp:g},{args.stretch_shear:g}) "
      f"bend=({args.bend_weft:g},{args.bend_warp:g},{args.bend_shear:g}) "
      f"tri_kd={args.damping if args.damping is not None else 'default(10.0)'}")
if args.bend_spec:
    print("[s3d] WARNING: --bend-spec set → edge_aniso_ke=8000 (mass-spring units). Expect a RIGID plank, "
          "not fabric. This is ~4e8× the Style3D real-fabric value (2e-5).")

# ── 2-panel UV labels (old scheme, label-compatible with uv_mapper_best.pth) ───────────
# node_uv/node_panel are per-particle (vertex i = particle i), so a visible point's particle
# index indexes straight into them — no fragile Z-split recompute.
if args.capture:
    _g = np.load(os.path.join(_ROOT, "reference", "majca_mesh_graph.npz"))
    PANEL_ID_ALL = _g["node_panel"].astype(np.int32)        # (N,) 0=front 1=back
    PANEL_UV_ALL = _g["node_uv"].astype(np.float32)         # (N,2) per-panel UV in [0,1]
    assert len(PANEL_ID_ALL) == len(V), f"labels {len(PANEL_ID_ALL)} != mesh {len(V)} verts"
    OUTDIR   = args.outdir if os.path.isabs(args.outdir) else os.path.join(_ROOT, args.outdir)
    PART_DIR = os.path.join(OUTDIR, "partial"); FULL_DIR = os.path.join(OUTDIR, "full")
    print(f"[s3d] capture ON → {OUTDIR}/{{partial,full}}  "
          f"(labels: front {(PANEL_ID_ALL==0).sum()} / back {(PANEL_ID_ALL==1).sum()})")

# ── 2. Style3D garment cloth ──────────────────────────────────────────────────────────
builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
newton.solvers.SolverStyle3D.register_custom_attributes(builder)
style3d.add_cloth_mesh(
    builder,
    pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
    vertices=V.tolist(), indices=F.tolist(),
    panel_verts=panel.tolist(), panel_indices=uv_indices.reshape(-1).tolist(),
    density=DENSITY, scale=1.0, particle_radius=args.prad,
    tri_aniso_ke=wp.vec3(args.stretch_weft, args.stretch_warp, args.stretch_shear),
    edge_aniso_ke=wp.vec3(args.bend_weft, args.bend_warp, args.bend_shear),
    **({"tri_kd": args.damping} if args.damping is not None else {}),  # else builder default 10.0
)
builder.add_ground_plane()
# static collision ball (cloth drapes over it via soft_contact). is_kinematic → fixed; Style3D
# doesn't integrate bodies so it stays put. density 0 → no gravity.
BALL_C = np.array([args.ball_offset, 0.0, args.ball_lift + args.ball_radius], dtype=np.float64)
if args.ball:
    bbody = builder.add_body(xform=wp.transform(p=wp.vec3(*BALL_C), q=wp.quat_identity()),
                             is_kinematic=True, label="ball")
    bcfg = newton.ModelBuilder.ShapeConfig(); bcfg.density, bcfg.mu = 0.0, 0.5
    builder.add_shape_sphere(bbody, radius=args.ball_radius, cfg=bcfg)

model = builder.finalize()

# soft contact (cloth↔ground + self-contact), Style3D example values
model.soft_contact_radius = 0.35e-2
model.soft_contact_margin  = 0.45e-2
model.soft_contact_ke = 5
model.soft_contact_kd = 1.0e-3
model.soft_contact_mu = 1
model.set_gravity((0.0, 0.0, -9.81))

solver = newton.solvers.SolverStyle3D(model=model, iterations=args.iters)
solver._precompute(builder)
s0, s1 = model.state(), model.state()
# save pre-simulation state for multi-sample reset (must be before any step())
_q0  = s0.particle_q.numpy().copy()
_qd0 = s0.particle_qd.numpy().copy()
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


# CUDA-graph capture of the per-frame substep loop (kills kernel-launch overhead — the speed
# lever this file was missing). The s0/s1 swap replays consistently only for EVEN substeps.
# Grasp writes held positions into s0 IN PLACE before each launch (like the ball-park trick),
# so the graph keeps reading the updated array.
_graph = None
if wp.get_device().is_cuda and args.substeps % 2 == 0:
    simulate(); wp.synchronize()
    with wp.ScopedCapture() as _cap:
        simulate()
    _graph = _cap.graph
    print(f"[s3d] CUDA graph captured (substeps={args.substeps})")
else:
    print(f"[s3d] eager stepping (substeps={args.substeps} odd or CPU)")

# Graph replay reads particle_flags / particle_q from device memory each launch,
# so in-place assign() during the grab phase is picked up — no eager fallback needed.
def step():
    if _graph is not None:
        wp.capture_launch(_graph)
    else:
        simulate()


def verts():
    return s0.particle_q.numpy()[:, :3].astype(np.float32)


# ── 3. Isaac visual mesh + suns (push particle_q here each frame) ──────────────────────
world = World(physics_dt=1 / 60.0, backend="numpy")
world.scene.add_ground_plane(size=25.0, color=np.array([0.5, 0.5, 0.5]))
stage = world.scene.stage
UsdGeom.Imageable(stage.GetPrimAtPath("/World/groundPlane")).MakeInvisible()

vmesh = UsdGeom.Mesh.Define(stage, "/World/garment_vis")
vmesh.GetFaceVertexIndicesAttr().Set(F.tolist())
vmesh.GetFaceVertexCountsAttr().Set([3] * (len(F) // 3))
vmesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
vmesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.20, 0.45, 0.85)]))

# visible ball (matches the collider above) so the GUI shows what the cloth drapes over
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

# overhead depth+rgb camera (data_gen capture): distance_to_image_plane → cloud, rgb → shadow
if args.capture:
    cam_path = "/World/DepthCamera"
    _cam = UsdGeom.Camera.Define(stage, cam_path)
    _cam.GetFocalLengthAttr().Set(24.0)
    _cam.GetHorizontalApertureAttr().Set(36.0)
    _cam.GetVerticalApertureAttr().Set(24.0)
    _cam.GetClippingRangeAttr().Set((0.01, 10.0))
    _rp = rep.create.render_product(cam_path, (CAM_W, CAM_H))
    depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"); depth_ann.attach(_rp)
    rgb_ann   = rep.AnnotatorRegistry.get_annotator("rgb");                      rgb_ann.attach(_rp)

    def place_camera(cxy):
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path)); xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(float(cxy[0]), float(cxy[1]), float(CAM_Z)))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))   # straight down
        return np.array([float(cxy[0]), float(cxy[1]), float(CAM_Z)], dtype=np.float64)

world.reset()

if args.capture:                       # let RTX settle so the first annotator read isn't blank
    for _ in range(CAM_WARMUP):
        world.step(render=True); rep.orchestrator.step(pause_timeline=False)


def pump(i, tag=""):
    p = verts()
    if GUI:
        vmesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(p))
    world.step(render=GUI)
    if i % 30 == 0:
        print(f"  {tag}frame {i:3d}  z∈[{p[:,2].min():.3f},{p[:,2].max():.3f}]")


# ── 4. crumple: grab-drape-drop (default) or ball (legacy --ball) ─────────────────────
def _grab_drape_drop(tag=""):
    """Freeze a random vertex patch near the centroid, hang the cloth, release → crumple."""
    # restore rest pose to both state buffers
    for _st in (s0, s1):
        _q  = _st.particle_q.numpy();  _q[:]  = _q0;  _st.particle_q.assign(_q)
        _qd = _st.particle_qd.numpy(); _qd[:] = _qd0; _st.particle_qd.assign(_qd)

    # pick a random grab center within grab_reach of the cloth centroid (XY)
    centroid_xy = _q0[:, :2].mean(0)
    angle = float(_rng.uniform(0, 2 * np.pi))
    reach = float(_rng.uniform(0, args.grab_reach))
    gc_xy = centroid_xy + reach * np.array([np.cos(angle), np.sin(angle)])

    # find verts within a random grab radius of that center
    grab_r = float(_rng.uniform(args.grab_radius_min, args.grab_radius_max))
    dists  = np.linalg.norm(_q0[:, :2] - gc_xy, axis=1)
    held   = np.where(dists <= grab_r)[0]
    if len(held) == 0:
        held = np.array([int(dists.argmin())])  # always grab at least 1 vert

    # lift all particles so grab verts sit at grab_height
    lift_dz = args.grab_height - float(_q0[held, 2].mean())
    for _st in (s0, s1):
        _q = _st.particle_q.numpy()
        _q[:, 2] += lift_dz
        _st.particle_q.assign(_q)

    # freeze held verts — ACTIVE=False makes Newton treat them as kinematic
    _flags = model.particle_flags.numpy()
    _flags[held] &= ~int(ParticleFlags.ACTIVE)
    model.particle_flags.assign(_flags)
    wp.synchronize()

    print(f"[s3d] {tag}grab {len(held)} verts @ r={grab_r:.3f}m h={args.grab_height:.2f}m, "
          f"drape {args.drape_frames} frames...")
    for i in range(args.drape_frames):
        step(); pump(i, tag + "drape ")

    # release — restore ACTIVE, cloth falls and crumples
    _flags = model.particle_flags.numpy()
    _flags[held] |= int(ParticleFlags.ACTIVE)
    model.particle_flags.assign(_flags)
    wp.synchronize()

    print(f"[s3d] {tag}settling {args.settle} frames...")
    for i in range(args.settle):
        step(); pump(i, tag + "settle ")


def _ball_drape_settle(tag=""):
    """Legacy ball-drape crumple (--ball flag)."""
    print(f"[s3d] {tag}draping {args.drape_frames} frames (ball)...")
    for i in range(args.drape_frames):
        step(); pump(i, tag + "drape ")
    _bq = s0.body_q.numpy()
    _bq[bbody] = [0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 1.0]
    s0.body_q.assign(_bq)
    _xb = UsdGeom.Xformable(stage.GetPrimAtPath("/World/ball_vis"))
    _xb.ClearXformOpOrder(); _xb.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -5.0))
    print(f"[s3d] {tag}settling {args.settle} frames...")
    for i in range(args.settle):
        step(); pump(i, tag + "settle ")


def _crumple(tag=""):
    if args.ball:
        _ball_drape_settle(tag)
    else:
        _grab_drape_drop(tag)


_crumple()

# ── 5. multi-sample capture loop ─────────────────────────────────────────────────────────
def _reset_for_sample(si):
    if args.ball:
        for _st in (s0, s1):
            _q  = _st.particle_q.numpy();  _q[:]  = _q0;  _st.particle_q.assign(_q)
            _qd = _st.particle_qd.numpy(); _qd[:] = _qd0; _st.particle_qd.assign(_qd)
        dx = float(_rng.uniform(-0.12, 0.12))
        dy = float(_rng.uniform(-0.12, 0.12))
        bc = np.array([BALL_C[0] + dx, BALL_C[1] + dy, BALL_C[2]])
        _bq = s0.body_q.numpy(); _bq[bbody] = [*bc, 0.0, 0.0, 0.0, 1.0]; s0.body_q.assign(_bq)
        _xb = UsdGeom.Xformable(stage.GetPrimAtPath("/World/ball_vis"))
        _xb.ClearXformOpOrder(); _xb.AddTranslateOp().Set(Gf.Vec3d(*bc.tolist()))
    _crumple(f"[{si+1}/{args.samples}] ")

for _si in range(args.samples):
    if _si > 0:
        _reset_for_sample(_si)

    p = verts()
    ok = np.isfinite(p).all()
    print(f"[s3d] sample {_si+1}/{args.samples}  {'OK' if ok else 'DIVERGED'}  "
          f"z∈[{p[:,2].min():.3f},{p[:,2].max():.3f}]")
    np.savez(args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out),
             mesh_pts=p, faces=F)

    if args.capture and ok:
        mesh_pts = p
        centroid = mesh_pts.mean(0).astype(np.float32)
        cam_pos  = place_camera(mesh_pts[:, :2].mean(0))
        vmesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh_pts))  # sync before capture
        for _ in range(CAM_WARMUP):
            world.step(render=True); rep.orchestrator.step(pause_timeline=False)

        raw   = depth_ann.get_data()
        depth = np.asarray(raw.get("data", raw) if isinstance(raw, dict) else raw).squeeze()
        rraw  = rgb_ann.get_data()
        rgb   = np.asarray(rraw.get("data", rraw) if isinstance(rraw, dict) else rraw)
        if rgb.ndim == 3 and rgb.shape[-1] >= 3:
            rgb = rgb[..., :3]

        valid  = np.isfinite(depth) & (depth > 0) & (depth < CAM_Z + 0.05)
        vs, us = np.where(valid)
        dd = depth[vs, us].astype(np.float64)
        X  = (us - CAM_CX) / CAM_FX * dd
        Y  = -(vs - CAM_CY) / CAM_FY * dd
        pts_world = np.stack([X + cam_pos[0], Y + cam_pos[1], -dd + cam_pos[2]], axis=1)
        rgb_v = rgb[vs, us].astype(np.float32) / 255.0

        nn_d, nn_i = cKDTree(mesh_pts).query(pts_world, k=1, workers=-1)
        seen = nn_d < 0.05
        vis_pts, vis_idx, vis_rgb = pts_world[seen], nn_i[seen], rgb_v[seen]
        print(f"[s3d] capture: {len(vis_pts)} visible pts ({100*len(vis_pts)/len(mesh_pts):.0f}% of {len(mesh_pts)})")

        if len(vis_pts) < N_PCD:
            print(f"[s3d] ✗ < {N_PCD} visible — too few for a sample, skipping")
        else:
            fps = furthest_point_sampling_idx(vis_pts, n_samples=N_PCD)
            pcd_xyz = (vis_pts[fps] - centroid).astype(np.float32)
            idx     = vis_idx[fps]
            os.makedirs(PART_DIR, exist_ok=True); os.makedirs(FULL_DIR, exist_ok=True)
            n = args.idx if args.idx >= 0 else len(
                [f for f in os.listdir(PART_DIR) if f.startswith(args.prefix) and f.endswith(".npz")])
            fname = f"{args.prefix}_{n:04d}.npz"
            np.savez(os.path.join(PART_DIR, fname),
                     pcd_points = pcd_xyz,
                     shadow     = compute_shadow(vis_rgb[fps]),
                     panel_id   = PANEL_ID_ALL[idx].astype(np.int32),
                     panel_uv   = PANEL_UV_ALL[idx].astype(np.float32),
                     centroid   = centroid,
                     normals    = estimate_normals(pcd_xyz, k=args.normal_k))
            np.savez(os.path.join(FULL_DIR, fname),
                     full_points = (mesh_pts - centroid).astype(np.float32),
                     panel_id    = PANEL_ID_ALL.astype(np.int32),
                     panel_uv    = PANEL_UV_ALL.astype(np.float32),
                     centroid    = centroid)
            print(f"[s3d] ✓ saved {fname} → {PART_DIR} (+ full/)")

if GUI:
    print("[s3d] keep-alive: stepping until you close the window (Ctrl-C to exit)...")
    try:
        i = 0
        while simulation_app.is_running():
            step(); pump(i); i += 1
    except KeyboardInterrupt:
        pass
simulation_app.close()
