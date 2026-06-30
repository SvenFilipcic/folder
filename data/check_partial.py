"""
Quick viewer for partial npz files — checks occlusion quality.

Shows partial (blue, left) + full (red, right) side by side by default.
Press Q to close the window and advance to the next sample.

Usage:
    python3 scripts/check_partial.py                   # random sample
    python3 scripts/check_partial.py --idx 5           # npz with number 5 (majca_0005.npz)
    python3 scripts/check_partial.py --idx 5 --uv      # colour by UV (default colour = RGB if present)
    python3 scripts/check_partial.py --idx 5 --rgb     # ALL variants side by side: RGB | color_dev | shadow | UV
    python3 scripts/check_partial.py --idx 5 --color-dev   # colour by color_dev (what the model sees)
    python3 scripts/check_partial.py --idx 5 --shadow      # colour by shadow (relative L* shading)
    python3 scripts/check_partial.py --all             # step through all  (Q → next)
    python3 scripts/check_partial.py --partial-only    # hide full cloud
"""

import sys, os, argparse, glob, random
os.environ["WAYLAND_DISPLAY"]  = ""
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ.setdefault("DISPLAY", ":0")
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("--idx",          type=int,  default=None)
parser.add_argument("--file",         type=str,  default=None, help="specific npz filename, e.g. majca_9964.npz")
parser.add_argument("--uv",           action="store_true", help="colour by UV")
parser.add_argument("--color-dev",    action="store_true", help="colour by color_dev channel (what the model sees — prints/odd-colour regions bright)")
parser.add_argument("--shadow",       action="store_true", help="colour by shadow channel (relative L* shading)")
parser.add_argument("--rgb",          action="store_true", help="show ALL channel variants of the partial cloud side by side (RGB | color_dev | shadow | UV)")
parser.add_argument("--partial-only", action="store_true", help="hide full cloud")
parser.add_argument("--all",          action="store_true", help="step through every sample")
parser.add_argument("--data",         type=str,  default=None)
parser.add_argument("--reference",    action="store_true", help="show flat reference coloured by UV")
args = parser.parse_args()

data_root   = args.data or os.environ.get("MAJCA_DATA", os.path.join(_root, "data", "majca"))
# accept either the depth-only set (partial/ full/) or the RGB set (partial_rgb/ full_rgb/).
# also accept --data pointed straight at a partial[_rgb] dir. Prefer RGB when both exist.
if os.path.basename(data_root.rstrip("/")).startswith("partial"):
    partial_dir = data_root
    full_dir    = data_root.replace("partial", "full")
    _setname    = "rgb" if "partial_rgb" in partial_dir else "depth"
elif os.path.isdir(os.path.join(data_root, "partial_rgb")):
    partial_dir, full_dir, _setname = (os.path.join(data_root, "partial_rgb"),
                                       os.path.join(data_root, "full_rgb"), "rgb")
else:
    partial_dir, full_dir, _setname = (os.path.join(data_root, "partial"),
                                       os.path.join(data_root, "full"), "depth")
print(f"Dataset set: {_setname}   partial={partial_dir}")

files = sorted(glob.glob(os.path.join(partial_dir, "*.npz")))
if not files:
    sys.exit(f"No npz files found in {partial_dir}")

if args.file:
    target = os.path.join(partial_dir, args.file)
    if target not in files:
        sys.exit(f"{args.file} not found in {partial_dir}")
    indices = [files.index(target)]
elif args.all:
    indices = list(range(len(files)))
elif args.idx is not None:
    # match by the NUMBER in the filename (majca_0007.npz → 7), not the list position
    match = None
    for j, f in enumerate(files):
        try:
            if int(os.path.basename(f).split("_")[1].split(".")[0]) == args.idx:
                match = j
                break
        except (IndexError, ValueError):
            continue
    if match is None:
        sys.exit(f"No npz with number {args.idx} in {partial_dir}")
    indices = [match]
else:
    indices = [random.randint(0, len(files) - 1)]


def scalar_to_rgb(s):
    """(N,) or (N,1) scalar in [0,1] → (N,3) 'inferno' heatmap (dark=0, bright=1)."""
    import matplotlib.cm as cm
    return cm.get_cmap("inferno")(np.clip(np.asarray(s).ravel(), 0, 1))[:, :3]


def make_pcd(pts, colour=None, uv=None, rgb=None, scalar=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if scalar is not None:
        pcd.colors = o3d.utility.Vector3dVector(scalar_to_rgb(scalar).astype(np.float64))
    elif rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb[:, :3], 0, 1).astype(np.float64))
    elif uv is not None:
        col = np.stack([uv[:, 0], uv[:, 1], np.zeros(len(uv))], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(col)
    elif colour is not None:
        pcd.paint_uniform_color(colour)
    return pcd


def make_plane(geoms_pts, z, pad_frac=0.1):
    """Flat/unlit grey reference plane at height z, spanning the XY extent of all
    given point arrays (a list of (N,3) arrays)."""
    allpts = np.concatenate(geoms_pts, axis=0)
    xy_min, xy_max = allpts[:, :2].min(0), allpts[:, :2].max(0)
    pad = pad_frac * (xy_max - xy_min + 1e-6)
    x0, y0 = xy_min - pad
    x1, y1 = xy_max + pad
    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(np.array(
        [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], np.float64))
    plane.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], np.int32))
    plane.paint_uniform_color([0.5, 0.5, 0.5])   # flat/unlit grey (no vertex normals)
    return plane


def show(geoms, title):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=1400, height=700)
    for g in geoms:
        vis.add_geometry(g)
    vis.register_key_callback(ord("Q"), lambda v: v.close())
    vis.run()
    vis.destroy_window()


def show_uv_2d(panel_uv, panel_id, name, full_uv=None, full_panel_id=None):
    import matplotlib.pyplot as plt
    uv  = full_uv       if full_uv       is not None else panel_uv
    pid = full_panel_id if full_panel_id is not None else panel_id
    K   = 128
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"UV — {name}  ({'full' if full_uv is not None else 'partial'})", fontsize=11)
    for pidx, (ax, label) in enumerate(zip(axes, ["Front (panel 0)", "Back (panel 1)"])):
        mask  = pid == pidx
        img   = np.zeros((K, K, 3), dtype=np.float32)
        ub    = np.clip((uv[mask, 0] * (K - 1)).astype(int), 0, K - 1)
        vb    = np.clip((uv[mask, 1] * (K - 1)).astype(int), 0, K - 1)
        img[vb, ub, 0] = uv[mask, 0]   # red   = U
        img[vb, ub, 1] = uv[mask, 1]   # green = V
        ax.imshow(img, origin="lower", extent=[0, 1, 0, 1], interpolation="nearest")
        ax.set_title(label)
        ax.set_xlabel("U")
        ax.set_ylabel("V")
    plt.tight_layout()
    plt.show()


if args.reference:
    garment = os.path.basename(data_root.rstrip("/"))   # e.g. "hoodie" or "majca"
    ref_path = os.path.join(_root, "reference", f"{garment}_flat_reference_uv.npz")
    if not os.path.exists(ref_path):
        sys.exit(f"No flat reference found at {ref_path}\n"
                 f"Run: python data/data_gen_uv.py --garment {garment} --samples 1")
    d    = np.load(ref_path)
    pts  = d["points"] + d["centroid"]
    Z_SPLIT  = 0.015
    panel_id = np.where(pts[:, 2] >= Z_SPLIT, 0, 1).astype(np.int32)
    bounds = {}
    for pid in [0, 1]:
        xy = pts[panel_id == pid, :2]
        bounds[pid] = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    uv = np.zeros((len(pts), 2), dtype=np.float32)
    for pid in [0, 1]:
        mask = panel_id == pid
        xmin, xmax, ymin, ymax = bounds[pid]
        uv[mask, 0] = (pts[mask, 0] - xmin) / (xmax - xmin)
        uv[mask, 1] = (pts[mask, 1] - ymin) / (ymax - ymin)
    rgb = np.stack([uv[:, 0], uv[:, 1], np.zeros(len(pts))], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    print(f"Flat reference: {ref_path}  ({len(pts)} particles)")
    print(f"  front: {(panel_id==0).sum()}  back: {(panel_id==1).sum()}")
    plane = make_plane([pts], z=0.0)   # reference adds centroid back → table at z=0
    show([pcd, plane], f"{garment} flat reference — red=U  green=V  (Q to close)")
    show_uv_2d(uv, panel_id, f"{garment} flat reference")
    sys.exit(0)

for i in indices:
    path = files[i]
    name = os.path.basename(path)
    d    = np.load(path)
    pts      = d["pcd_points"]
    panel_uv = d["panel_uv"]
    panel_id = d["panel_id"]
    pcd_rgb  = d["pcd_rgb"]   if "pcd_rgb"   in d.files else None
    color_dev = d["color_dev"] if "color_dev" in d.files else None
    shadow    = d["shadow"]    if "shadow"    in d.files else None

    print(f"\n[{i}/{len(files)-1}]  {name}")
    print(f"  partial  : {pts.shape}  z [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
    print(f"  panel_uv : u [{panel_uv[:,0].min():.3f}, {panel_uv[:,0].max():.3f}]"
          f"  v [{panel_uv[:,1].min():.3f}, {panel_uv[:,1].max():.3f}]")
    print(f"  front: {(panel_id==0).sum()}  back: {(panel_id==1).sum()}"
          f"  rgb: {'yes' if pcd_rgb is not None else 'no'}")

    z_table = -float(d["centroid"][2]) if "centroid" in d.files else 0.0

    # --rgb: lay every available channel variant of the SAME partial cloud side by side.
    if args.rgb:
        variants = []
        if pcd_rgb   is not None: variants.append(("RGB",       make_pcd(pts, rgb=pcd_rgb)))
        if color_dev is not None: variants.append(("color_dev", make_pcd(pts, scalar=color_dev)))
        if shadow    is not None: variants.append(("shadow",    make_pcd(pts, scalar=shadow)))
        variants.append(("UV", make_pcd(pts, uv=panel_uv)))
        dx    = (pts.max(axis=0) - pts.min(axis=0))[0] * 1.4
        geoms = []
        for k, (_, g) in enumerate(variants):
            g.translate([k * dx, 0, 0])
            geoms.append(g)
        geoms.append(make_plane([np.asarray(g.points) for g in geoms], z=z_table))
        order = "  |  ".join(lbl for lbl, _ in variants)
        print(f"  side-by-side L→R: {order}")
        show(geoms, f"{name} — {order}  (Q to close)")
        continue

    # default colouring: RGB if the npz has it, else solid blue. --uv/--color-dev/--shadow override.
    geoms = []
    if args.color_dev and color_dev is not None:
        geoms.append(make_pcd(pts, scalar=color_dev))
        print(f"  color_dev: frac>0.1={float((color_dev.ravel()>0.1).mean()):.3f}  max={float(color_dev.max()):.3f}")
    elif args.shadow and shadow is not None:
        geoms.append(make_pcd(pts, scalar=shadow))
    elif args.uv:
        geoms.append(make_pcd(pts, uv=panel_uv))
    elif pcd_rgb is not None:
        geoms.append(make_pcd(pts, rgb=pcd_rgb))
    else:
        geoms.append(make_pcd(pts, colour=[0.3, 0.3, 1.0]))

    _full_uv = _full_panel_id = None
    if not args.partial_only:
        full_path = os.path.join(full_dir, name)
        if os.path.exists(full_path):
            fd       = np.load(full_path)
            full_pts = fd["full_points"]
            full_uv  = fd["panel_uv"]
            _full_uv       = full_uv
            _full_panel_id = fd["panel_id"]
            offset   = (pts.max(axis=0) - pts.min(axis=0))[0] * 1.5
            pcd_full = make_pcd(full_pts, uv=full_uv) if args.uv else make_pcd(full_pts, colour=[1.0, 0.2, 0.2])
            pcd_full.translate([offset, 0, 0])
            geoms.append(pcd_full)
            print(f"  full     : {full_pts.shape}")
        else:
            print("  [no matching full file]")

    # grey reference plane at the table level. pts are centroid-normalised, so the
    # table (z_table=0) sits at z=-centroid[2] (computed above); span every cloud in geoms.
    plane_pts = [np.asarray(g.points) for g in geoms]
    geoms.append(make_plane(plane_pts, z=z_table))

    remaining = len(indices) - indices.index(i) - 1
    title = f"{name}  [{i}/{len(files)-1}]"
    title += f"  — Q → next  ({remaining} left)" if remaining else "  — Q to close"

    show(geoms, title)

