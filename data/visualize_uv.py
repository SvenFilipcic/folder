"""
Visualize UV Mapper predictions vs ground truth.

Left  (blue tones) : input partial cloud coloured by PREDICTED UV
Right (red tones)  : input partial cloud coloured by GT UV
Panel side shown as brightness: front=bright, back=dim

Inference runs in the 'infer' conda env (spconv) via subprocess.
Visualization runs here in 'fold' (open3d, no spconv needed).

Usage:
    python3 data/visualize_uv.py --checkpoint checkpoints/uv_mapper_best.pth
    python3 data/visualize_uv.py --checkpoint checkpoints/uv_mapper_best.pth --idx 1
    python3 data/visualize_uv.py --checkpoint checkpoints/uv_mapper_best.pth --all
"""

import sys, os, argparse, glob, random, subprocess, tempfile
os.environ["WAYLAND_DISPLAY"]  = ""
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ.setdefault("DISPLAY", ":0")
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",  type=str, required=True)
parser.add_argument("--data",        type=str, default="data/majca")
parser.add_argument("--idx",         type=int, default=None)
parser.add_argument("--all",         action="store_true")
parser.add_argument("--infer-env",   type=str, default="infer",
                    help="conda env that has spconv (default: infer)")
parser.add_argument("--device",      type=str, default=None)
args = parser.parse_args()

partial_dir  = os.path.join(_root, args.data, "partial")
checkpoint   = args.checkpoint if os.path.isabs(args.checkpoint) \
               else os.path.join(_root, args.checkpoint)
worker       = os.path.join(_root, "data", "_uv_predict_worker.py")

files = sorted(glob.glob(os.path.join(partial_dir, "*.npz")))
if not files:
    sys.exit(f"No npz files in {partial_dir}")

if args.all:
    indices = list(range(len(files)))
elif args.idx is not None:
    indices = [args.idx]
else:
    indices = [random.randint(0, len(files) - 1)]


def run_inference(npz_path, out_path):
    cmd = ["conda", "run", "-n", args.infer_env, "--no-capture-output",
           "python", worker,
           "--npz",        npz_path,
           "--checkpoint", checkpoint,
           "--out",        out_path]
    if args.device:
        cmd += ["--device", args.device]
    ret = subprocess.run(cmd, cwd=_root)
    return ret.returncode == 0


def uv_to_rgb(uv, panel_id, predicted=True):
    u = uv[:, 0]; v = uv[:, 1]
    brightness = np.where(panel_id == 0, 1.0, 0.5)
    if predicted:
        rgb = np.stack([np.zeros_like(u), v * brightness, u * brightness], axis=1)
    else:
        rgb = np.stack([u * brightness, v * brightness * 0.5, np.zeros_like(u)], axis=1)
    return np.clip(rgb, 0, 1)


def make_pcd(pts, rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def show(geoms, title):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=1600, height=800)
    for g in geoms:
        vis.add_geometry(g)
    vis.register_key_callback(ord("Q"), lambda v: v.close())
    vis.run()
    vis.destroy_window()


for i in indices:
    path = files[i]
    name = os.path.basename(path)
    d    = np.load(path)

    pts      = d["pcd_points"].astype(np.float32)
    gt_uv    = d["panel_uv"].astype(np.float32)
    gt_panel = d["panel_id"].astype(np.int64)

    print(f"\n[{i}] {name}  — running inference in '{args.infer_env}' env ...")

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        tmp_out = tf.name

    try:
        ok = run_inference(path, tmp_out)
        if not ok:
            print(f"  ✗ inference failed for {name} — skipping")
            continue

        pred = np.load(tmp_out)
        pred_uv   = pred["uv_pred"].astype(np.float32)    # (N, 2)
        conf_pred = pred["conf_pred"].astype(np.float32)  # (N,)
    finally:
        os.unlink(tmp_out)

    # infer panel from UV (front=panel 0, back=panel 1 — use GT panel for coloring accuracy)
    pred_panel = gt_panel  # for coloring; model outputs UV not panel directly

    # ── accuracy ─────────────────────────────────────────────────────────────────
    K = 128
    gt_u_bins = np.clip(np.round(gt_uv[:, 0]   * (K-1)).astype(int), 0, K-1)
    gt_v_bins = np.clip(np.round(gt_uv[:, 1]   * (K-1)).astype(int), 0, K-1)
    pr_u_bins = np.clip(np.round(pred_uv[:, 0] * (K-1)).astype(int), 0, K-1)
    pr_v_bins = np.clip(np.round(pred_uv[:, 1] * (K-1)).astype(int), 0, K-1)

    u_acc = (pr_u_bins == gt_u_bins).mean()
    v_acc = (pr_v_bins == gt_v_bins).mean()
    u_k3  = (np.abs(pr_u_bins - gt_u_bins) <= 3).mean()
    v_k3  = (np.abs(pr_v_bins - gt_v_bins) <= 3).mean()

    print(f"  u_acc={u_acc:.3f}  v_acc={v_acc:.3f}  "
          f"u_±3={u_k3:.3f}  v_±3={v_k3:.3f}  "
          f"conf_mean={conf_pred.mean():.3f}")

    # ── visualise ────────────────────────────────────────────────────────────────
    offset   = (pts.max(axis=0) - pts.min(axis=0))[0] * 1.5
    pcd_pred = make_pcd(pts, uv_to_rgb(pred_uv, pred_panel, predicted=True))
    pcd_gt   = make_pcd(pts, uv_to_rgb(gt_uv,   gt_panel,   predicted=False))
    pcd_gt.translate([offset, 0, 0])

    remaining = len(indices) - indices.index(i) - 1
    title = f"PRED (blue) | GT (red) — {name} — Q to {'next' if remaining else 'close'}"
    show([pcd_pred, pcd_gt], title)
