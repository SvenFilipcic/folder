"""
One-time: add per-point surface normals to existing UV-Mapper partial npz files.

Normals are estimated from `pcd_points` via local-PCA over k nearest neighbours
(open3d), then oriented to point toward the overhead camera (+z), unit length.
NO Isaac Sim / re-simulation needed — normals are a pure function of the existing
point geometry, so the current 15k samples just get the new key added.

The UV Mapper feeds normals as a separate, zero-init feature branch
(see model/uv_mapper.py) so they cannot swamp the xyz signal.

Rewrites each partial/*.npz IN PLACE (atomic: tmp file in same dir → os.replace),
adding a 'normals' (N,3) float32 key. Idempotent — skips files that already have
it unless --force.

    python data/precompute_normals.py --data HDD_data/data/majca
    python data/precompute_normals.py --data data/majca --k 30 --force
"""

import sys, os, argparse, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import open3d as o3d
from tqdm import tqdm
from config import resolve_data_dir


def estimate_normals(pts, k=30):
    """pts: (N,3) float → (N,3) float32 unit normals oriented toward +z (overhead
    camera). Normal estimation is translation-invariant, so the centroid-normalised
    cloud is fine to feed directly. Used here and (later) by inference."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    return np.asarray(pcd.normals, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="data dir containing partial/ (default: resolve_data_dir('majca'))")
    ap.add_argument("--k", type=int, default=30, help="kNN for normal estimation")
    ap.add_argument("--force", action="store_true", help="recompute even if normals exist")
    ap.add_argument("--val", action="store_true",
                    help="process partial_val/ (held-out set) instead of partial/")
    args = ap.parse_args()
    if args.data is None:
        args.data = resolve_data_dir("majca")

    partial_dir = os.path.join(args.data, "partial_val" if args.val else "partial")
    files = sorted(f for f in os.listdir(partial_dir) if f.endswith(".npz"))
    print(f"{len(files)} files in {partial_dir}  (k={args.k}, force={args.force})")

    n_done = n_skip = 0
    for f in tqdm(files):
        path = os.path.join(partial_dir, f)
        d = np.load(path)
        if "normals" in d.files and not args.force:
            d.close()
            n_skip += 1
            continue
        data = {key: d[key] for key in d.files}   # read all arrays into memory
        d.close()
        data["normals"] = estimate_normals(data["pcd_points"].astype(np.float32), k=args.k)

        # atomic rewrite — tmp in same dir then replace, so an interrupt can't corrupt
        fd, tmp = tempfile.mkstemp(dir=partial_dir, suffix=".npz")
        os.close(fd)
        np.savez(tmp, **data)
        os.replace(tmp, path)
        n_done += 1

    print(f"done: wrote {n_done}, skipped {n_skip} (already had normals)")


if __name__ == "__main__":
    main()
