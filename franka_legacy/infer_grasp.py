"""
infer_grasp.py — UV Mapper inference + priority-group grasp point selection.

Env: garmentlab  (needs spconv for UVMapper)

Given a captured partial npz and grasp_config.yaml, runs the UV Mapper then
walks the priority groups and returns the best (xyz, normal) to grasp.

Library usage (e.g. from a ROS2 node):
    from infer_grasp import select_grasp
    result = select_grasp("data/majca/partial/majca_0000.npz", "grasp_config.yaml")
    if result:
        print(result["xyz_world"], result["normal"])

Standalone:
    conda run -n garmentlab python infer_grasp.py data/majca/partial/majca_0000.npz
    conda run -n garmentlab python infer_grasp.py <npz> --config grasp_config.yaml --out grasp_result.json
"""

import os, sys, json, argparse
import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (franka_legacy/ → ..)


# ── preprocessing (mirrors UVMapperDataset.__getitem__ exactly) ───────────────────────────
def _build_input(npz_path, device):
    """Load partial npz → (1, N, 7) tensor + raw arrays for post-processing."""
    d        = np.load(npz_path)
    pts      = d["pcd_points"].astype(np.float32)   # (N, 3) centroid-normalised
    centroid = d["centroid"].astype(np.float32)      # (3,) absolute world centroid
    normals  = d["normals"].astype(np.float32)       # (N, 3) unit, oriented +z
    z_table  = (pts[:, 2] + centroid[2])[:, None]   # (N, 1) absolute z height
    pts7     = np.concatenate([pts, z_table, normals], axis=1)  # (N, 7)
    tensor   = torch.from_numpy(pts7).unsqueeze(0).to(device)   # (1, N, 7)
    return tensor, pts, centroid, normals


# ── model loading ─────────────────────────────────────────────────────────────────────────
def _load_model(model_path, device):
    sys.path.insert(0, _ROOT)
    from model.uv_mapper import UVMapper
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))
    model = UVMapper().to(device)
    model.load_state_dict(state)
    model.eval()
    return model


# ── grasp pair selection ──────────────────────────────────────────────────────────────────
def _best_in_group(g, uv_pred, conf_pred, normals, pts_norm, centroid):
    """Return point dict for the best match in group g, or None if no pts match."""
    center  = np.array(g["uv_center"], dtype=np.float32)
    radius  = float(g["uv_radius"])
    mask    = np.linalg.norm(uv_pred - center, axis=1) <= radius
    n_match = int(mask.sum())
    if n_match == 0:
        return None
    cands    = np.where(mask)[0]
    best_idx = cands[conf_pred[cands].argmax()]
    xyz_norm = pts_norm[best_idx]
    return {
        "group_name": g["name"],
        "pcd_idx":    int(best_idx),
        "xyz_world":  (xyz_norm + centroid).astype(np.float32).tolist(),
        "xyz_norm":   xyz_norm.tolist(),
        "centroid":   centroid.tolist(),
        "normal":     normals[best_idx].tolist(),
        "uv":         uv_pred[best_idx].tolist(),
        "confidence": float(conf_pred[best_idx]),
        "n_matching": n_match,
    }


def _pick_pair(uv_pred, conf_pred, normals, pts_norm, centroid, pairs):
    """
    Walk priority pairs. For each pair both 'a' and 'b' must match (single-point
    pairs only have 'a'). Returns the first fully-matched pair.
    """
    for i, pair in enumerate(pairs):
        points = []
        matched = True
        for key in ("a", "b"):
            if key not in pair:
                continue
            pt = _best_in_group(pair[key], uv_pred, conf_pred, normals, pts_norm, centroid)
            if pt is None:
                print(f"  [grasp] pair {i+1} '{pair['name']}' / '{pair[key]['name']}': 0 pts — skip pair")
                matched = False
                break
            print(f"  [grasp] pair {i+1} '{pair['name']}' / '{pair[key]['name']}': "
                  f"{pt['n_matching']} pts  conf={pt['confidence']:.3f}")
            points.append(pt)

        if matched and points:
            print(f"  [grasp] ✓ selected pair '{pair['name']}' ({len(points)} point(s))")
            return {"found": True, "pair_name": pair["name"], "points": points}

    print("  [grasp] no pair matched — returning None")
    return None


# ── public API ────────────────────────────────────────────────────────────────────────────
def select_grasp(npz_path,
                 config_path=None,
                 model_path=None,
                 device=None):
    """
    Run UV Mapper inference on a captured npz and return the best grasp point.

    Returns a dict (see module docstring) or None if no group matched.
    Coordinates:
      xyz_world = absolute world coordinates (same frame as Isaac sim, or camera frame in lab).
      In the real lab: transform xyz_world from camera frame to robot base frame using
      the camera-robot extrinsic calibration before sending to Franka.
    """
    if config_path is None:
        config_path = os.path.join(_ROOT, "grasp_config.yaml")
    if model_path is None:
        model_path = os.path.join(_ROOT, "checkpoints", "uv_mapper_best.pth")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pairs = cfg["grasp_pairs"]

    print(f"[infer_grasp] npz={npz_path}  device={device}  pairs={len(pairs)}")
    tensor, pts_norm, centroid, normals = _build_input(npz_path, device)

    model = _load_model(model_path, device)
    with torch.no_grad():
        uv_t, conf_t = model.predict_uv(tensor)   # (1,N,2), (1,N)
    uv_pred   = uv_t[0].cpu().numpy()             # (N, 2)
    conf_pred = conf_t[0].cpu().numpy()            # (N,)

    return _pick_pair(uv_pred, conf_pred, normals, pts_norm, centroid, pairs)


# ── standalone entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz",            help="path to partial npz (from data_gen.py capture)")
    ap.add_argument("--config",       default=None, help="grasp_config.yaml path")
    ap.add_argument("--model",        default=None, help="uv_mapper checkpoint path")
    ap.add_argument("--out",          default=None, help="save result to this JSON file")
    ap.add_argument("--device",       default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    result = select_grasp(args.npz, args.config, args.model, args.device)
    if result:
        print(f"\n[infer_grasp] GRASP PAIR: '{result['pair_name']}' ({len(result['points'])} point(s))")
        for pt in result["points"]:
            print(f"  {pt['group_name']}: xyz_world={np.array(pt['xyz_world']).round(4)}"
                  f"  conf={pt['confidence']:.3f}  uv={np.array(pt['uv']).round(3)}")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n[infer_grasp] saved → {args.out}")
    else:
        print("\n[infer_grasp] no grasp pair found")
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"found": False}, f)
