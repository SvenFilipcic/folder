"""
student_infer.py — StudentVLA trajectory sample for the RL loop.  Runs in 'infer' conda env.

Called by head_RL.py --rl-student each turn:
    conda run -n infer python student_infer.py --npz state.npz --out action.json [--greedy]

Computes uv_pred with the frozen UV Mapper (the capture npz has no UV), featurizes to (N,9),
samples (grasp_bin, drag) from the policy, and writes the action + the log_prob + the uv_pred
path (so student_update can re-featurize without re-running the UV Mapper).
"""
import os, sys, json, argparse
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import il_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--npz",    required=True)
ap.add_argument("--out",    required=True)
ap.add_argument("--model",  default=os.path.join(_ROOT, "checkpoints", "uv_mapper_best.pth"))
ap.add_argument("--policy", default=os.path.join(_ROOT, "checkpoints", "student_vla.pth"))
ap.add_argument("--greedy", action="store_true")
ap.add_argument("--device", default=None)
args = ap.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

# ── frozen UV Mapper (for per-point uv_pred) ──────────────────────────────────────────────────
from model.uv_mapper import UVMapper
ck   = torch.load(args.model, map_location=device, weights_only=False)
uv_mapper = UVMapper().to(device)
uv_mapper.load_state_dict(ck.get("model_state_dict", ck.get("ema_state_dict", ck)))
uv_mapper.eval()
for p in uv_mapper.parameters():
    p.requires_grad_(False)

# ── student policy ────────────────────────────────────────────────────────────────────────────
from model.student_vla import StudentVLA
pck = torch.load(args.policy, map_location=device, weights_only=False)
policy = StudentVLA(max_wp=pck.get("max_wp", il_dataset.MAX_WP)).to(device)
policy.load_state_dict(pck["model_state_dict"], strict=False)   # value head unused at inference
policy.eval()

# ── state → uv_pred → featurize ──────────────────────────────────────────────────────────────
d        = np.load(args.npz)
pts      = d["pcd_xyz"].astype(np.float32)
normals  = d["normals"].astype(np.float32)
centroid = d["centroid"].astype(np.float32)

z_abs  = (pts[:, 2] + centroid[2])[:, None]
pts7   = np.concatenate([pts, z_abs, normals], axis=1)
tensor = torch.from_numpy(pts7).unsqueeze(0).to(device)
with torch.no_grad():
    f = uv_mapper.encode(tensor)                                  # (1,N,384)
    ku = uv_mapper.head_u(f)[0].argmax(-1).float() / (uv_mapper.k - 1)
    kv = uv_mapper.head_v(f)[0].argmax(-1).float() / (uv_mapper.k - 1)
    uv_pred = torch.stack([ku, kv], 1).cpu().numpy().astype(np.float32)   # (N,2)

state = {"pcd_xyz": pts, "uv_pred": uv_pred, "normals": normals, "centroid": centroid}
x = torch.from_numpy(il_dataset.featurize(state)).unsqueeze(0).to(device)  # (1,N,9)

with torch.no_grad():
    r = policy.sample(x, greedy=args.greedy)
grab_idx  = int(r["grab_idx"][0].item())
waypoints = r["waypoints"][0].cpu().numpy().astype(np.float32)    # (max_wp,3)
active    = r["active"][0].cpu().numpy().astype(np.float32)       # (max_wp,)
wp_quat   = r["wp_quat"][0].cpu().numpy().astype(np.float32)      # (max_wp,4) deterministic wrist rot
log_prob  = float(r["log_prob"][0].item())

release, path  = StudentVLA.traj_split(waypoints, active)         # contiguous active prefix
k              = len(path) + 1                                    # active waypoints = path + release
path_quat      = wp_quat[:k - 1].tolist()                         # orientation per path waypoint
release_quat   = wp_quat[k - 1].tolist()                          # orientation at release
grab_u, grab_v = float(uv_pred[grab_idx, 0]), float(uv_pred[grab_idx, 1])

uv_pred_path = args.npz.replace(".npz", "_uvpred.npy")
np.save(uv_pred_path, uv_pred)

out = {
    "grab_idx": grab_idx,
    "grab_u":   grab_u, "grab_v": grab_v,
    "release":  release,
    "path":     path,
    "path_quat":    path_quat,          # per-waypoint wrist orientation (deterministic, BC-learned)
    "release_quat": release_quat,
    "waypoints": waypoints.tolist(),    # raw sampled action — stored verbatim for log_prob recompute
    "active":    active.tolist(),
    "log_prob": log_prob,
    "uv_pred_path": uv_pred_path,
}
with open(args.out, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"[student_infer] grab_idx={grab_idx} uv=({grab_u:.2f},{grab_v:.2f}) "
      f"k={len(path)+1} release={[round(c,3) for c in release]} log_prob={log_prob:.2f}")
