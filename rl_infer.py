"""
rl_infer.py — UV Mapper encode + RL policy sample.  Runs in 'infer' conda env.

Called by demo_gui.py (fold env) as a subprocess each turn:
    conda run -n infer python rl_infer.py --npz /tmp/state.npz --out /tmp/action.json

Outputs JSON:
    {"pcd_idx": 1234,
     "dx": 0.18, "dy": -0.22, "z": 0.12,        # bounded, for sim execution
     "raw_dx": 0.37, "raw_dy": -0.49, "raw_z": 0.51,  # pre-bound, for log_prob recompute
     "log_prob": -4.1,
     "uv_pred_u": 0.42, "uv_pred_v": 0.71}       # UV at chosen point, for aux loss
"""
import os, sys, json, argparse
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--npz",     required=True, help="state NPZ from demo_gui capture")
ap.add_argument("--out",     required=True, help="output JSON path")
ap.add_argument("--model",   default=None,  help="UV Mapper checkpoint")
ap.add_argument("--policy",  default=None,  help="RL policy checkpoint (None = random init)")
ap.add_argument("--greedy",  action="store_true", help="deterministic action (eval mode)")
ap.add_argument("--device",  default=None)
ap.add_argument("--uv-render", action="store_true", help="also save UV render image + uv_pred npy")
args = ap.parse_args()

# camera constants (must match head_RL.py)
_CAM_Z  = 1.0
_CAM_W, _CAM_H = 640, 480
_CAM_FX = 24.0 / 36.0 * _CAM_W
_CAM_FY = 24.0 / 24.0 * _CAM_H
_CAM_CX, _CAM_CY = _CAM_W / 2.0, _CAM_H / 2.0

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

model_path  = args.model  or os.path.join(_ROOT, "checkpoints", "uv_mapper_best.pth")
policy_path = args.policy or os.path.join(_ROOT, "checkpoints", "rl_policy.pth")

# ── load UV Mapper (frozen) ───────────────────────────────────────────────────
from model.uv_mapper import UVMapper
ckpt  = torch.load(model_path, map_location=device, weights_only=False)
state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))
uv_mapper = UVMapper().to(device)
uv_mapper.load_state_dict(state)
uv_mapper.eval()
for p in uv_mapper.parameters():
    p.requires_grad_(False)

# ── load RL policy ────────────────────────────────────────────────────────────
from rl_policy import RLPolicy
policy = RLPolicy().to(device)
if os.path.exists(policy_path):
    ckpt_p = torch.load(policy_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt_p["policy_state_dict"])
    print(f"[rl_infer] loaded policy from {policy_path}")
else:
    print(f"[rl_infer] no policy checkpoint — saving random init so all runs share same weights")
    os.makedirs(os.path.dirname(policy_path), exist_ok=True)
    torch.save({"policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": {},
                "episode": 0}, policy_path)
policy.eval()

# ── load state NPZ ────────────────────────────────────────────────────────────
d        = np.load(args.npz)
pts      = d["pcd_xyz"].astype(np.float32)     # (N, 3) centroid-normalised
normals  = d["normals"].astype(np.float32)     # (N, 3)
centroid = d["centroid"].astype(np.float32)    # (3,)

z_table = (pts[:, 2] + centroid[2])[:, None]   # (N, 1) absolute z height
pts7    = np.concatenate([pts, z_table, normals], axis=1)  # (N, 7)
tensor  = torch.from_numpy(pts7).unsqueeze(0).to(device)   # (1, N, 7)

# ── UV Mapper encode → per-point features ────────────────────────────────────
with torch.no_grad():
    f = uv_mapper.encode(tensor)               # (1, N, 384)

    # also get UV predictions for aux loss logging
    phi_u, phi_v = uv_mapper.head_u(f), uv_mapper.head_v(f)
    ku = phi_u[0].argmax(dim=-1).float()       # (N,)
    kv = phi_v[0].argmax(dim=-1).float()       # (N,)
    uv_pred = torch.stack([ku / (uv_mapper.k - 1),
                           kv / (uv_mapper.k - 1)], dim=1).cpu().numpy()  # (N, 2)

# ── policy sample ─────────────────────────────────────────────────────────────
with torch.no_grad():
    result = policy.sample_head1(f.detach(), greedy=args.greedy)

pcd_idx  = int(result["pcd_idx"][0].item())
dx       = float(result["dx"][0].item())
dy       = float(result["dy"][0].item())
z        = float(result["z"][0].item())
raw_dx   = float(result["raw_dx"][0].item())
raw_dy   = float(result["raw_dy"][0].item())
raw_z    = float(result["raw_z"][0].item())
log_prob = float(result["log_prob"][0].item())

uv_pred_path  = args.npz.replace(".npz", "_uvpred.npy")
uv_render_path = args.npz.replace(".npz", "_uv_render.png")

np.save(uv_pred_path, uv_pred)   # always save full (N,2) UV predictions

if args.uv_render:
    from PIL import Image as _PIL
    _SZ = 128
    world_z    = pts[:, 2] + centroid[2]                    # absolute height above table
    height_norm = np.clip(world_z / 0.40, 0.0, 1.0)        # 0=flat, 1=max raised
    tilt        = np.clip(1.0 - normals[:, 2], 0.0, 1.0)   # 0=face-up flat, 1=tilted/fold

    ux = np.clip((uv_pred[:, 0] * (_SZ - 1)).astype(np.int32), 0, _SZ - 1)
    vy = np.clip((uv_pred[:, 1] * (_SZ - 1)).astype(np.int32), 0, _SZ - 1)

    R = (height_norm * 255).astype(np.uint8)
    G = (tilt        * 255).astype(np.uint8)

    img = np.zeros((_SZ, _SZ, 3), dtype=np.uint8)
    for du in range(-1, 2):
        for dv in range(-1, 2):
            ux_ = np.clip(ux + du, 0, _SZ - 1)
            vy_ = np.clip(vy + dv, 0, _SZ - 1)
            img[vy_, ux_, 0] = np.maximum(img[vy_, ux_, 0], R)
            img[vy_, ux_, 1] = np.maximum(img[vy_, ux_, 1], G)

    _PIL.fromarray(img).save(uv_render_path)
    print(f"[rl_infer] UV render saved → {uv_render_path}  ({_SZ}×{_SZ}, UV-space)")

out = {
    "pcd_idx":      pcd_idx,
    "dx":           dx,
    "dy":           dy,
    "z":            z,
    "raw_dx":       raw_dx,
    "raw_dy":       raw_dy,
    "raw_z":        raw_z,
    "log_prob":     log_prob,
    "uv_pred_u":    float(uv_pred[pcd_idx, 0]),
    "uv_pred_v":    float(uv_pred[pcd_idx, 1]),
    "uv_pred_path": uv_pred_path,
    "uv_render":    uv_render_path if args.uv_render else None,
}

with open(args.out, "w") as fh:
    json.dump(out, fh, indent=2)

print(f"[rl_infer] pcd_idx={pcd_idx}  dx={dx:.3f}  dy={dy:.3f}  z={z:.3f}"
      f"  log_prob={log_prob:.3f}")
