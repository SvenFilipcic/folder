"""
uv_infer.py — UV Mapper encode only (no RL policy). Runs in 'infer' conda env.

Called by head_RL.py VLA mode each turn:
    conda run -n infer python uv_infer.py --npz /tmp/state.npz --out /tmp/uv_pred.npy

Loads the frozen UV Mapper, predicts per-point (u,v) for every captured point,
and saves an (N,2) float32 array. head_RL.py feeds these UV coords + xyz to Haiku.
"""
import os, sys, argparse
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--npz",   required=True, help="state NPZ with pcd_xyz, normals, centroid")
ap.add_argument("--out",   required=True, help="output .npy path for (N,2) UV predictions")
ap.add_argument("--model", default=None,  help="UV Mapper checkpoint")
ap.add_argument("--device", default=None)
args = ap.parse_args()

device     = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
model_path = args.model  or os.path.join(_ROOT, "checkpoints", "uv_mapper_best.pth")

# ── load UV Mapper (frozen) ───────────────────────────────────────────────────
from model.uv_mapper import UVMapper
ckpt  = torch.load(model_path, map_location=device, weights_only=False)
state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))
uv_mapper = UVMapper().to(device)
uv_mapper.load_state_dict(state)
uv_mapper.eval()
for p in uv_mapper.parameters():
    p.requires_grad_(False)

# ── load state NPZ ────────────────────────────────────────────────────────────
d        = np.load(args.npz)
pts      = d["pcd_xyz"].astype(np.float32)     # (N, 3) centroid-normalised
normals  = d["normals"].astype(np.float32)     # (N, 3)
centroid = d["centroid"].astype(np.float32)    # (3,)

z_table = (pts[:, 2] + centroid[2])[:, None]   # (N, 1) absolute z height
pts7    = np.concatenate([pts, z_table, normals], axis=1)  # (N, 7)
tensor  = torch.from_numpy(pts7).unsqueeze(0).to(device)   # (1, N, 7)

# ── UV Mapper encode → per-point UV ──────────────────────────────────────────
with torch.no_grad():
    f = uv_mapper.encode(tensor)                       # (1, N, 384)
    phi_u, phi_v = uv_mapper.head_u(f), uv_mapper.head_v(f)
    ku = phi_u[0].argmax(dim=-1).float()               # (N,)
    kv = phi_v[0].argmax(dim=-1).float()               # (N,)
    uv_pred = torch.stack([ku / (uv_mapper.k - 1),
                           kv / (uv_mapper.k - 1)], dim=1).cpu().numpy()  # (N, 2)

np.save(args.out, uv_pred.astype(np.float32))
print(f"[uv_infer] {uv_pred.shape[0]} points → UV saved to {args.out}  "
      f"u∈[{uv_pred[:,0].min():.2f},{uv_pred[:,0].max():.2f}] "
      f"v∈[{uv_pred[:,1].min():.2f},{uv_pred[:,1].max():.2f}]")
