"""
_uv_predict_worker.py — inference-only helper (runs in infer conda env, needs spconv).
Called as subprocess by visualize_uv.py. Not meant to be run directly.

Writes a .npz with keys: uv_pred (N,2), conf_pred (N,).
"""
import sys, os, argparse
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--npz",        required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out",        required=True)
ap.add_argument("--device",     default=None)
args = ap.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

from model.uv_mapper import UVMapper

ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))
model = UVMapper().to(device)
model.load_state_dict(state)
model.eval()

d        = np.load(args.npz)
pts      = d["pcd_points"].astype(np.float32)   # (N, 3) centroid-normalised
centroid = d["centroid"].astype(np.float32)      # (3,)
normals  = d["normals"].astype(np.float32)       # (N, 3)
z_table  = (pts[:, 2] + centroid[2])[:, None]   # (N, 1) absolute z height
pts7     = np.concatenate([pts, z_table, normals], axis=1)  # (N, 7)
tensor   = torch.from_numpy(pts7).unsqueeze(0).to(device)   # (1, N, 7)

with torch.no_grad():
    uv_t, conf_t = model.predict_uv(tensor)

np.savez(args.out,
         uv_pred   = uv_t[0].cpu().numpy(),
         conf_pred = conf_t[0].cpu().numpy())
print(f"[worker] saved → {args.out}")
