"""
rl_update.py — REINFORCE update for the RL policy head.  Runs in 'infer' conda env.

Called by demo_gui.py every K episodes:
    conda run -n infer python rl_update.py --buffer rl_buffer.json --k 6

Loss:
    L_rl  = -mean( log_prob(a|s) * (reward - baseline) )   REINFORCE w/ mean baseline
    L_uv  = mean( ||uv_pred[action_idx] - gt_uv[mesh_vert]||^2 )  aux UV quality loss
    L     = L_rl + UV_WEIGHT * L_uv
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (workers/ → ..)
sys.path.insert(0, _ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--buffer",     required=True, help="rl_buffer.json path")
ap.add_argument("--k",          type=int, default=6, help="episodes per update (used to slice buffer)")
ap.add_argument("--model",      default=None, help="UV Mapper checkpoint")
ap.add_argument("--policy",     default=None, help="RL policy checkpoint to update")
ap.add_argument("--mesh-graph", default=None, help="majca_mesh_graph.npz for GT UV")
ap.add_argument("--lr",         type=float, default=3e-4)
ap.add_argument("--uv-weight",  type=float, default=0.1)
ap.add_argument("--device",     default=None)
args = ap.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

model_path  = args.model      or os.path.join(_ROOT, "checkpoints", "uv_mapper_best.pth")
policy_path = args.policy     or os.path.join(_ROOT, "checkpoints", "rl_policy.pth")
graph_path  = args.mesh_graph or os.path.join(_ROOT, "reference", "majca_mesh_graph.npz")

# ── load UV Mapper (frozen) ───────────────────────────────────────────────────
from model.uv_mapper import UVMapper
ckpt  = torch.load(model_path, map_location=device, weights_only=False)
state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))
uv_mapper = UVMapper().to(device)
uv_mapper.load_state_dict(state)
uv_mapper.eval()
for p in uv_mapper.parameters():
    p.requires_grad_(False)

# ── load / init RL policy ─────────────────────────────────────────────────────
from rl_policy import RLPolicy
policy = RLPolicy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=args.lr)
start_ep = 0

if os.path.exists(policy_path):
    ckpt_p = torch.load(policy_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt_p["policy_state_dict"])
    if ckpt_p.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt_p["optimizer_state_dict"])
    start_ep = ckpt_p.get("episode", 0)
    print(f"[rl_update] resumed from {policy_path}  (episode {start_ep})")
else:
    print(f"[rl_update] no checkpoint — initialising fresh policy")

# ── GT UV lookup table ────────────────────────────────────────────────────────
g      = np.load(graph_path)
gt_uv  = torch.from_numpy(g["node_uv"].astype(np.float32)).to(device)  # (22139, 2)

# ── load buffer — last k*4 entries ───────────────────────────────────────────
with open(args.buffer) as fh:
    buf = json.load(fh)

n_turns = args.k * 4
entries = buf[-n_turns:]
if len(entries) < n_turns:
    print(f"[rl_update] buffer has {len(entries)} entries, need {n_turns} — skipping update")
    sys.exit(0)

print(f"[rl_update] updating on {len(entries)} turns  (k={args.k}, {args.k}×4 turns)")

# ── build batch ──────────────────────────────────────────────────────────────
rewards    = torch.tensor([e["reward"]   for e in entries], dtype=torch.float32, device=device)
pcd_idxs   = torch.tensor([e["action_idx"] for e in entries], dtype=torch.long,    device=device)
raw_dx     = torch.tensor([e["raw_dx"]   for e in entries], dtype=torch.float32, device=device)
raw_dy     = torch.tensor([e["raw_dy"]   for e in entries], dtype=torch.float32, device=device)
raw_z      = torch.tensor([e["raw_z"]    for e in entries], dtype=torch.float32, device=device)

# pcd_to_mesh: list of (N,) arrays — one per entry
mesh_verts = [np.load(e["state_npz"])["pcd_to_mesh"].astype(np.int64) for e in entries]
# GT UV at each chosen point
gt_uv_batch = torch.stack([
    gt_uv[mesh_verts[i][entries[i]["action_idx"]]]
    for i in range(len(entries))
])  # (T, 2)

# ── encode features per entry (batched where possible, UV Mapper needs spconv) ──
# spconv doesn't support true batching across variable point clouds easily,
# so we process one at a time and stack log_probs
policy.train()
optimizer.zero_grad()

log_probs = []
uv_losses = []

for i, e in enumerate(entries):
    d       = np.load(e["state_npz"])
    pts     = d["pcd_xyz"].astype(np.float32)
    normals = d["normals"].astype(np.float32)
    centroid= d["centroid"].astype(np.float32)
    z_table = (pts[:, 2] + centroid[2])[:, None]
    pts7    = np.concatenate([pts, z_table, normals], axis=1)
    tensor  = torch.from_numpy(pts7).unsqueeze(0).to(device)

    with torch.no_grad():
        f = uv_mapper.encode(tensor)   # (1, N, 384)
        # UV prediction at chosen point for aux loss
        phi_u = uv_mapper.head_u(f)   # (1, N, K)
        phi_v = uv_mapper.head_v(f)
        ku    = phi_u[0, pcd_idxs[i]].argmax().float() / (uv_mapper.k - 1)
        kv    = phi_v[0, pcd_idxs[i]].argmax().float() / (uv_mapper.k - 1)
        uv_at_grasp = torch.stack([ku, kv])                    # (2,)

    f_det = f.detach()
    lp = policy.log_prob_head1(
        f_det,
        pcd_idxs[i:i+1],
        raw_dx[i:i+1], raw_dy[i:i+1], raw_z[i:i+1],
    )                                                           # (1,)
    log_probs.append(lp)
    uv_losses.append(F.mse_loss(uv_at_grasp, gt_uv_batch[i]))

log_probs_t = torch.cat(log_probs)                             # (T,)
uv_loss     = torch.stack(uv_losses).mean()

baseline = rewards.mean()
L_rl  = -(log_probs_t * (rewards - baseline)).mean()
L     = L_rl + args.uv_weight * uv_loss

L.backward()
torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
optimizer.step()

last_ep = entries[-1].get("episode", start_ep)
torch.save({
    "policy_state_dict":    policy.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "episode":              last_ep,
}, policy_path)

print(f"[rl_update] L={L.item():.4f}  L_rl={L_rl.item():.4f}"
      f"  L_uv={uv_loss.item():.4f}  baseline={baseline.item():.4f}"
      f"  saved → {policy_path}")

# delete processed state NPZs to avoid disk fill
for e in entries:
    try:    os.unlink(e["state_npz"])
    except: pass
