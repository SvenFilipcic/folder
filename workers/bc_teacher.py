"""
bc_teacher.py — behaviour-clone the RL policy head from IL demos, then hand off to REINFORCE.

Pipeline:
    manual mouse drags / Haiku demos  →  il_dataset/index.jsonl  →  (this script, BC)
    →  checkpoints/rl_policy.pth  →  head_RL.py --rl  (REINFORCE fine-tune)

Runs in the 'infer' conda env (needs torch + model.uv_mapper + rl_policy):
    conda run -n infer python workers/bc_teacher.py --epochs 300
    conda run -n infer python workers/bc_teacher.py --sources manual --epochs 500

What it trains:
    The UV Mapper is FROZEN (same as rl_infer/rl_update). We clone RLPolicy.head-1:
      grasp_head_1  ← cross-entropy to the demo's grab point index
      action_head_1 ← MSE on the Gaussian mean (in the policy's RAW pre-bound space) to the
                      demo's (dx, dy, z) target.

Demo → policy target:
    grasp idx = arm.grab_pcd_idx                       (index into the N-point cloud)
    dx, dy    = arm.release[:2] − pcd_xyz[idx][:2]     (drag displacement from the grab point;
                                                        centroid cancels — both are centroid-rel)
    z         = arm.release[2]                          (absolute height above the table)
    then invert the policy bounds (tanh·MAX_DRAG for dx,dy; sigmoid·[Z_MIN,Z_MAX] for z) to get
    the raw target the Gaussian mean should regress to. Targets outside the policy's reachable
    range are clipped (e.g. a low z<Z_MIN slide becomes z=Z_MIN).

The saved checkpoint matches rl_infer.py / rl_update.py exactly
    {"policy_state_dict", "optimizer_state_dict", "episode"}
so `head_RL.py --rl` resumes straight from it and continues with REINFORCE.
"""
import os, sys, json, math, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # workers/ → repo root
sys.path.insert(0, _ROOT)

import il_dataset
from model.uv_mapper import UVMapper
from rl_policy import RLPolicy, MAX_DRAG, Z_MIN, Z_MAX

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--il-dir",  default=os.path.join(_ROOT, "il_dataset"), help="IL dataset dir (has index.jsonl)")
ap.add_argument("--model",   default=os.path.join(_ROOT, "checkpoints", "uv_mapper_best.pth"), help="frozen UV Mapper ckpt")
ap.add_argument("--out",     default=os.path.join(_ROOT, "checkpoints", "rl_policy.pth"), help="policy ckpt to write (RL resumes this)")
ap.add_argument("--sources", default="", help="comma list to keep (e.g. 'manual' or 'manual,haiku'); empty = all")
ap.add_argument("--min-improved", action="store_true", help="keep only demos with improved==True (skips unscored manual demos)")
ap.add_argument("--epochs",  type=int,   default=300)
ap.add_argument("--batch",   type=int,   default=8)
ap.add_argument("--lr",      type=float, default=3e-4)
ap.add_argument("--grasp-weight",  type=float, default=1.0)
ap.add_argument("--action-weight", type=float, default=1.0)
ap.add_argument("--val-frac", type=float, default=0.0, help="fraction held out for validation")
ap.add_argument("--no-cache", action="store_true", help="don't cache frozen UV features in RAM (slower, lower mem)")
ap.add_argument("--seed",    type=int, default=0)
ap.add_argument("--device",  default=None)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")


# ── bound inversion: bounded action → raw pre-bound value the Gaussian mean regresses to ──────
def _to_raw(dx, dy, z):
    cdx = np.clip(dx, -0.999 * MAX_DRAG, 0.999 * MAX_DRAG) / MAX_DRAG
    cdy = np.clip(dy, -0.999 * MAX_DRAG, 0.999 * MAX_DRAG) / MAX_DRAG
    zn  = float(np.clip((z - Z_MIN) / (Z_MAX - Z_MIN), 1e-3, 1.0 - 1e-3))
    return (math.atanh(cdx), math.atanh(cdy), math.log(zn / (1.0 - zn)))


# ── build the demo set ────────────────────────────────────────────────────────────────────────
keep_sources = {s for s in args.sources.split(",") if s.strip()}
rows = il_dataset.load_index(args.il_dir)
samples = []   # (pts7 (N,7) f32, grasp_idx int, raw_target (3,) f32, src)
skipped = 0
for row in rows:
    if keep_sources and row.get("source") not in keep_sources:
        continue
    if args.min_improved and not row.get("improved"):
        continue
    arm = row["action"].get("arm1")
    if arm is None:
        skipped += 1; continue
    idx = int(arm.get("grab_pcd_idx", -1))
    if idx < 0:                          # VR / RealSense demos have no point index → can't supervise grasp head
        skipped += 1; continue
    state, _ = il_dataset.load_sample(args.il_dir, row)
    pts      = state["pcd_xyz"].astype(np.float32)        # (N,3) centroid-normalised
    normals  = state["normals"].astype(np.float32)        # (N,3)
    centroid = state["centroid"].astype(np.float32)       # (3,)
    if not (0 <= idx < len(pts)):
        skipped += 1; continue
    rel = np.asarray(arm["release"], np.float32)          # [x-cx, y-cy, z_abs]
    dx  = float(rel[0] - pts[idx, 0])                     # displacement from the grasp point
    dy  = float(rel[1] - pts[idx, 1])
    z   = float(rel[2])
    z_abs = (pts[:, 2] + centroid[2])[:, None]            # (N,1) absolute height — same feature as rl_infer
    pts7  = np.concatenate([pts, z_abs, normals], axis=1).astype(np.float32)   # (N,7)
    samples.append((pts7, idx, np.array(_to_raw(dx, dy, z), np.float32), row.get("source", "?")))

if not samples:
    print(f"[bc] no usable demos in {args.il_dir} "
          f"(sources={keep_sources or 'all'}, skipped={skipped}). Nothing to train.")
    sys.exit(1)

N = samples[0][0].shape[0]
from collections import Counter
print(f"[bc] {len(samples)} demos  (skipped {skipped})  N={N}  by source: "
      f"{dict(Counter(s[3] for s in samples))}")

random.shuffle(samples)
n_val   = int(len(samples) * args.val_frac)
val     = samples[:n_val]
train   = samples[n_val:]
print(f"[bc] train={len(train)}  val={len(val)}  device={device}")


# ── frozen UV Mapper ──────────────────────────────────────────────────────────────────────────
ckpt  = torch.load(args.model, map_location=device, weights_only=False)
mstate = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))
uv_mapper = UVMapper().to(device)
uv_mapper.load_state_dict(mstate)
uv_mapper.eval()
for p in uv_mapper.parameters():
    p.requires_grad_(False)

policy = RLPolicy(n_points=N).to(device)
if os.path.exists(args.out):
    try:
        prev = torch.load(args.out, map_location=device, weights_only=False)
        policy.load_state_dict(prev["policy_state_dict"])
        print(f"[bc] warm-started policy from {args.out}")
    except Exception as e:
        print(f"[bc] could not warm-start ({e}); training from scratch")
optimizer = optim.Adam(policy.parameters(), lr=args.lr)

# cache frozen features (UV Mapper is frozen → encode each demo once, reuse every epoch).
# Stored on the CPU (~6.3 MB/demo) and moved to device per use, so 1000s of demos use host
# RAM, not VRAM (caching on GPU would OOM at ~1k demos). --no-cache re-encodes each epoch.
_feat_cache = {}
def _encode(pts7):
    key = id(pts7)
    cached = _feat_cache.get(key)
    if cached is not None:
        return cached.to(device, non_blocking=True)
    with torch.no_grad():
        t = torch.from_numpy(pts7).unsqueeze(0).to(device)    # (1,N,7)
        f = uv_mapper.encode(t).detach()                      # (1,N,384)
    if not args.no_cache:
        _feat_cache[key] = f.cpu()
    return f


def _step_batch(batch, train_mode):
    """Encode + head-1 BC loss over a minibatch (one encode per sample; spconv isn't batched)."""
    grasp_losses, action_losses, grasp_dist = [], [], []
    for pts7, idx, raw_t, _src in batch:
        f = _encode(pts7)                                     # (1,N,384)
        g = policy._global(f)                                 # (1,256)
        logits = policy.grasp_head_1(g)                       # (1,N)
        tgt    = torch.tensor([idx], device=device)
        grasp_losses.append(F.cross_entropy(logits, tgt))
        mean   = policy.action_head_1(g)[:, :3]               # (1,3) Gaussian mean (raw space)
        raw    = torch.tensor(raw_t, device=device).unsqueeze(0)
        action_losses.append(F.mse_loss(mean, raw))
        with torch.no_grad():                                 # grasp quality: metres between argmax & target point
            pred = int(logits.argmax(1).item())
            grasp_dist.append(float(np.linalg.norm(pts7[pred, :3] - pts7[idx, :3])))
    gl = torch.stack(grasp_losses).mean()
    al = torch.stack(action_losses).mean()
    loss = args.grasp_weight * gl + args.action_weight * al
    if train_mode:
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
    return float(gl), float(al), float(np.mean(grasp_dist))


def _epoch(data, train_mode):
    if train_mode:
        random.shuffle(data)
        policy.train()
    else:
        policy.eval()
    gls, als, gds, nb = 0.0, 0.0, 0.0, 0
    for i in range(0, len(data), args.batch):
        gl, al, gd = _step_batch(data[i:i + args.batch], train_mode)
        gls += gl; als += al; gds += gd; nb += 1
    return gls / nb, als / nb, gds / nb


# ── train ───────────────────────────────────────────────────────────────────────────────────
best = math.inf
for ep in range(1, args.epochs + 1):
    tg, ta, td = _epoch(train, True)
    line = f"[bc] ep {ep:4d}/{args.epochs}  grasp_ce={tg:.4f}  act_mse={ta:.4f}  grasp_dist={td*100:.1f}cm"
    score = tg + ta
    if val:
        vg, va, vd = _epoch(val, False)
        line += f"  | val grasp_ce={vg:.4f} act_mse={va:.4f} dist={vd*100:.1f}cm"
        score = vg + va
    if ep % 10 == 0 or ep == 1 or ep == args.epochs:
        print(line)
    if score < best:
        best = score
        torch.save({"policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "episode": 0}, args.out)

print(f"[bc] done. best score={best:.4f}  →  {args.out}")
print(f"[bc] next: conda run -n fold python head_RL.py --rl   "
      f"(REINFORCE fine-tunes this checkpoint)")
