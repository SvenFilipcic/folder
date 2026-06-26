"""
train_student.py — BC pretrain the transformer StudentVLA from IL demos (index.jsonl).

The IMITATION init for the RL student: teaches grasp + full grab→path→release from your demos, so
REINFORCE (head_RL.py --rl-student) starts from sane trajectories. Runs in the 'infer' conda env
(torch only — uv_pred is already baked into each saved state):

    conda run -n infer python workers/train_student.py --sources manual --epochs 300

I/O (from il_dataset, source-agnostic):
    input   featurize(state)                 → (N, 9)  [x,y,z,u,v,nx,ny,nz,z_abs]
    grasp   action_to_targets.grab_idx       → CE over the N points (which point to grab)
    drag    action_to_targets.waypoints      → masked Gaussian-NLL/MSE on the active waypoints
            action_to_targets.wp_quat        → masked geodesic loss on the per-waypoint wrist rotation
            action_to_targets.active         → BCE on the per-query "active" (length / stop) logit
    value   discounted return-to-go of ΔΦ    → MSE on value_head (CRITIC WARM START)

CRITIC PRETRAIN: the value_head is the same shared-encoder head PPO uses (workers/student_update.py),
NOT a separate net/file. We regress it here to the demo's MULTI-STEP discounted return
    G_t = Σ_{k≥t} γ^(k-t) · ΔΦ_k ,   ΔΦ_k = reward_after_k − reward_before_k   (same Φ as RL reward)
grouped per trajectory (source, episode), ordered by turn. Because G_t folds in FUTURE flatness gains,
a setup move that loses flatness now but unlocks more later gets a high target — so RL doesn't start
with a random critic (no cold-start cancellation). Samples missing reward_before/after (or episode) are
masked out of the value loss; with γ→1, G_t → Φ_final − Φ(s_t), i.e. it learns the flatness potential.

Saves checkpoints/student_vla.pth = {model_state_dict (actor + critic), optimizer_state_dict, max_wp}.
"""
import os, sys, math, random, argparse, collections
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import il_dataset
from model.student_vla import StudentVLA

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--il-dir",  default=os.path.join(_ROOT, "il_dataset"))
ap.add_argument("--out",     default=os.path.join(_ROOT, "checkpoints", "student_vla.pth"))
ap.add_argument("--sources", default="", help="comma list to keep (e.g. 'manual'); empty = all")
ap.add_argument("--max-wp",  type=int,   default=il_dataset.MAX_WP, help="drag waypoint cap (model + targets)")
ap.add_argument("--epochs",  type=int,   default=300)
ap.add_argument("--batch",   type=int,   default=16)
ap.add_argument("--lr",      type=float, default=1e-3)
ap.add_argument("--grasp-weight", type=float, default=1.0)
ap.add_argument("--drag-weight",  type=float, default=1.0)
ap.add_argument("--stop-weight",  type=float, default=0.5)
ap.add_argument("--rot-weight",   type=float, default=1.0, help="weight of the per-waypoint wrist-rotation geodesic loss")
ap.add_argument("--value-weight", type=float, default=0.5, help="weight of the critic (value) warm-start loss; 0 disables")
ap.add_argument("--gamma",        type=float, default=0.97, help="discount for the value target — match head_RL --rl-gamma")
ap.add_argument("--val-frac", type=float, default=0.0)
ap.add_argument("--seed",    type=int, default=0)
ap.add_argument("--device",  default=None)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

# ── build dataset (preload) ───────────────────────────────────────────────────────────────────
keep = {s for s in args.sources.split(",") if s.strip()}
rows = il_dataset.load_index(args.il_dir)
X, GIDX, WP, WQ, ACT = [], [], [], [], []
META = []                                                       # (source, episode, turn, reward_before, reward_after) per kept sample
skipped = 0
for row in rows:
    if keep and row.get("source") not in keep:
        continue
    action = row["action"]
    if "arm1" not in action:
        skipped += 1; continue
    state, _ = il_dataset.load_sample(args.il_dir, row)
    t = il_dataset.action_to_targets(action, state, max_wp=args.max_wp)["arm1"]
    X.append(il_dataset.featurize(state))                       # (N,9)
    GIDX.append(t["grab_idx"]); WP.append(t["waypoints"]); WQ.append(t["wp_quat"]); ACT.append(t["active"])
    META.append((row.get("source"), row.get("episode"), row.get("turn"),
                 row.get("reward_before"), row.get("reward_after")))

if not X:
    print(f"[student] no demos in {args.il_dir} (sources={keep or 'all'}, skipped={skipped})"); sys.exit(1)

N = X[0].shape[0]
assert all(x.shape[0] == N for x in X), "all states must share N points (FPS to head_RL.N_PCD)"
X    = torch.tensor(np.stack(X), dtype=torch.float32)           # (S,N,9)
GIDX = torch.tensor(GIDX, dtype=torch.long)                     # (S,)
WP   = torch.tensor(np.stack(WP), dtype=torch.float32)          # (S,max_wp,3)
WQ   = torch.tensor(np.stack(WQ), dtype=torch.float32)          # (S,max_wp,4) wrist quaternions
ACT  = torch.tensor(np.stack(ACT), dtype=torch.float32)         # (S,max_wp)
S    = len(X)

# ── critic targets: per-trajectory discounted return-to-go of ΔΦ ────────────────────────────────
# Group by (source, episode), order by turn, reverse-scan G_t = ΔΦ_t + γ·G_{t+1}. A sample is only
# supervised (mask=1) if it has an episode id AND every step in its trajectory has reward_before/after.
G_np  = np.zeros(S, np.float32)
VM_np = np.zeros(S, np.float32)
groups = collections.OrderedDict()
for i, (src, ep, turn, rb, ra) in enumerate(META):
    groups.setdefault((src, ep), []).append(i)
for (src, ep), idxs in groups.items():
    if ep is None or any(META[i][3] is None or META[i][4] is None for i in idxs):
        continue                                                # can't form a trajectory return
    idxs = sorted(idxs, key=lambda i: (META[i][2] if META[i][2] is not None else i))
    g = 0.0
    for i in reversed(idxs):
        g = (float(META[i][4]) - float(META[i][3])) + args.gamma * g
        G_np[i], VM_np[i] = g, 1.0
n_value = int(VM_np.sum())
use_value = args.value_weight > 0 and n_value > 0
G  = torch.tensor(G_np,  dtype=torch.float32)
VM = torch.tensor(VM_np, dtype=torch.float32)

perm = torch.randperm(S)
n_val = int(S * args.val_frac)
val_i, tr_i = perm[:n_val], perm[n_val:]
print(f"[student] {S} demos (skipped {skipped})  N={N}  max_wp={args.max_wp}  "
      f"train={len(tr_i)} val={len(val_i)}  device={device}")
print(f"[student] critic warm-start: {n_value}/{S} samples have a return target  "
      f"(γ={args.gamma}, weight={args.value_weight})" + ("" if use_value else "  → DISABLED"))

model = StudentVLA(in_dim=X.shape[2], max_wp=args.max_wp).to(device)
opt   = optim.Adam(model.parameters(), lr=args.lr)
X, GIDX, WP, ACT = X.to(device), GIDX.to(device), WP.to(device), ACT.to(device)
WQ, G, VM = WQ.to(device), G.to(device), VM.to(device)


def run(idx, train):
    model.train(train)
    tot_g = tot_d = tot_s = tot_r = tot_v = tot_acc = tot_sacc = 0.0; nb = 0
    order = idx[torch.randperm(len(idx))] if train else idx
    for i in range(0, len(order), args.batch):
        b = order[i:i + args.batch]
        grasp_logits, wp_mean, stop_logits, wp_rot = model(X[b], GIDX[b])
        gl = F.cross_entropy(grasp_logits, GIDX[b])
        m  = ACT[b].unsqueeze(-1)                                       # (B,max_wp,1) active mask
        dl = ((wp_mean - WP[b]) ** 2 * m).sum() / m.sum().clamp_min(1.0) / 3.0    # masked MSE per coord
        sl = F.binary_cross_entropy_with_logits(stop_logits, ACT[b])
        rl = model.rotation_loss(wp_rot, WQ[b], ACT[b])               # masked geodesic (radians²)
        loss = args.grasp_weight * gl + args.drag_weight * dl + args.stop_weight * sl + args.rot_weight * rl
        vl = torch.zeros((), device=device)
        if use_value:                                                   # critic warm-start (masked MSE to return-to-go)
            vm = VM[b]
            vl = ((model.value(X[b]) - G[b]) ** 2 * vm).sum() / vm.sum().clamp_min(1.0)
            loss = loss + args.value_weight * vl
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            tot_acc  += (grasp_logits.argmax(-1) == GIDX[b]).float().mean().item()
            tot_sacc += ((stop_logits > 0).float() == ACT[b]).float().mean().item()
        tot_g += gl.item(); tot_d += dl.item(); tot_s += sl.item(); tot_r += rl.item(); tot_v += float(vl); nb += 1
    return tot_g/nb, tot_d/nb, tot_s/nb, tot_r/nb, tot_v/nb, tot_acc/nb, tot_sacc/nb


best = math.inf
for ep in range(1, args.epochs + 1):
    tg, td, ts, tr, tv, tacc, tsacc = run(tr_i, True)
    score = tg + td + ts + tr
    line = (f"[student] ep {ep:4d}/{args.epochs}  grasp_ce={tg:.3f} acc={tacc*100:4.1f}%  "
            f"drag_mse={td:.4f}  stop_bce={ts:.3f} acc={tsacc*100:4.1f}%  rot={tr:.4f}"
            + (f"  value_mse={tv:.4f}" if use_value else ""))
    if len(val_i):
        vg, vd, vs, vr, vv, vacc, vsacc = run(val_i, False)
        score = vg + vd + vs + vr; line += f"  | val ce={vg:.3f} acc={vacc*100:4.1f}% mse={vd:.4f} rot={vr:.4f}"
        if use_value: line += f" vmse={vv:.4f}"
    if ep % 10 == 0 or ep == 1 or ep == args.epochs:
        print(line)
    if score < best:
        best = score
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                    "max_wp": args.max_wp}, args.out)

print(f"[student] done. best={best:.4f} → {args.out}  (actor + warm-started critic)")
print(f"[student] next: conda run -n fold python head_RL.py --rl-student   (PPO+GAE fine-tune)")
