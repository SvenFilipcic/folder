"""
student_update.py — MULTI-STEP PPO+GAE update for the StudentVLA trajectory policy. 'infer' env.

Called by head_RL.py --rl-student every K episodes:
    conda run -n infer python student_update.py --buffer student_buffer.json --n 96

Why multi-step (vs the old single-action REINFORCE):
    A grab is credited not by its own immediate flatness but by the DISCOUNTED FUTURE of the whole
    smoothing sequence it belongs to. A tension-building move that lowers flatness now (r_t < 0) is
    rewarded if it unlocks a bigger gain later — exactly the planning behaviour single-step RL can't
    learn. head_RL writes one buffer entry per grab, tagged with its trajectory id `traj`, step `t`,
    per-step reward `reward` (= flatness improvement ΔΦ) and `done` (last grab of the sequence).

Algorithm (textbook PPO + Generalised Advantage Estimation):
    per trajectory, sorted by t:
        δ_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)         # TD error (critic baseline)
        Â_t = δ_t + γλ·(1−done_t)·Â_{t+1}                     # GAE (reverse scan)
        G_t = Â_t + V(s_t)                                    # critic target (discounted return)
    Â is normalised across the batch. Then for `--epochs` passes (minibatched):
        ρ_t   = exp( logπ_new(a_t|s_t) − logπ_old )          # old = behaviour log_prob in the buffer
        L_pi  = − mean( min( ρ·Â , clip(ρ,1±ε)·Â ) )         # clipped surrogate
        L_v   = mean( ( V_new(s_t) − G_t )² )                # critic regression
        L     = L_pi + vf·L_v − ent·entropy                  # entropy bonus = exploration

V(s_{t+1}) is read from the NEXT grab's stored state in the same trajectory (the cloth we captured
at turn t+1); the last grab is `done`, so its bootstrap is 0. Each entry stores the FROZEN uv_pred,
so this re-featurizes straight to (N,9) and never touches the UV Mapper.
"""
import os, sys, json, argparse, collections
import numpy as np
import torch
import torch.optim as optim

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import il_dataset
from model.student_vla import StudentVLA

ap = argparse.ArgumentParser()
ap.add_argument("--buffer", required=True)
ap.add_argument("--policy", default=os.path.join(_ROOT, "checkpoints", "student_vla.pth"))
ap.add_argument("--n",      type=int, default=96, help="most-recent grabs (buffer entries) to update on")
ap.add_argument("--lr",     type=float, default=3e-4)
ap.add_argument("--gamma",  type=float, default=0.97, help="discount γ")
ap.add_argument("--lam",    type=float, default=0.95, help="GAE λ")
ap.add_argument("--clip",   type=float, default=0.2,  help="PPO clip ε")
ap.add_argument("--epochs", type=int,   default=4,    help="PPO gradient epochs over the batch")
ap.add_argument("--minibatch",  type=int,   default=16,   help="states per PPO minibatch (VRAM bound)")
ap.add_argument("--ent-weight", type=float, default=1e-3, help="entropy bonus weight")
ap.add_argument("--vf-weight",  type=float, default=0.5,  help="critic (value) loss weight")
ap.add_argument("--deterministic-critic", action="store_true",
                help="use the EXACT potential-based value V(s)=phi_target-Phi(s) (from the buffer's "
                     "'phi') instead of the learned value head, and skip the value-loss term. For the "
                     "from-scratch RL run where the value head is random (no IL value-pretraining), "
                     "this gives a clean baseline with zero cold-start noise. See module docstring.")
ap.add_argument("--phi-target", type=float, default=0.5,
                help="(--deterministic-critic) potential of a perfectly flat garment = max Phi = "
                     "FLAT_W_COV (shape→0, height→0, iou→1). V(s)=phi_target-Phi(s) = flatness left to gain")
ap.add_argument("--device", default=None)
args = ap.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

pck = torch.load(args.policy, map_location=device, weights_only=False)
policy = StudentVLA(max_wp=pck.get("max_wp", il_dataset.MAX_WP)).to(device)
policy.load_state_dict(pck["model_state_dict"], strict=False)   # BC ckpt may predate the value head
opt = optim.Adam(policy.parameters(), lr=args.lr)
if pck.get("optimizer_state_dict"):
    try: opt.load_state_dict(pck["optimizer_state_dict"])
    except Exception: pass

with open(args.buffer) as fh:
    buf = json.load(fh)
entries = buf[-args.n:]
if len(entries) < args.n:
    print(f"[student_update] {len(entries)}/{args.n} entries — skipping"); sys.exit(0)

# ── re-featurize each grab from its stored state + frozen uv_pred ──────────────────────────────
def _featurize(e):
    d  = np.load(e["state_npz"])
    uv = np.load(e["uv_pred_path"]).astype(np.float32)
    state = {"pcd_xyz": d["pcd_xyz"].astype(np.float32), "uv_pred": uv,
             "normals": d["normals"].astype(np.float32), "centroid": d["centroid"].astype(np.float32)}
    return il_dataset.featurize(state)

xs   = np.stack([_featurize(e) for e in entries])                          # (T,N,9)
gidx = np.array([int(e["grab_idx"]) for e in entries], np.int64)           # (T,)
wps  = np.array([e["waypoints"]     for e in entries], np.float32)         # (T,max_wp,3)
acts = np.array([e["active"]        for e in entries], np.float32)         # (T,max_wp)
rew  = np.array([float(e["reward"]) for e in entries], np.float32)         # (T,) per-step ΔΦ
oldlp= np.array([float(e["log_prob"]) for e in entries], np.float32)       # (T,) behaviour log_prob
done = np.array([bool(e.get("done", False)) for e in entries])            # (T,)
traj = [e.get("traj", e.get("episode", 0)) for e in entries]              # trajectory id per grab
phi  = np.array([float(e.get("phi", 0.0)) for e in entries], np.float32)   # (T,) absolute potential Φ(s)
# rotation action: present only when rollouts ran WITH rotation (not --no-rot). All-or-nothing across
# the window — if any entry lacks it, train position only (rotation excluded from the PPO objective).
has_rot = all(e.get("wp_rot3") is not None for e in entries)
wr_t = (torch.tensor(np.array([e["wp_rot3"] for e in entries], np.float32), device=device)
        if has_rot else None)                                              # (T,max_wp,3) or None

x      = torch.tensor(xs,   dtype=torch.float32, device=device)
g_t    = torch.tensor(gidx, dtype=torch.long,    device=device)
wp_t   = torch.tensor(wps,  dtype=torch.float32, device=device)
ac_t   = torch.tensor(acts, dtype=torch.float32, device=device)
oldlp_t= torch.tensor(oldlp,dtype=torch.float32, device=device)

# ── critic values V(s) for every state ─────────────────────────────────────────────────────────
# Deterministic critic: the reward r_t = Φ(s')−Φ(s) is a potential difference, so the return-to-go
# telescopes to Φ_terminal − Φ(s).  V(s) = phi_target − Φ(s) is therefore the EXACT value — no MLP,
# no cold-start.  Learned critic: the value head (needs IL value-pretraining to be sane at start).
if args.deterministic_critic:
    values_np = (args.phi_target - phi).astype(np.float32)                 # (T,)  exact, no grad needed
else:
    policy.eval()
    with torch.no_grad():
        values = torch.cat([policy.value(x[i:i + args.minibatch]) for i in range(0, len(x), args.minibatch)])
    values_np = values.cpu().numpy()                                      # (T,)

# ── GAE advantage + discounted return, computed WITHIN each trajectory (reverse scan) ───────────
# group entry indices by trajectory id, preserving order (entries are appended in time order)
by_traj = collections.OrderedDict()
for i, tid in enumerate(traj):
    by_traj.setdefault(tid, []).append(i)

adv_np = np.zeros(len(entries), np.float32)
ret_np = np.zeros(len(entries), np.float32)
for idxs in by_traj.values():
    gae = 0.0
    for j in reversed(range(len(idxs))):
        i      = idxs[j]
        v      = values_np[i]
        v_next = 0.0 if done[i] or j == len(idxs) - 1 else values_np[idxs[j + 1]]
        nonterm= 0.0 if done[i] else 1.0
        delta  = rew[i] + args.gamma * v_next * nonterm - v
        gae    = delta + args.gamma * args.lam * nonterm * gae
        adv_np[i] = gae
        ret_np[i] = gae + v                                               # critic target

adv = torch.tensor(adv_np, dtype=torch.float32, device=device)
ret = torch.tensor(ret_np, dtype=torch.float32, device=device)
adv = (adv - adv.mean()) / (adv.std() + 1e-6)                             # normalise for stable gradients

# ── PPO epochs (clipped actor + critic regression + entropy bonus), minibatched ─────────────────
policy.train()
T = len(entries)
last = {}
for _ in range(args.epochs):
    for s in range(0, T, args.minibatch):
        mb = slice(s, s + args.minibatch)
        log_prob, entropy, value = policy.evaluate(
            x[mb], g_t[mb], wp_t[mb], ac_t[mb], None if wr_t is None else wr_t[mb])
        ratio   = torch.exp(log_prob - oldlp_t[mb])
        unclip  = ratio * adv[mb]
        clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[mb]
        L_pi    = -torch.min(unclip, clipped).mean()
        # deterministic critic ⇒ nothing to regress; only the actor + entropy bonus train
        L_v     = torch.zeros((), device=device) if args.deterministic_critic else (value - ret[mb]).pow(2).mean()
        ent     = entropy.mean()
        L       = L_pi + args.vf_weight * L_v - args.ent_weight * ent

        opt.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        last = {"L": L.item(), "L_pi": L_pi.item(), "L_v": L_v.item(), "ent": ent.item()}

torch.save({"model_state_dict": policy.state_dict(), "optimizer_state_dict": opt.state_dict(),
            "max_wp": pck.get("max_wp", il_dataset.MAX_WP)}, args.policy)
print(f"[student_update] L={last['L']:.4f} (π {last['L_pi']:.4f}  V {last['L_v']:.4f}  H {last['ent']:.3f})  "
      f"reward Σ/step μ={rew.mean():.4f}  ret μ={ret_np.mean():.4f}  "
      f"{len(by_traj)} trajs × {T} grabs → {args.policy}")

# update windows don't overlap (n = k*K*turns = grabs since last update) → safe to clear /tmp
for e in entries:
    for k in ("state_npz", "uv_pred_path"):
        try: os.unlink(e[k])
        except Exception: pass
