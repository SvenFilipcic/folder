"""
infer_server.py — persistent UV Mapper + StudentVLA inference/learning server. Env 'infer'.

Replaces the per-turn `conda run` cold-start that head_RL.py pays on every grab (import torch +
load two checkpoints, ~tens of seconds, GPU idle the whole time). Here both models are loaded ONCE
and stay resident; head_parallel.py (env 'fold') talks to it over a raw localhost socket
(see socket_ipc.py). State arrays travel in the message as numpy — no /tmp npz round-trips.

Launch (separately, before head_parallel.py):
    conda run -n infer --no-capture-output python workers/infer_server.py --port 5557

Protocol (one dict per message, length-prefixed pickle):
  {"op":"infer", "states":[{pcd_xyz,normals,centroid}, ...], "greedy":bool, "no_rot":bool}
        → {"actions":[{grab_idx,grab_u,grab_v,release,path,path_quat,release_quat,
                       waypoints,active,wp_rot3,log_prob,uv_pred}, ...]}
  {"op":"update", "entries":[{pcd_xyz,normals,centroid,uv_pred,grab_idx,waypoints,active,
                              wp_rot3,reward,log_prob,phi,done,traj}, ...],
                  "gamma","lam","clip","epochs","minibatch","ent_weight","vf_weight",
                  "det_critic","phi_target"}
        → {"L","L_pi","L_v","ent","reward_mean","ret_mean","n_traj","n"}
  {"op":"save",     "path":...}  → {"ok":True}
  {"op":"ping"}                  → {"ok":True}
  {"op":"shutdown"}             → server exits
"""
import os, sys, argparse, collections
import numpy as np
import torch
import torch.optim as optim

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import il_dataset
import socket_ipc
from model.uv_mapper import UVMapper
from model.student_vla import StudentVLA

ap = argparse.ArgumentParser()
ap.add_argument("--port",   type=int, default=5557)
ap.add_argument("--model",  default=os.path.join(_ROOT, "checkpoints", "uv_mapper_best.pth"))
ap.add_argument("--policy", default=os.path.join(_ROOT, "checkpoints", "student_vla.pth"))
ap.add_argument("--lr",     type=float, default=3e-4)
ap.add_argument("--device", default=None)
args = ap.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

# ── frozen UV Mapper (per-point uv_pred) ────────────────────────────────────────────────────────
ck = torch.load(args.model, map_location=device, weights_only=False)
uv_mapper = UVMapper().to(device)
uv_mapper.load_state_dict(ck.get("model_state_dict", ck.get("ema_state_dict", ck)))
uv_mapper.eval()
for p in uv_mapper.parameters():
    p.requires_grad_(False)

# ── student policy + persistent optimizer (Adam state continues across updates) ──────────────────
pck = torch.load(args.policy, map_location=device, weights_only=False)
MAX_WP = pck.get("max_wp", il_dataset.MAX_WP)
policy = StudentVLA(max_wp=MAX_WP).to(device)
policy.load_state_dict(pck["model_state_dict"], strict=False)
opt = optim.Adam(policy.parameters(), lr=args.lr)
if pck.get("optimizer_state_dict"):
    try: opt.load_state_dict(pck["optimizer_state_dict"])
    except Exception: pass

print(f"[infer_server] loaded UV Mapper + StudentVLA (max_wp={MAX_WP}) on {device}", flush=True)


def _featurize(pcd_xyz, normals, centroid, uv_pred):
    state = {"pcd_xyz": pcd_xyz.astype(np.float32), "uv_pred": uv_pred.astype(np.float32),
             "normals": normals.astype(np.float32), "centroid": centroid.astype(np.float32)}
    return il_dataset.featurize(state)


def _uv_pred_batch(states):
    """Run the frozen UV Mapper on a batch of states → list of (N,2) uv_pred arrays."""
    pts7 = []
    for s in states:
        pts      = s["pcd_xyz"].astype(np.float32)
        normals  = s["normals"].astype(np.float32)
        centroid = s["centroid"].astype(np.float32)
        z_abs    = (pts[:, 2] + centroid[2])[:, None]
        pts7.append(np.concatenate([pts, z_abs, normals], axis=1))
    t = torch.from_numpy(np.stack(pts7)).to(device)                  # (B,N,7)
    with torch.no_grad():
        f  = uv_mapper.encode(t)                                     # (B,N,384)
        ku = uv_mapper.head_u(f).argmax(-1).float() / (uv_mapper.k - 1)
        kv = uv_mapper.head_v(f).argmax(-1).float() / (uv_mapper.k - 1)
        uv = torch.stack([ku, kv], -1).cpu().numpy().astype(np.float32)  # (B,N,2)
    return [uv[i] for i in range(len(states))]


def op_infer(msg):
    states = msg["states"]
    greedy = bool(msg.get("greedy", False))
    no_rot = bool(msg.get("no_rot", False))
    uv_preds = _uv_pred_batch(states)

    xs = np.stack([_featurize(s["pcd_xyz"], s["normals"], s["centroid"], uv)
                   for s, uv in zip(states, uv_preds)])               # (B,N,9)
    x  = torch.from_numpy(xs).to(device)
    with torch.no_grad():
        r = policy.sample(x, greedy=greedy, rot=not no_rot)

    actions = []
    for i in range(len(states)):
        grab_idx  = int(r["grab_idx"][i].item())
        waypoints = r["waypoints"][i].cpu().numpy().astype(np.float32)
        active    = r["active"][i].cpu().numpy().astype(np.float32)
        wp_quat   = r["wp_quat"][i].cpu().numpy().astype(np.float32)
        log_prob  = float(r["log_prob"][i].item())
        wp_rot3   = None if r["wp_rot3"] is None else r["wp_rot3"][i].cpu().numpy().astype(np.float32)
        uv        = uv_preds[i]

        release, path = StudentVLA.traj_split(waypoints, active)
        k             = len(path) + 1
        actions.append({
            "grab_idx":     grab_idx,
            "grab_u":       float(uv[grab_idx, 0]),
            "grab_v":       float(uv[grab_idx, 1]),
            "release":      release,
            "path":         path,
            "path_quat":    wp_quat[:k - 1].tolist(),
            "release_quat": wp_quat[k - 1].tolist(),
            "waypoints":    waypoints.tolist(),
            "active":       active.tolist(),
            "wp_rot3":      None if wp_rot3 is None else wp_rot3.tolist(),
            "log_prob":     log_prob,
            "uv_pred":      uv,                       # (N,2) — client keeps it for the update entry
        })
    return {"actions": actions}


def op_update(msg):
    """PPO+GAE over the entries (port of student_update.py, in-memory, persistent optimizer)."""
    entries     = msg["entries"]
    gamma       = float(msg.get("gamma", 0.97))
    lam         = float(msg.get("lam", 0.95))
    clip        = float(msg.get("clip", 0.2))
    epochs      = int(msg.get("epochs", 4))
    minibatch   = int(msg.get("minibatch", 16))
    ent_weight  = float(msg.get("ent_weight", 1e-3))
    vf_weight   = float(msg.get("vf_weight", 0.5))
    det_critic  = bool(msg.get("det_critic", False))
    phi_target  = float(msg.get("phi_target", 0.5))

    T = len(entries)
    xs   = np.stack([_featurize(e["pcd_xyz"], e["normals"], e["centroid"], e["uv_pred"])
                     for e in entries])                                       # (T,N,9)
    gidx = np.array([int(e["grab_idx"])      for e in entries], np.int64)
    wps  = np.array([e["waypoints"]          for e in entries], np.float32)
    acts = np.array([e["active"]             for e in entries], np.float32)
    rew  = np.array([float(e["reward"])      for e in entries], np.float32)
    oldlp= np.array([float(e["log_prob"])    for e in entries], np.float32)
    done = np.array([bool(e.get("done", False)) for e in entries])
    traj = [e.get("traj", 0) for e in entries]
    phi  = np.array([float(e.get("phi", 0.0)) for e in entries], np.float32)
    has_rot = all(e.get("wp_rot3") is not None for e in entries)
    wr_t = (torch.tensor(np.array([e["wp_rot3"] for e in entries], np.float32), device=device)
            if has_rot else None)

    x       = torch.tensor(xs,    dtype=torch.float32, device=device)
    g_t     = torch.tensor(gidx,  dtype=torch.long,    device=device)
    wp_t    = torch.tensor(wps,   dtype=torch.float32, device=device)
    ac_t    = torch.tensor(acts,  dtype=torch.float32, device=device)
    oldlp_t = torch.tensor(oldlp, dtype=torch.float32, device=device)

    # critic values V(s)
    if det_critic:
        values_np = (phi_target - phi).astype(np.float32)
    else:
        policy.eval()
        with torch.no_grad():
            values = torch.cat([policy.value(x[i:i + minibatch]) for i in range(0, T, minibatch)])
        values_np = values.cpu().numpy()

    # GAE within each trajectory (reverse scan)
    by_traj = collections.OrderedDict()
    for i, tid in enumerate(traj):
        by_traj.setdefault(tid, []).append(i)
    adv_np = np.zeros(T, np.float32)
    ret_np = np.zeros(T, np.float32)
    for idxs in by_traj.values():
        gae = 0.0
        for j in reversed(range(len(idxs))):
            i       = idxs[j]
            v       = values_np[i]
            v_next  = 0.0 if done[i] or j == len(idxs) - 1 else values_np[idxs[j + 1]]
            nonterm = 0.0 if done[i] else 1.0
            delta   = rew[i] + gamma * v_next * nonterm - v
            gae     = delta + gamma * lam * nonterm * gae
            adv_np[i] = gae
            ret_np[i] = gae + v
    adv = torch.tensor(adv_np, dtype=torch.float32, device=device)
    ret = torch.tensor(ret_np, dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)

    policy.train()
    last = {"L": 0.0, "L_pi": 0.0, "L_v": 0.0, "ent": 0.0}
    for _ in range(epochs):
        for s in range(0, T, minibatch):
            mb = slice(s, s + minibatch)
            log_prob, entropy, value = policy.evaluate(
                x[mb], g_t[mb], wp_t[mb], ac_t[mb], None if wr_t is None else wr_t[mb])
            ratio   = torch.exp(log_prob - oldlp_t[mb])
            unclip  = ratio * adv[mb]
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[mb]
            L_pi    = -torch.min(unclip, clipped).mean()
            L_v     = torch.zeros((), device=device) if det_critic else (value - ret[mb]).pow(2).mean()
            ent     = entropy.mean()
            L       = L_pi + vf_weight * L_v - ent_weight * ent
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            last = {"L": L.item(), "L_pi": L_pi.item(), "L_v": L_v.item(), "ent": ent.item()}

    print(f"[infer_server] update L={last['L']:.4f} (π {last['L_pi']:.4f} V {last['L_v']:.4f} "
          f"H {last['ent']:.3f}) rew μ={rew.mean():.4f} ret μ={ret_np.mean():.4f} "
          f"{len(by_traj)} trajs × {T} grabs", flush=True)
    return {**last, "reward_mean": float(rew.mean()), "ret_mean": float(ret_np.mean()),
            "n_traj": len(by_traj), "n": T}


def op_save(msg):
    path = msg.get("path", args.policy)
    torch.save({"model_state_dict": policy.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "max_wp": MAX_WP}, path)
    print(f"[infer_server] saved policy → {path}", flush=True)
    return {"ok": True}


def handler(msg):
    op = msg.get("op")
    try:
        if op == "infer":     return op_infer(msg)
        if op == "update":    return op_update(msg)
        if op == "save":      return op_save(msg)
        if op == "ping":      return {"ok": True}
        if op == "shutdown":  return "__SHUTDOWN__"
        return {"error": f"unknown op {op!r}"}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    socket_ipc.serve(args.port, handler)
