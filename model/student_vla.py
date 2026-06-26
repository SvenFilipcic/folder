"""
student_vla.py — transformer imitation/RL policy for single-arm garment manipulation.

No pooling bottleneck. The point cloud becomes a set of tokens; a self-attention encoder gives ONE
contextualised feature per point; the two heads are attention read-outs:

  GRASP : per-point logit  → Categorical over the N points        (dense, multimodal, frame-free)
  DRAG  : max_wp learned "waypoint queries" cross-attend the points, conditioned on the grabbed
          point's feature, → a Gaussian (x,y,z) per query + a per-query "active" logit + a wrist
          ORIENTATION per query (6D rot rep → quaternion). The active (contiguous) prefix is the
          trajectory: waypoints[:k], last = release, the rest = path. Variable-length ⇒ no fixed
          PATH_LEN; a long demo just uses more active waypoints. Orientation is BC-only (deterministic
          in RL); the grab pose itself comes from the grabbed point's surface normal, not a head.
  VALUE : a critic head pools the encoder features → one scalar V(s) = the expected discounted
          flatness-return from this state. Only used by PPO/GAE in student_update.py (multi-step
          credit assignment); inference (sample) never touches it.

Input  (source-agnostic): il_dataset.featurize(state) → (B, N, 9) = [x,y,z,u,v,nx,ny,nz,z_abs].
Frame  (waypoints): XY relative to the cloud centroid, Z absolute (table=0) — matches the state
       input and head_RL._execute_drag_path.

Interfaces kept for head_RL / workers:
  forward(x, grab_idx)                         BC (teacher-forced grasp): grasp_logits, wp_mean, stop_logits
  sample(x, greedy)                            RL/exec: dict(grab_idx, waypoints, active, log_prob)
  log_prob(x, grab_idx, waypoints, active)     REINFORCE: recompute log_prob of a stored action

N≈4096 (head_RL.N_PCD), batch-1 inference → attention is cheap; sdpa (flash) keeps it off the books.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Bernoulli


class StudentVLA(nn.Module):
    def __init__(self, in_dim=9, d_model=384, n_enc=6, n_dec=4, n_heads=8,
                 max_wp=8, init_log_std=-2.0):
        super().__init__()
        self.max_wp  = max_wp
        self.d_model = d_model

        # tokenize: per-point embedding + a positional embedding from xyz (the geometry a pool destroys)
        self.embed   = nn.Linear(in_dim, d_model)
        self.pos_mlp = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.0,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_enc)

        # GRASP: one logit per point → Categorical over the N points
        self.grasp_head = nn.Linear(d_model, 1)

        # DRAG: learned waypoint queries → decoder cross-attends the points → per-query Gaussian + stop
        self.wp_queries = nn.Parameter(torch.randn(max_wp, d_model) * 0.02)
        dec = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.0,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.decoder      = nn.TransformerDecoder(dec, n_dec)
        self.wp_mean_head = nn.Linear(d_model, 3)               # (x,y,z) mean per query
        self.wp_stop_head = nn.Linear(d_model, 1)              # "this waypoint is real" logit per query
        self.wp_log_std   = nn.Parameter(torch.full((max_wp, 3), float(init_log_std)))

        # WRIST ROTATION per waypoint: predict a 6D continuous rotation rep (Zhou et al. 2019) — two
        # 3-vectors → Gram-Schmidt → rotation matrix → quaternion. 6D is the smallest representation
        # with NO discontinuities, so the net learns it smoothly (raw-quaternion regression suffers the
        # q≡-q double cover + unit-norm constraint). Grasp orientation is NOT predicted here: the robot
        # grabs along the surface normal, so the grab pose comes from the grabbed point's normal.
        # DETERMINISTIC in RL: used at execution, but never sampled and never in log_prob (BC learns it).
        self.wp_rot_head  = nn.Linear(d_model, 6)

        # VALUE (critic): mean-pool the encoder features → one scalar V(s). Pooling is fine here — a
        # value is a single number, not an action, so the per-point detail the policy needs is moot.
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    # ── encoder / decoder ────────────────────────────────────────────────────────
    def encode(self, x):
        """x (B,N,in_dim) → H (B,N,d), grasp_logits (B,N)."""
        h = self.embed(x) + self.pos_mlp(x[..., :3])
        H = self.encoder(h)
        return H, self.grasp_head(H).squeeze(-1)

    def decode(self, H, grab_idx):
        """H (B,N,d), grab_idx (B,) long → wp_mean (B,max_wp,3), stop_logits (B,max_wp),
        wp_rot6d (B,max_wp,6). Each query is conditioned on the grabbed point's feature, so the
        motion AND the wrist orientation depend on the fold."""
        B = H.shape[0]
        grab_feat = H[torch.arange(B, device=H.device), grab_idx]            # (B,d)
        tgt = self.wp_queries.unsqueeze(0).expand(B, -1, -1) + grab_feat.unsqueeze(1)
        dec = self.decoder(tgt, H)                                           # (B,max_wp,d)
        return self.wp_mean_head(dec), self.wp_stop_head(dec).squeeze(-1), self.wp_rot_head(dec)

    def forward(self, x, grab_idx):
        """BC (grasp teacher-forced to the demo's point): grasp_logits (B,N), wp_mean (B,max_wp,3),
        stop_logits (B,max_wp), wp_rot6d (B,max_wp,6)."""
        H, grasp_logits = self.encode(x)
        wp_mean, stop_logits, wp_rot6d = self.decode(H, grab_idx)
        return grasp_logits, wp_mean, stop_logits, wp_rot6d

    # ── RL / inference API ───────────────────────────────────────────────────────
    def sample(self, x, greedy=False):
        """Sample (or greedily pick) an action + its log_prob.
        Returns dict: grab_idx (B,), waypoints (B,max_wp,3), active (B,max_wp), wp_quat (B,max_wp,4),
        log_prob (B,). wp_quat is the per-waypoint wrist orientation [x,y,z,w] — DETERMINISTIC: taken
        straight from the head (never sampled, not in log_prob, so RL doesn't explore/optimise it)."""
        H, grasp_logits = self.encode(x)
        std = self.wp_log_std.exp()
        if greedy:
            grab_idx = grasp_logits.argmax(-1)
        else:
            grab_idx = Categorical(logits=grasp_logits).sample()
        wp_mean, stop_logits, wp_rot6d = self.decode(H, grab_idx)
        if greedy:
            waypoints = wp_mean
            active    = (stop_logits > 0).float()
        else:
            waypoints = Normal(wp_mean, std).rsample()
            active    = Bernoulli(logits=stop_logits).sample()
        lp = (Categorical(logits=grasp_logits).log_prob(grab_idx)
              + Normal(wp_mean, std).log_prob(waypoints).sum((-1, -2))
              + Bernoulli(logits=stop_logits).log_prob(active).sum(-1))
        wp_quat = self.matrix_to_quat(self.rot6d_to_matrix(wp_rot6d))         # deterministic orientation
        return {"grab_idx": grab_idx, "waypoints": waypoints, "active": active,
                "wp_quat": wp_quat, "log_prob": lp}

    def log_prob(self, x, grab_idx, waypoints, active):
        """Recompute log_prob of a stored (grab_idx, waypoints, active) under current weights (REINFORCE).
        grab_idx (T,) long, waypoints (T,max_wp,3), active (T,max_wp). Rotation is excluded by design."""
        H, grasp_logits = self.encode(x)
        wp_mean, stop_logits, _ = self.decode(H, grab_idx)
        std = self.wp_log_std.exp()
        return (Categorical(logits=grasp_logits).log_prob(grab_idx)
                + Normal(wp_mean, std).log_prob(waypoints).sum((-1, -2))
                + Bernoulli(logits=stop_logits).log_prob(active).sum(-1))

    # ── critic / PPO API ─────────────────────────────────────────────────────────
    def value(self, x):
        """Critic V(s) (T,) — pooled encoder feature → expected discounted flatness-return."""
        H, _ = self.encode(x)
        return self.value_head(H.mean(1)).squeeze(-1)

    def evaluate(self, x, grab_idx, waypoints, active):
        """PPO: ONE encode → (log_prob, entropy, value) for a stored action. Sharing the encode keeps
        the actor and critic on the same features and halves the forward cost.
          log_prob (T,) : same decomposition as log_prob()/sample()
          entropy  (T,) : true entropy of grasp Categorical + drag Gaussian + stop Bernoulli
          value    (T,) : V(s) from the critic head"""
        H, grasp_logits = self.encode(x)
        wp_mean, stop_logits, _ = self.decode(H, grab_idx)
        std = self.wp_log_std.exp()
        g, n, b = Categorical(logits=grasp_logits), Normal(wp_mean, std), Bernoulli(logits=stop_logits)
        log_prob = (g.log_prob(grab_idx)
                    + n.log_prob(waypoints).sum((-1, -2))
                    + b.log_prob(active).sum(-1))
        entropy  = g.entropy() + n.entropy().sum((-1, -2)) + b.entropy().sum(-1)
        value    = self.value_head(H.mean(1)).squeeze(-1)
        return log_prob, entropy, value

    # ── rotation helpers (6D rep ↔ matrix ↔ quaternion) ──────────────────────────
    @staticmethod
    def rot6d_to_matrix(d6):
        """6D continuous rotation rep (Zhou et al. 2019) → rotation matrix (...,3,3) via Gram-Schmidt:
        read d6 as two 3-vectors a1,a2. b1 = a1 normalised; b2 = (a2 minus its component along b1),
        normalised → perpendicular to b1; b3 = b1×b2. The 3 perpendicular unit axes ARE the rotation.
        Always valid + continuous, so the net learns it smoothly."""
        a1, a2 = d6[..., 0:3], d6[..., 3:6]
        b1 = F.normalize(a1, dim=-1)
        a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(a2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)                         # (...,3,3), columns = axes

    @staticmethod
    def quat_to_matrix(q):
        """Unit quaternion [x,y,z,w] (...,4) → rotation matrix (...,3,3). Used to turn demo-target
        quaternions into matrices for the (sign-invariant) geodesic loss."""
        q = F.normalize(q, dim=-1)
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        R = torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
            2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)], dim=-1)
        return R.reshape(q.shape[:-1] + (3, 3))

    @staticmethod
    def matrix_to_quat(R):
        """Rotation matrix (...,3,3) → unit quaternion [x,y,z,w] (...,4). Shepperd's stable method:
        build the quaternion from whichever of {trace, m00, m11, m22} is largest to avoid divide-by-~0."""
        m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
        m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
        m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
        t = m00 + m11 + m22
        qw = torch.stack([m21 - m12, m02 - m20, m10 - m01, 1 + t], -1)           # trace-dominant
        qx = torch.stack([1 + m00 - m11 - m22, m01 + m10, m02 + m20, m21 - m12], -1)
        qy = torch.stack([m01 + m10, 1 - m00 + m11 - m22, m12 + m21, m02 - m20], -1)
        qz = torch.stack([m02 + m20, m12 + m21, 1 - m00 - m11 + m22, m10 - m01], -1)
        cand = torch.stack([qw, qx, qy, qz], dim=-2)                              # (...,4 cases,4)
        idx  = torch.stack([t, m00, m11, m22], dim=-1).argmax(-1)                 # most stable case
        q = torch.gather(cand, -2, idx[..., None, None].expand(idx.shape + (1, 4))).squeeze(-2)
        return F.normalize(q, dim=-1)

    def rotation_loss(self, wp_rot6d, target_quat, active):
        """BC rotation loss: GEODESIC angle (shortest rotation between predicted & demo orientation),
        masked to active waypoints. Computed via matrices, which are unique per rotation — so the
        quaternion q≡-q double cover can't corrupt the loss. wp_rot6d (B,W,6), target_quat (B,W,4),
        active (B,W). Returns mean squared angle (radians²) over active waypoints."""
        R_pred = self.rot6d_to_matrix(wp_rot6d)
        R_tgt  = self.quat_to_matrix(target_quat)
        Rrel   = R_pred.transpose(-1, -2) @ R_tgt                                 # relative rotation
        cos    = ((Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]) - 1) * 0.5
        ang    = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6))                     # (B,W) geodesic angle
        return (ang.pow(2) * active).sum() / active.sum().clamp_min(1.0)

    # ── decode helper (trajectory from queries) ──────────────────────────────────
    @staticmethod
    def traj_split(waypoints, active):
        """waypoints (max_wp,3), active (max_wp,) → (release [x,y,z], path [[x,y,z]...]).
        Trajectory = the CONTIGUOUS active prefix (≥1 waypoint); last = release, the rest = path."""
        wp = np.asarray(waypoints, np.float32)
        ac = np.asarray(active, np.float32)
        k = 0
        for a in ac:
            if a > 0.5:
                k += 1
            else:
                break
        k = max(1, k)
        traj = wp[:k]
        release = traj[-1].tolist()
        path    = [p.tolist() for p in traj[:-1]]
        return release, path
