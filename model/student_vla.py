"""
student_vla.py — transformer imitation/RL policy for single-arm garment manipulation.

No pooling bottleneck. The point cloud becomes a set of tokens; a self-attention encoder gives ONE
contextualised feature per point; the two heads are attention read-outs:

  GRASP : per-point logit  → Categorical over the N points        (dense, multimodal, frame-free)
  DRAG  : max_wp learned "waypoint queries" cross-attend the points, conditioned on the grabbed
          point's feature, → a Gaussian (x,y,z) per query + a per-query "active" logit + a wrist
          ORIENTATION per query. Orientation is a bounded SWING-TWIST about world vertical (swing ≤30°
          off vertical, twist free) and is a STOCHASTIC RL action like position — sampled, in log_prob,
          optimised by PPO (freeze it with sample(rot=False) for a position-only pass). The active
          (contiguous) prefix is the trajectory: waypoints[:k], last = release, the rest = path.
          Variable-length ⇒ no fixed PATH_LEN; a long demo just uses more active waypoints.
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
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Bernoulli

# Wrist-rotation limits, measured from the FIXED flat-base normal (world vertical = gripper straight
# down). SWING = how far the approach axis may tilt off vertical (a cone); TWIST = free spin about it.
SWING_MAX = math.radians(30.0)   # ≤30° tilt off vertical in any direction (circular cone)
TWIST_MAX = math.pi              # ±180° roll about vertical (effectively free)


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

        # WRIST ROTATION per waypoint, as a BOUNDED SWING-TWIST about world vertical (flat-base normal):
        # the head outputs 3 numbers (swing_x, swing_y, twist) in radians. swing=(x,y) tilts the approach
        # axis off vertical and is CLAMPED to a SWING_MAX cone; twist spins about vertical, clamped to
        # ±TWIST_MAX (free). This is a STOCHASTIC RL action like position: wp_rot_log_std gives the
        # exploration noise, sample()/evaluate() include its log_prob, and PPO optimises rotation
        # alongside the drag. The clamp makes the gripper physically reachable (≤30° off straight-down),
        # and the cone is rotationally symmetric so no arbitrary roll convention is needed.
        self.wp_rot_head    = nn.Linear(d_model, 3)
        self.wp_rot_log_std = nn.Parameter(torch.full((max_wp, 3), float(init_log_std)))

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
        wp_rot (B,max_wp,3) = the swing-twist (swing_x,swing_y,twist) MEAN per waypoint. Each query is
        conditioned on the grabbed point's feature, so the motion AND the wrist orientation depend on
        the fold."""
        B = H.shape[0]
        grab_feat = H[torch.arange(B, device=H.device), grab_idx]            # (B,d)
        tgt = self.wp_queries.unsqueeze(0).expand(B, -1, -1) + grab_feat.unsqueeze(1)
        dec = self.decoder(tgt, H)                                           # (B,max_wp,d)
        return self.wp_mean_head(dec), self.wp_stop_head(dec).squeeze(-1), self.wp_rot_head(dec)

    def forward(self, x, grab_idx):
        """BC (grasp teacher-forced to the demo's point): grasp_logits (B,N), wp_mean (B,max_wp,3),
        stop_logits (B,max_wp), wp_rot (B,max_wp,3) swing-twist means."""
        H, grasp_logits = self.encode(x)
        wp_mean, stop_logits, wp_rot = self.decode(H, grab_idx)
        return grasp_logits, wp_mean, stop_logits, wp_rot

    # ── RL / inference API ───────────────────────────────────────────────────────
    def sample(self, x, greedy=False, rot=True):
        """Sample (or greedily pick) an action + its log_prob.
        Returns dict: grab_idx (B,), waypoints (B,max_wp,3), active (B,max_wp), wp_quat (B,max_wp,4),
        wp_rot3 (B,max_wp,3) or None, log_prob (B,).
        Rotation is a STOCHASTIC action (swing-twist, bounded to the SWING_MAX cone): we sample the raw
        (swing_x,swing_y,twist), add its log_prob to the action, and the executed wp_quat is built from
        the CLAMPED sample. wp_rot3 (the raw sample) is stored so PPO can recompute the ratio.
        rot=False FREEZES rotation: identity quats (pure-translation drag), no rotation log_prob, wp_rot3
        None — for the position-only sanity pass (--no-rot)."""
        H, grasp_logits = self.encode(x)
        std = self.wp_log_std.exp()
        if greedy:
            grab_idx = grasp_logits.argmax(-1)
        else:
            grab_idx = Categorical(logits=grasp_logits).sample()
        wp_mean, stop_logits, wp_rot_mean = self.decode(H, grab_idx)
        if greedy:
            waypoints = wp_mean
            active    = (stop_logits > 0).float()
        else:
            waypoints = Normal(wp_mean, std).rsample()
            active    = Bernoulli(logits=stop_logits).sample()
        lp = (Categorical(logits=grasp_logits).log_prob(grab_idx)
              + Normal(wp_mean, std).log_prob(waypoints).sum((-1, -2))
              + Bernoulli(logits=stop_logits).log_prob(active).sum(-1))
        if rot:
            rstd       = self.wp_rot_log_std.exp()
            rot_sample = wp_rot_mean if greedy else Normal(wp_rot_mean, rstd).rsample()
            lp         = lp + Normal(wp_rot_mean, rstd).log_prob(rot_sample).sum((-1, -2))
            wp_quat    = self.rot3_to_quat(rot_sample)                        # clamped to the cone inside
        else:
            rot_sample = None
            wp_quat = torch.zeros(*wp_mean.shape[:-1], 4, device=x.device, dtype=wp_mean.dtype)
            wp_quat[..., 3] = 1.0                                             # identity ⇒ straight-down
        return {"grab_idx": grab_idx, "waypoints": waypoints, "active": active,
                "wp_quat": wp_quat, "wp_rot3": rot_sample, "log_prob": lp}

    def log_prob(self, x, grab_idx, waypoints, active, wp_rot3=None):
        """Recompute log_prob of a stored (grab_idx, waypoints, active[, wp_rot3]) under current weights.
        grab_idx (T,) long, waypoints (T,max_wp,3), active (T,max_wp). If wp_rot3 (T,max_wp,3) is given,
        the rotation log_prob is included (RL rotation); None ⇒ rotation excluded (frozen / --no-rot)."""
        H, grasp_logits = self.encode(x)
        wp_mean, stop_logits, wp_rot_mean = self.decode(H, grab_idx)
        std = self.wp_log_std.exp()
        lp = (Categorical(logits=grasp_logits).log_prob(grab_idx)
              + Normal(wp_mean, std).log_prob(waypoints).sum((-1, -2))
              + Bernoulli(logits=stop_logits).log_prob(active).sum(-1))
        if wp_rot3 is not None:
            lp = lp + Normal(wp_rot_mean, self.wp_rot_log_std.exp()).log_prob(wp_rot3).sum((-1, -2))
        return lp

    # ── critic / PPO API ─────────────────────────────────────────────────────────
    def value(self, x):
        """Critic V(s) (T,) — pooled encoder feature → expected discounted flatness-return."""
        H, _ = self.encode(x)
        return self.value_head(H.mean(1)).squeeze(-1)

    def evaluate(self, x, grab_idx, waypoints, active, wp_rot3=None):
        """PPO: ONE encode → (log_prob, entropy, value) for a stored action. Sharing the encode keeps
        the actor and critic on the same features and halves the forward cost.
          log_prob (T,) : grasp Categorical + drag Gaussian + stop Bernoulli (+ rotation Gaussian if
                          wp_rot3 given — the swing-twist action; None ⇒ frozen rotation, excluded)
          entropy  (T,) : matching entropy sum
          value    (T,) : V(s) from the critic head"""
        H, grasp_logits = self.encode(x)
        wp_mean, stop_logits, wp_rot_mean = self.decode(H, grab_idx)
        std = self.wp_log_std.exp()
        g, n, b = Categorical(logits=grasp_logits), Normal(wp_mean, std), Bernoulli(logits=stop_logits)
        log_prob = (g.log_prob(grab_idx)
                    + n.log_prob(waypoints).sum((-1, -2))
                    + b.log_prob(active).sum(-1))
        entropy  = g.entropy() + n.entropy().sum((-1, -2)) + b.entropy().sum(-1)
        if wp_rot3 is not None:
            r = Normal(wp_rot_mean, self.wp_rot_log_std.exp())
            log_prob = log_prob + r.log_prob(wp_rot3).sum((-1, -2))
            entropy  = entropy + r.entropy().sum((-1, -2))
        value    = self.value_head(H.mean(1)).squeeze(-1)
        return log_prob, entropy, value

    # ── rotation helpers (swing-twist ↔ matrix ↔ quaternion) ─────────────────────
    @staticmethod
    def _rotvec_to_matrix(v):
        """Rodrigues / exp map: a rotation VECTOR v (...,3) = axis·angle → rotation matrix (...,3,3).
        |v| is the angle, v/|v| the axis. As |v|→0 the sin/(1−cos) terms vanish so R→I smoothly."""
        th = torch.linalg.norm(v, dim=-1, keepdim=True)                  # (...,1) angle
        k  = v / th.clamp_min(1e-8)                                      # unit axis
        kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
        zero = torch.zeros_like(kx)
        K = torch.stack([torch.stack([zero, -kz,  ky], -1),
                         torch.stack([ kz, zero, -kx], -1),
                         torch.stack([-ky,  kx, zero], -1)], -2)          # (...,3,3) skew-symmetric
        th_ = th[..., None]                                              # (...,1,1)
        I = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(K)
        return I + torch.sin(th_) * K + (1 - torch.cos(th_)) * (K @ K)

    @classmethod
    def rot3_to_matrix(cls, rot3):
        """Swing-twist (swing_x, swing_y, twist) in radians → rotation matrix (...,3,3), CLAMPED to the
        reachable cone: swing magnitude ≤ SWING_MAX (tilt off vertical), |twist| ≤ TWIST_MAX. Built as
        R = R_swing · R_twist — twist spins about vertical z, swing tilts the approach axis off it, so
        the angle between R's z-axis and vertical equals the (clamped) swing magnitude."""
        sw  = rot3[..., 0:2]
        tw  = rot3[..., 2:3].clamp(-TWIST_MAX, TWIST_MAX)
        mag = torch.linalg.norm(sw, dim=-1, keepdim=True)
        sw  = sw * (SWING_MAX / mag.clamp_min(1e-8)).clamp(max=1.0)       # cap swing to the cone
        zero       = torch.zeros_like(tw)
        swing_vec  = torch.cat([sw, zero], -1)                           # (...,3) tilt in the xy-plane
        twist_vec  = torch.cat([torch.zeros_like(sw), tw], -1)           # (...,3) roll about z
        return cls._rotvec_to_matrix(swing_vec) @ cls._rotvec_to_matrix(twist_vec)

    @classmethod
    def rot3_to_quat(cls, rot3):
        """Swing-twist params → unit quaternion [x,y,z,w] (cone-clamped)."""
        return cls.matrix_to_quat(cls.rot3_to_matrix(rot3))

    @staticmethod
    def _quat_mul(a, b):
        """Hamilton product of two [x,y,z,w] quaternions (...,4)."""
        ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return torch.stack([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz], -1)

    @classmethod
    def clamp_quat_to_cone(cls, q, max_swing=SWING_MAX):
        """Clamp an arbitrary orientation into the reachable cone via swing-twist about vertical z:
        split q into twist (roll about z) · swing (tilt off z), cap the swing angle to max_swing, and
        recompose. Used so BC targets from demos respect the same ≤30° wrist limit. q [x,y,z,w] (...,4)."""
        q = F.normalize(q, dim=-1)
        pz, w = q[..., 2:3], q[..., 3:4]
        twist = F.normalize(torch.cat([torch.zeros_like(pz), torch.zeros_like(pz), pz, w], -1), dim=-1)
        twist_inv = twist * torch.tensor([-1., -1., -1., 1.], device=q.device, dtype=q.dtype)
        swing = cls._quat_mul(q, twist_inv)                              # q = swing · twist
        sxyz, sw = swing[..., 0:3], swing[..., 3:4]
        sn   = torch.linalg.norm(sxyz, dim=-1, keepdim=True)
        ang  = 2 * torch.atan2(sn, sw.abs())                            # swing angle (0..π)
        half = ang.clamp(max=max_swing) * 0.5
        axis = sxyz / sn.clamp_min(1e-8)
        swing_c = torch.cat([axis * torch.sin(half), torch.cos(half)], -1)
        swing_c = torch.where(sn < 1e-8, swing, swing_c)                 # no swing ⇒ leave unchanged
        return F.normalize(cls._quat_mul(swing_c, twist), dim=-1)

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

    def rotation_loss(self, wp_rot, target_quat, active):
        """BC rotation loss: GEODESIC angle (shortest rotation between predicted & demo orientation),
        masked to active waypoints. Computed via matrices, which are unique per rotation — so the
        quaternion q≡-q double cover can't corrupt the loss. The prediction is the swing-twist mean
        (cone-clamped); the demo TARGET is clamped into the SAME cone so BC never chases an unreachable
        orientation. wp_rot (B,W,3), target_quat (B,W,4), active (B,W). Returns mean squared angle."""
        R_pred = self.rot3_to_matrix(wp_rot)
        R_tgt  = self.quat_to_matrix(self.clamp_quat_to_cone(target_quat))
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
