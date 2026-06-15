"""
RL policy head for garment smoothing — sits on top of frozen UV Mapper features.

Head 1 (single Franka, active now):
  grasp_head_1  : global feat → N logits        (which point to grab)
  action_head_1 : global feat → (dx, dy, z)     (drag displacement + height)

Head 2 (second Franka, architecture wired, not yet trained):
  grasp_head_2  : [global feat + point1 feat] → N logits
  action_head_2 : [global feat + point1 feat] → (dx2, dy2, z2)

Action bounds:
  dx, dy : ±MAX_DRAG  (tanh-bounded)
  z      : Z_MIN..Z_MAX  (sigmoid-bounded)
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

MAX_DRAG = 0.5
Z_MIN    = 0.05
Z_MAX    = 0.40


class RLPolicy(nn.Module):
    def __init__(self, d_model=384, n_points=4096):
        super().__init__()
        self.n_points = n_points
        self.d_model  = d_model

        self.shared = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
        )

        # ── head 1 ───────────────────────────────────────────────────────────
        self.grasp_head_1  = nn.Linear(256, n_points)
        self.action_head_1 = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 6),       # mean_dx, mean_dy, mean_z, log_std_dx, log_std_dy, log_std_z
        )

        # ── head 2 (conditioned on head-1 grasp choice) ──────────────────────
        self.grasp_head_2  = nn.Linear(256 + d_model, n_points)
        self.action_head_2 = nn.Sequential(
            nn.Linear(256 + d_model, 64), nn.ReLU(),
            nn.Linear(64, 6),
        )

    # ── internals ────────────────────────────────────────────────────────────

    def _global(self, f):
        """(B, N, d_model) → (B, 256) global feature via max pool."""
        return self.shared(f.max(dim=1).values)

    def _action_from(self, head, feat):
        """
        Run action head on feat, sample (dx,dy,z) from Gaussian.
        Returns: bounded action (B,3), raw pre-bound sample (B,3), log_prob (B,).
        Raw samples are stored so rl_update can recompute log_prob exactly.
        """
        params  = head(feat)                         # (B, 6)
        mean    = params[:, :3]
        log_std = params[:, 3:].clamp(-4, 0)
        dist    = Normal(mean, log_std.exp())
        raw     = dist.rsample()                     # (B, 3)
        log_p   = dist.log_prob(raw).sum(-1)         # (B,)
        bounded = torch.stack([
            torch.tanh(raw[:, 0]) * MAX_DRAG,
            torch.tanh(raw[:, 1]) * MAX_DRAG,
            torch.sigmoid(raw[:, 2]) * (Z_MAX - Z_MIN) + Z_MIN,
        ], dim=1)                                    # (B, 3)
        return bounded, raw, log_p

    # ── public API ────────────────────────────────────────────────────────────

    def sample_head1(self, f, greedy=False):
        """
        Sample head-1 action.
        f: (B, N, d_model) UV Mapper features (detached from UV Mapper graph).
        Returns dict with action components, raw samples, log_prob.
        greedy=True for evaluation (argmax grasp, mean action).
        """
        g = self._global(f)                              # (B, 256)
        logits = self.grasp_head_1(g)                    # (B, N)

        if greedy:
            pcd_idx  = logits.argmax(dim=1)
            lp_grasp = logits.new_zeros(logits.shape[0])
        else:
            cat      = Categorical(logits=logits)
            pcd_idx  = cat.sample()                      # (B,)
            lp_grasp = cat.log_prob(pcd_idx)             # (B,)

        bounded, raw, lp_act = self._action_from(self.action_head_1, g)

        return {
            "pcd_idx":  pcd_idx,          # (B,) long
            "dx":       bounded[:, 0],    # (B,) bounded
            "dy":       bounded[:, 1],
            "z":        bounded[:, 2],
            "raw_dx":   raw[:, 0],        # (B,) raw — stored for log_prob recompute
            "raw_dy":   raw[:, 1],
            "raw_z":    raw[:, 2],
            "log_prob": lp_grasp + lp_act,
        }

    def log_prob_head1(self, f, pcd_idx, raw_dx, raw_dy, raw_z):
        """
        Recompute log_prob of a stored action under the current policy weights.
        Used in rl_update.py for REINFORCE loss with updated weights.
        f        : (B, N, d_model)
        pcd_idx  : (B,) long
        raw_*    : (B,) float — pre-bound raw samples from inference time
        """
        g      = self._global(f)
        logits = self.grasp_head_1(g)
        lp_grasp = Categorical(logits=logits).log_prob(pcd_idx)

        params   = self.action_head_1(g)
        mean     = params[:, :3]
        log_std  = params[:, 3:].clamp(-4, 0)
        raw      = torch.stack([raw_dx, raw_dy, raw_z], dim=1)
        lp_act   = Normal(mean, log_std.exp()).log_prob(raw).sum(-1)
        return lp_grasp + lp_act                         # (B,)

    def sample_head2(self, f, grasp1_idx, shared_feat=None):
        """
        Sample head-2 action conditioned on head-1 grasp choice.
        grasp1_idx: (B,) long
        shared_feat: (B,256) from a previous _global call (pass to avoid recomputing).
        """
        if shared_feat is None:
            shared_feat = self._global(f)
        B      = f.shape[0]
        pt1    = f[torch.arange(B, device=f.device), grasp1_idx]   # (B, d_model)
        cond   = torch.cat([shared_feat, pt1], dim=1)               # (B, 256+d_model)
        logits = self.grasp_head_2(cond)                            # (B, N)
        cat    = Categorical(logits=logits)
        pcd_idx  = cat.sample()
        lp_grasp = cat.log_prob(pcd_idx)
        bounded, raw, lp_act = self._action_from(self.action_head_2, cond)
        return {
            "pcd_idx":  pcd_idx,
            "dx":       bounded[:, 0],
            "dy":       bounded[:, 1],
            "z":        bounded[:, 2],
            "raw_dx":   raw[:, 0],
            "raw_dy":   raw[:, 1],
            "raw_z":    raw[:, 2],
            "log_prob": lp_grasp + lp_act,
        }
