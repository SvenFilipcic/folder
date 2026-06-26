"""
init_student.py — write a RANDOM-INITIALISED StudentVLA checkpoint so the RL loop can run from
scratch (no BC / no demos).  'infer' env.

    conda run -n infer python init_student.py [--out checkpoints/student_vla.pth] [--init-log-std -0.7]

Why this exists: student_infer.py and student_update.py both torch.load(--policy) and crash if the
file is missing.  Normally the file is the BC checkpoint from train_student.py.  For the FROM-SCRATCH
RL sanity run (does _flat_reward + the grab→drag primitive actually drive de-wrinkling?) there are no
demos yet, so we seed one fresh random policy here.

--init-log-std widens the action exploration: waypoints sample from Normal(mean, exp(log_std)).  The
default StudentVLA uses -2.0 (std≈0.135 m), fine once BC has placed the means sensibly, but too timid
around the random ~zero means of a cold start.  -0.7 ≈ 0.5 m lets the arm actually drag across the
cloth so PPO has something to reinforce.
"""
import os, sys, argparse
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import il_dataset
from model.student_vla import StudentVLA

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(_ROOT, "checkpoints", "student_vla.pth"))
ap.add_argument("--max-wp", type=int, default=il_dataset.MAX_WP)
ap.add_argument("--init-log-std", type=float, default=-0.7,
                help="initial wp_log_std (exploration width); -0.7≈0.5 m vs the -2.0≈0.135 m default")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--force", action="store_true", help="overwrite an existing checkpoint")
args = ap.parse_args()

if os.path.exists(args.out) and not args.force:
    sys.exit(f"[init_student] {args.out} already exists — pass --force to overwrite "
             f"(refusing to clobber a possibly-trained checkpoint)")

torch.manual_seed(args.seed)
model = StudentVLA(max_wp=args.max_wp, init_log_std=args.init_log_std)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save({"model_state_dict": model.state_dict(), "max_wp": args.max_wp}, args.out)
n = sum(p.numel() for p in model.parameters())
print(f"[init_student] wrote random policy ({n/1e6:.1f}M params, max_wp={args.max_wp}, "
      f"init_log_std={args.init_log_std}) → {args.out}")
