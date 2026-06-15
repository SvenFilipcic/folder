"""
Train UV Mapper G.

Features (all toggleable, default ON):
  • AMP        mixed precision (1.5-2x on tensor cores)
  • soft-tau   ordinal soft labels — Gaussian target around true bin instead of
               one-hot, so adjacent bins get partial credit (respects UV's 1D
               geometry). --soft-tau 0 falls back to plain hard CE.
  • rot-aug    random rotation about the vertical axis. UV is a per-particle
               intrinsic label, so a rigid z-rotation of the (centroid-centred)
               cloud leaves every label unchanged → free data multiplication,
               matches deployment (garment dropped at any orientation).
  • EMA        exponential moving average of weights, used for val + best ckpt.
  • ±k metric  reports within-k-bins accuracy alongside exact-bin accuracy
               (exact-bin u_acc understates quality — off-by-one counts as wrong).

Checkpoints saved as checkpoints/uv_mapper_best.pth (+ periodic uv_mapper_epoch_*.pth).

Usage:
    python3 train_uv_mapper.py --data data/majca --epochs 200 --lr 1e-3 --batch 48
    python3 train_uv_mapper.py ... --no-amp --soft-tau 0 --no-rot-aug   # plain hard-CE behaviour
    python3 train_uv_mapper.py --overfit 2 --epochs 500                 # arch sanity check
"""

import sys, os, math, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import resolve_data_dir

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm
import wandb

from dataloader.uv_dataset import UVMapperDataset
from model.uv_mapper import UVMapper

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--resume",     type=str,   default=None)
parser.add_argument("--data",       type=str,   default=None)
parser.add_argument("--epochs",     type=int,   default=200)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--batch",      type=int,   default=48)
parser.add_argument("--workers",    type=int,   default=8)
parser.add_argument("--prefetch",   type=int,   default=4,
                    help="batches each worker prefetches ahead (hides network I/O)")
parser.add_argument("--save-every", type=int,   default=20)
parser.add_argument("--split",      type=float, default=0.99)
parser.add_argument("--new",  type=float, default=0.65,
                    help="fraction of each epoch's training draws taken from recent (high-index) "
                         "npzs; 0 = uniform sampling. e.g. 0.65 = 65%% from idx>=--from")
parser.add_argument("--from", type=int, default=15000, dest="from_idx",
                    help="npz index at/above which a sample counts as 'new' for --new")
parser.add_argument("--val-cap",    type=int,   default=100,
                    help="hard cap on number of val samples (0 = no cap)")
parser.add_argument("--decay",      type=float, default=0.0)
parser.add_argument("--small",      action="store_true",
                    help="use small model (~1.2M params) instead of default (~5.8M)")
parser.add_argument("--overfit",    type=int,   default=0,
                    help="overfit N samples (0 = normal training)")
# ── v2 knobs ──
parser.add_argument("--no-amp",     dest="amp", action="store_false",
                    help="disable mixed precision")
parser.add_argument("--soft-tau",   type=float, default=1.5,
                    help="std (in bins) of Gaussian soft label; 0 = hard CE")
parser.add_argument("--no-rot-aug", dest="rot_aug", action="store_false",
                    help="disable random z-axis rotation augmentation")
parser.add_argument("--jitter",     type=float, default=0.0,
                    help="per-point xyz gaussian jitter std in metres (0 = off)")
parser.add_argument("--ema-decay",  type=float, default=0.999,
                    help="EMA decay; 0 = disable EMA")
parser.add_argument("--pmk",        type=int,   default=2,
                    help="tolerance (in bins) for the within-+/-k accuracy metric")
parser.add_argument("--conf-frac",  type=float, default=0.2,
                    help="fraction of most-confident val points for conf-filtered acc")
parser.set_defaults(amp=True, rot_aug=True)
args = parser.parse_args()
if args.data is None:
    args.data = resolve_data_dir("majca")

device = "cuda" if torch.cuda.is_available() else "cpu"
partial_dir    = os.path.join(args.data, "partial")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

OVERFIT = args.overfit > 0
# augmentation + EMA off in overfit mode (we want to watch raw memorisation)
USE_EMA = (args.ema_decay > 0) and not OVERFIT
USE_AUG = (args.rot_aug or args.jitter > 0) and not OVERFIT


# ── helpers ───────────────────────────────────────────────────────────────────
def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def augment(pts):
    """pts: (B, N, 7) = [x, y, z, z_table, nx, ny, nz]. Cloud is centroid-centred,
    so rotating xy about the origin == rotating about the garment centroid. z and
    z_table are invariant about the vertical axis; the normal's xy rotates WITH the
    cloud (it's a direction glued to the surface), nz is invariant."""
    B = pts.shape[0]
    pts = pts.clone()
    if args.rot_aug:
        theta = torch.rand(B, device=pts.device) * (2 * math.pi)
        c, s = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
        x,  y  = pts[..., 0].clone(), pts[..., 1].clone()
        pts[..., 0] = c * x - s * y
        pts[..., 1] = s * x + c * y
        nx, ny = pts[..., 4].clone(), pts[..., 5].clone()   # rotate normals too
        pts[..., 4] = c * nx - s * ny
        pts[..., 5] = s * nx + c * ny
    if args.jitter > 0:
        noise = torch.randn(B, pts.shape[1], 3, device=pts.device) * args.jitter
        pts[..., :3] += noise
        pts[..., 3]  += noise[..., 2]   # keep z_table = z + const consistent
    return pts


def soft_uv_loss(phi_u, phi_v, u_bins, v_bins, tau):
    """CE with ordinal Gaussian soft labels (tau in bins). tau<=0 → hard CE.
    NOTE: soft-label CE has a non-zero floor (target entropy), so its absolute
    value is NOT comparable to the v1 hard-CE ~9.5 baseline — watch accuracy."""
    K = phi_u.shape[-1]
    if tau <= 0:
        lu = F.cross_entropy(phi_u.reshape(-1, K), u_bins.reshape(-1))
        lv = F.cross_entropy(phi_v.reshape(-1, K), v_bins.reshape(-1))
        return lu + lv, lu, lv
    bins = torch.arange(K, device=phi_u.device).float().view(1, 1, K)

    def term(phi, tgt):
        d = bins - tgt.unsqueeze(-1).float()
        w = torch.exp(-0.5 * (d / tau) ** 2)
        w = w / w.sum(-1, keepdim=True)
        return -(w * F.log_softmax(phi.float(), dim=-1)).sum(-1).mean()

    lu, lv = term(phi_u, u_bins), term(phi_v, v_bins)
    return lu + lv, lu, lv


class EMA:
    """Tracks an exponential moving average over the full state_dict (params +
    buffers). Non-float buffers copied verbatim. Decay is warmed up: at step t the
    effective decay is min(decay, (1+t)/(10+t)), so the EMA tracks the live weights
    closely early instead of sitting at ~random-init for the first epoch (plain
    0.999 over ~310 steps/epoch leaves the shadow 73% init → val stuck at chance).
    Ramps to the full decay over a few thousand steps."""
    def __init__(self, model, decay):
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.detach().clone()
                       for k, v in unwrap(model).state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for k, v in unwrap(model).state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)


# ── data ──────────────────────────────────────────────────────────────────────
if OVERFIT:
    full_dataset = UVMapperDataset(partial_dir, "train", train_split=1.0)
    subset       = Subset(full_dataset, list(range(min(args.overfit, len(full_dataset)))))
    train_loader = DataLoader(subset, batch_size=args.overfit, shuffle=False, num_workers=0)
    val_loader   = train_loader   # same samples — we want to see overfit
    print(f"[OVERFIT MODE] fitting {args.overfit} sample(s) for {args.epochs} epochs")
else:
    # If a "partial_val" folder exists alongside "partial", use it as a dedicated
    # held-out val set automatically (no flag): whole partial/ = train, whole
    # partial_val/ = val → zero leakage. Otherwise fall back to a random --split.
    val_partial = os.path.join(args.data, "partial_val")
    if os.path.isdir(val_partial):
        print(f"[UVMapper] found val folder → {val_partial}")
        train_dataset = UVMapperDataset(partial_dir, "train", train_split=1.0)
        val_dataset   = UVMapperDataset(val_partial,  "val",   train_split=1.0)
    else:
        train_dataset = UVMapperDataset(partial_dir, "train", args.split)
        val_dataset   = UVMapperDataset(partial_dir, "val",   args.split)
    if args.val_cap > 0 and len(val_dataset) > args.val_cap:
        val_dataset = Subset(val_dataset, list(range(args.val_cap)))
        print(f"[UVMapper/val] capped to {args.val_cap} samples")
    loader_kw = dict(pin_memory=True)
    if args.workers > 0:
        loader_kw.update(num_workers=args.workers,
                         prefetch_factor=args.prefetch,
                         persistent_workers=True)
    if args.new > 0:
        w, n_rec, n_old = train_dataset.recent_weights(args.from_idx, args.new)
        if n_rec and n_old:
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                            num_samples=len(train_dataset), replacement=True)
            print(f"[UVMapper/train] weighted sampling: {args.new:.0%} of draws from "
                  f"idx>={args.from_idx} ({n_rec} recent / {n_old} older)")
            train_loader = DataLoader(train_dataset, batch_size=args.batch, sampler=sampler,
                                      **loader_kw)
        else:
            print(f"[UVMapper/train] --new ignored: one group empty "
                  f"({n_rec} recent / {n_old} older) — using uniform shuffle")
            train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                                      **loader_kw)
    else:
        train_loader  = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                                   **loader_kw)
    val_loader    = DataLoader(val_dataset,   batch_size=args.batch, shuffle=False,
                               **loader_kw)

# ── model ─────────────────────────────────────────────────────────────────────
model     = UVMapper(small=args.small).to(device)
start_epoch = 0
lr        = args.lr if not OVERFIT else 1e-3   # higher LR for overfit test
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.decay)
scaler    = torch.amp.GradScaler("cuda", enabled=args.amp)

if args.resume:
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if args.amp and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    if args.lr != 1e-4:
        for pg in optimizer.param_groups:
            pg["lr"] = lr
    print(f"Resumed from epoch {start_epoch}")

if torch.cuda.device_count() > 1 and not OVERFIT:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

ema = EMA(model, args.ema_decay) if USE_EMA else None
scheduler = None
best_val_loss = float("inf")

# ── wandb ─────────────────────────────────────────────────────────────────────
wandb.init(
    project="uv-mapper",
    name=f"overfit_{args.overfit}" if OVERFIT else (f"resume2_{start_epoch}" if args.resume else "train2"),
    config={"epochs": args.epochs, "lr": lr, "batch": args.batch, "overfit": args.overfit,
            "amp": args.amp, "soft_tau": args.soft_tau, "rot_aug": args.rot_aug,
            "jitter": args.jitter, "ema_decay": args.ema_decay, "pmk": args.pmk},
)

# ── training loop ─────────────────────────────────────────────────────────────
print(f"Training UV Mapper v2 for {args.epochs} epochs on {device} "
      f"(amp={args.amp} soft_tau={args.soft_tau} rot_aug={args.rot_aug} "
      f"jitter={args.jitter} ema={args.ema_decay if USE_EMA else 0})")

for epoch in range(start_epoch, args.epochs):
    model.train()
    total_loss = 0.0

    with tqdm(enumerate(train_loader), total=len(train_loader),
              desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:

        for i, (pts, u_bins, v_bins, _panel_id) in pbar:
            pts    = pts.to(device, non_blocking=True)
            u_bins = u_bins.to(device, non_blocking=True)
            v_bins = v_bins.to(device, non_blocking=True)
            if USE_AUG:
                pts = augment(pts)

            with torch.amp.autocast("cuda", enabled=args.amp):
                phi_u, phi_v = model(pts)
                loss, lu, lv = soft_uv_loss(phi_u, phi_v, u_bins, v_bins, args.soft_tau)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            total_loss += loss.item()
            avg = total_loss / (i + 1)
            with torch.no_grad():
                pu, pv = phi_u.argmax(2), phi_v.argmax(2)
                tr_u_acc = (pu == u_bins).float().mean().item()
                tr_v_acc = (pv == v_bins).float().mean().item()
                tr_u_pm  = ((pu - u_bins).abs() <= args.pmk).float().mean().item()
            pbar.set_postfix(avg=f"{avg:.4f}",
                             lu=f"{lu.item():.3f}", lv=f"{lv.item():.3f}",
                             u_acc=f"{tr_u_acc:.3f}", v_acc=f"{tr_v_acc:.3f}",
                             u_pm=f"{tr_u_pm:.3f}")

            if not OVERFIT and (i + 1) % 50 == 0:
                wandb.log({"epoch": epoch, "train_loss": loss.item(),
                           "loss_u": lu.item(), "loss_v": lv.item(),
                           "lr": optimizer.param_groups[0]["lr"]})

    # ── validation (on EMA weights if enabled) ─────────────────────────────────
    if ema is not None:
        backup = copy.deepcopy(unwrap(model).state_dict())
        unwrap(model).load_state_dict(ema.shadow)

    model.eval()
    val_loss = val_u_acc = val_v_acc = val_u_pm = val_v_pm = 0.0
    n_val = 0
    conf_all, cu_all, cv_all, pmu_all, pmv_all = [], [], [], [], []

    with torch.no_grad():
        for pts, u_bins, v_bins, _panel_id in val_loader:
            pts    = pts.to(device, non_blocking=True)
            u_bins = u_bins.to(device, non_blocking=True)
            v_bins = v_bins.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=args.amp):
                phi_u, phi_v = model(pts)
                loss, lu, lv = soft_uv_loss(phi_u, phi_v, u_bins, v_bins, args.soft_tau)
            val_loss  += loss.item()
            pu, pv = phi_u.argmax(2), phi_v.argmax(2)
            val_u_acc += (pu == u_bins).float().mean().item()
            val_v_acc += (pv == v_bins).float().mean().item()
            val_u_pm  += ((pu - u_bins).abs() <= args.pmk).float().mean().item()
            val_v_pm  += ((pv - v_bins).abs() <= args.pmk).float().mean().item()
            n_val += 1

            # per-point confidence = softmax_max(u) * softmax_max(v)  (pipeline def)
            conf = (F.softmax(phi_u.float(), -1).amax(-1) *
                    F.softmax(phi_v.float(), -1).amax(-1)).flatten().cpu()
            conf_all.append(conf)
            cu_all.append((pu == u_bins).flatten().cpu())
            cv_all.append((pv == v_bins).flatten().cpu())
            pmu_all.append(((pu - u_bins).abs() <= args.pmk).flatten().cpu())
            pmv_all.append(((pv - v_bins).abs() <= args.pmk).flatten().cpu())

    val_loss /= n_val
    u_acc, v_acc = val_u_acc / n_val, val_v_acc / n_val
    u_pm,  v_pm  = val_u_pm  / n_val, val_v_pm  / n_val

    # confidence-filtered accuracy on the top conf-frac most-confident points
    # (this is what the anchor stage actually sees: it keeps high-conf preds)
    conf_all = torch.cat(conf_all)
    cu_all, cv_all = torch.cat(cu_all), torch.cat(cv_all)
    pmu_all, pmv_all = torch.cat(pmu_all), torch.cat(pmv_all)
    k = max(1, int(len(conf_all) * args.conf_frac))
    top = conf_all.topk(k).indices
    cf_u, cf_v   = cu_all[top].float().mean().item(),  cv_all[top].float().mean().item()
    cf_um, cf_vm = pmu_all[top].float().mean().item(), pmv_all[top].float().mean().item()

    wandb.log({"epoch": epoch, "val_loss": val_loss,
               "val_u_acc": u_acc, "val_v_acc": v_acc,
               "val_u_pm": u_pm, "val_v_pm": v_pm,
               "val_cf_u_acc": cf_u, "val_cf_v_acc": cf_v,
               "val_cf_u_pm": cf_um, "val_cf_v_pm": cf_vm,
               "lr": optimizer.param_groups[0]["lr"]})

    # weights to persist = the ones we just evaluated (EMA if enabled, else live)
    eval_state = copy.deepcopy(ema.shadow) if ema is not None else copy.deepcopy(unwrap(model).state_dict())

    if OVERFIT:
        print(f"  [{epoch+1:>4}]  loss: {val_loss:.4f}  "
              f"u_acc: {u_acc:.3f}  v_acc: {v_acc:.3f}  (u±{args.pmk}: {u_pm:.3f})")
        is_last = (epoch + 1 == args.epochs)
        if u_acc > 0.99 and v_acc > 0.99:
            print("  ✓ Overfit achieved — architecture OK")
            path = os.path.join(CHECKPOINT_DIR, f"uv_mapper_overfit_epoch_{epoch+1}.pth")
            torch.save({"epoch": epoch, "model_state_dict": eval_state}, path)
            print(f"  saved {os.path.basename(path)}")
            break
        if is_last:
            path = os.path.join(CHECKPOINT_DIR, f"uv_mapper_overfit_epoch_{epoch+1}.pth")
            torch.save({"epoch": epoch, "model_state_dict": eval_state}, path)
            print(f"  saved {os.path.basename(path)}")
    else:
        print(f"  val loss: {val_loss:.4f}  u_acc: {u_acc:.3f}  v_acc: {v_acc:.3f}  "
              f"(within±{args.pmk}: u {u_pm:.3f} / v {v_pm:.3f})")
        print(f"  conf-top{int(args.conf_frac*100)}%:  u_acc {cf_u:.3f} / v_acc {cf_v:.3f}  "
              f"(within±{args.pmk}: u {cf_um:.3f} / v {cf_vm:.3f})")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(CHECKPOINT_DIR, "uv_mapper_best.pth")
            torch.save({"epoch": epoch,
                        "model_state_dict":     eval_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict":    scaler.state_dict()}, path)
            print(f"  saved best (val_loss={val_loss:.4f})")

    # restore live weights so training continues from them (not the EMA)
    if ema is not None:
        unwrap(model).load_state_dict(backup)

    if scheduler:
        scheduler.step()

    if not OVERFIT and (epoch + 1) % args.save_every == 0:
        path = os.path.join(CHECKPOINT_DIR, f"uv_mapper_epoch_{epoch+1}.pth")
        torch.save({"epoch": epoch,
                    "model_state_dict":     eval_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict":    scaler.state_dict()}, path)
        print(f"  saved {os.path.basename(path)}")

wandb.finish()
