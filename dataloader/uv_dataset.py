"""
Datasets for UV Mapper and Diffusion model training.

UVMapperDataset  — loads partial npz, returns per-point bin labels
DiffusionDataset — loads full npz, returns rasterized (2, K, K) UV occupancy image

K = 128 bins (must match model/uv_mapper.py K)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

K = 128


def _uv_to_bins(panel_uv):
    """Convert UV in [0,1] to bin indices [0, K-1]."""
    bins = np.clip(np.round(panel_uv * (K - 1)).astype(np.int64), 0, K - 1)
    return bins[:, 0], bins[:, 1]   # u_bins, v_bins


def rasterize_uv(panel_id, panel_uv, k=K):
    """
    Scatter per-point UV onto a (2, K, K) occupancy image.
      panel_id : (N,) int   0=front 1=back
      panel_uv : (N, 2) float  [0,1]
    Returns float32 array (2, K, K)  — channel 0=front, channel 1=back.
    """
    img = np.zeros((2, k, k), dtype=np.float32)
    u_bins = np.clip(np.round(panel_uv[:, 0] * (k - 1)).astype(int), 0, k - 1)
    v_bins = np.clip(np.round(panel_uv[:, 1] * (k - 1)).astype(int), 0, k - 1)
    img[panel_id, u_bins, v_bins] = 1.0   # vectorised numpy fancy indexing
    return img


class UVMapperDataset(Dataset):
    """
    Returns per-point labels for UV Mapper training.
    Loads from partial npz files: pcd_points, panel_id, panel_uv.
    """

    def __init__(self, partial_dir, mode, train_split=0.9):
        assert mode in ("train", "val")
        files = sorted(f for f in os.listdir(partial_dir) if f.endswith(".npz"))
        if train_split >= 1.0:
            # no split — use the ENTIRE folder. Used for dedicated train OR val
            # directories (separate folders → zero leakage), and for --overfit.
            sel = files
        else:
            rng = np.random.default_rng(42)
            rng.shuffle(files)
            split = int(len(files) * train_split)
            sel = files[:split] if mode == "train" else files[split:]
        self.paths = [os.path.join(partial_dir, f) for f in sel]
        print(f"[UVMapper/{mode}] {len(self.paths)} samples ({partial_dir})")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        d        = np.load(self.paths[idx])
        pts      = d["pcd_points"].astype(np.float32)   # (N, 3) centroid-normalised
        centroid = d["centroid"].astype(np.float32)
        panel_id = d["panel_id"].astype(np.int64)
        panel_uv = d["panel_uv"].astype(np.float32)
        if "normals" not in d.files:
            raise KeyError(
                f"{self.paths[idx]} has no 'normals' key. Run "
                "`python data/precompute_normals.py --data <dir>` once to add them."
            )
        normals  = d["normals"].astype(np.float32)      # (N, 3) unit, oriented +z

        # ch 3: z_from_table — height above ground plane (z=0 world). ch 4-6: normals.
        z_table = (pts[:, 2] + centroid[2])[:, None]            # (N, 1)
        pts7 = np.concatenate([pts, z_table, normals], axis=1)  # (N, 7)

        ub, vb = _uv_to_bins(panel_uv)
        return (
            torch.from_numpy(pts7),
            torch.from_numpy(ub),
            torch.from_numpy(vb),
            torch.from_numpy(panel_id),   # kept for future GNN stage
        )

    def recent_weights(self, cutoff, recent_frac):
        """Per-sample weights for a WeightedRandomSampler so that files with npz index
        >= cutoff collectively receive `recent_frac` of the sampling mass each epoch
        (the older files share the remaining 1 - recent_frac). Returns (weights, n_recent,
        n_older). If either group is empty, falls back to uniform weights."""
        idx = np.array(
            [int(os.path.basename(p).split("_")[1].split(".")[0]) for p in self.paths]
        )
        recent = idx >= cutoff
        n_rec, n_old = int(recent.sum()), int((~recent).sum())
        w = np.ones(len(self.paths), dtype=np.float64)
        if n_rec and n_old:
            w[recent]  = recent_frac / n_rec
            w[~recent] = (1.0 - recent_frac) / n_old
        return w, n_rec, n_old


class DiffusionDataset(Dataset):
    """
    Returns rasterized UV occupancy images for diffusion training.
    Loads from full npz files: full_points, panel_id, panel_uv.
    Image shape: (2, K, K) float32  channel 0=front, channel 1=back.
    """

    def __init__(self, full_dir, mode, train_split=0.9):
        assert mode in ("train", "val")
        files = sorted(f for f in os.listdir(full_dir) if f.endswith(".npz"))
        split = int(len(files) * train_split)
        files = files[:split] if mode == "train" else files[split:]
        self.paths = [os.path.join(full_dir, f) for f in files]
        print(f"[Diffusion/{mode}] {len(files)} samples")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        d        = np.load(self.paths[idx])
        panel_id = d["panel_id"].astype(np.int64)
        panel_uv = d["panel_uv"].astype(np.float32)
        return torch.from_numpy(rasterize_uv(panel_id, panel_uv))


if __name__ == "__main__":
    import sys
    partial_dir = sys.argv[1] if len(sys.argv) > 1 else "data/majca/partial"
    full_dir    = sys.argv[2] if len(sys.argv) > 2 else "data/majca/full"

    ds_uv = UVMapperDataset(partial_dir, "train")
    pts, u_bins, v_bins, panel_id = ds_uv[0]
    print("pts:     ", pts.shape)
    print("u_bins:  ", u_bins.shape, u_bins.min().item(), u_bins.max().item())
    print("panel_id:", panel_id.shape, panel_id.unique())

    ds_diff = DiffusionDataset(full_dir, "train")
    img = ds_diff[0]
    print("diff img:", img.shape, "occ front:", img[0].sum().item(),
          "occ back:", img[1].sum().item())
