"""
UV Mapper G  (arxiv 2405.10934)

Input : partial point cloud  (B, N, 7)  — [x, y, z, z_from_table, nx, ny, nz]
Output per point:
  phi_u  (B, N, K)  — logits over K discrete u-bins
  phi_v  (B, N, K)  — logits over K discrete v-bins

Architecture:
  ResNet3D encoder  →  (B, N, d)    sparse 3D conv + trilinear interp on xyz (first 3 coords)
  Sinusoidal pos emb (xyz + z_table) + zero-init normal linear proj, added to encoder output
  Transformer (6L, 6H, d=384)  →  (B, N, 384)
  MLP head  →  phi_u, phi_v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

K = 128   # number of UV bins


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding on coords → d_model. n_coords=4 includes z_from_table."""

    def __init__(self, d_model, n_freq=8, n_coords=4):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freq).float()
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(n_coords * n_freq * 2, d_model)

    def forward(self, coords):
        # coords: (B, N, n_coords)
        x   = coords.unsqueeze(-1) * self.freqs        # (B, N, n_coords, n_freq)
        emb = torch.cat([x.sin(), x.cos()], dim=-1)    # (B, N, n_coords, 2*n_freq)
        return self.proj(emb.flatten(-2))               # (B, N, d_model)


def _bnr(c):
    return [nn.BatchNorm1d(c), nn.ReLU()]


class ResNet3DEncoder(nn.Module):
    """
    Sparse ResNet3D encoder.

    Voxelizes (B, N, 3) into a 120x120x60 grid (voxel_size=0.01 m),
    runs sparse 3D conv blocks (120→60→30→15, 60→30→15→8 via stride-2),
    then trilinear-interpolates features back to the original N points.
    Output: (B, N, out_dim) — dense volume (B, out_dim, 8, 15, 15) before interp.
    z range ±0.3 m (garment rarely exceeds 30 cm above/below centroid).
    """

    def __init__(self, voxel_size=0.01,
                 point_range=(-0.6, -0.6, -0.3, 0.6, 0.6, 0.3),
                 out_dim=256, small=False):
        super().__init__()
        self.voxel_size  = voxel_size
        self.point_range = point_range

        gx = int((point_range[3] - point_range[0]) / voxel_size)   # 120
        gy = int((point_range[4] - point_range[1]) / voxel_size)   # 120
        gz = int((point_range[5] - point_range[2]) / voxel_size)   # 60
        self.grid_xyz      = (gx, gy, gz)
        self.spatial_shape = [gz, gy, gx]   # spconv order: (D=Z, H=Y, W=X)

        c0, c1, c2 = (16, 32, 64) if small else (32, 64, 128)
        self.conv = spconv.SparseSequential(
            # stage 0 — initial embed, keeps sparsity (80×80×40)
            spconv.SubMConv3d(1,  c0, 3, padding=1, bias=False), *_bnr(c0),
            spconv.SubMConv3d(c0, c0, 3, padding=1, bias=False), *_bnr(c0),
            # stage 1 — xy: 80→40  z: 40→20
            spconv.SparseConv3d(c0, c1, 3, stride=2, padding=1, bias=False), *_bnr(c1),
            spconv.SubMConv3d(c1, c1, 3, padding=1, bias=False), *_bnr(c1),
            # stage 2 — xy: 40→20  z: 20→10
            spconv.SparseConv3d(c1, c2, 3, stride=2, padding=1, bias=False), *_bnr(c2),
            spconv.SubMConv3d(c2, c2, 3, padding=1, bias=False), *_bnr(c2),
            # stage 3 — xy: 20→10  z: 10→5
            spconv.SparseConv3d(c2, out_dim, 3, stride=2, padding=1, bias=False), *_bnr(out_dim),
            spconv.SubMConv3d(out_dim, out_dim, 3, padding=1, bias=False), *_bnr(out_dim),
            spconv.ToDense(),                           # (B, out_dim, 5, 10, 10)
        )

    def _voxelize(self, pts):
        B, N, _ = pts.shape
        device   = pts.device
        xmin, ymin, zmin = self.point_range[:3]
        gx, gy, gz       = self.grid_xyz

        coords = ((pts - pts.new_tensor([xmin, ymin, zmin]))
                  / self.voxel_size).floor().long()        # (B, N, 3) — X,Y,Z
        coords[..., 0].clamp_(0, gx - 1)
        coords[..., 1].clamp_(0, gy - 1)
        coords[..., 2].clamp_(0, gz - 1)
        czyx = coords[..., [2, 1, 0]]                      # (B, N, 3) — Z,Y,X

        all_c, all_f = [], []
        for b in range(B):
            c = torch.unique(czyx[b], dim=0)               # deduplicate voxels
            M = c.shape[0]
            all_c.append(torch.cat([c.new_full((M, 1), b), c], dim=1))
            all_f.append(c.new_ones(M, 1, dtype=torch.float32))

        return spconv.SparseConvTensor(
            features=torch.cat(all_f),
            indices=torch.cat(all_c).int(),
            spatial_shape=self.spatial_shape,
            batch_size=B,
        )

    def forward(self, pts):
        # pts: (B, N, 3)  — centroid-normalised xyz only (first 3 coords of input)
        dense = self.conv(self._voxelize(pts))              # (B, C, D=Z, H=Y, W=X)

        xmin, ymin, zmin, xmax, ymax, zmax = self.point_range
        nx = 2 * (pts[..., 0] - xmin) / (xmax - xmin) - 1
        ny = 2 * (pts[..., 1] - ymin) / (ymax - ymin) - 1
        nz = 2 * (pts[..., 2] - zmin) / (zmax - zmin) - 1
        # grid_sample: grid dim order (x,y,z) → maps to (W=X, H=Y, D=Z) in dense
        grid = torch.stack([nx, ny, nz], dim=-1).unsqueeze(2).unsqueeze(2)  # (B,N,1,1,3)

        # grid_sample has no bf16 autocast kernel — run it in fp32 explicitly
        with torch.autocast(device_type="cuda", enabled=False):
            feat = F.grid_sample(dense.float(), grid.float(), mode="bilinear",
                                 padding_mode="border", align_corners=True)
        return feat.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, N, C)


class UVMapper(nn.Module):

    def __init__(self, k=K, d_model=384, nhead=6, num_layers=6, small=False):
        super().__init__()
        self.k = k

        if small:
            d_model, nhead, num_layers = 128, 4, 2

        self.encoder = ResNet3DEncoder(out_dim=d_model, small=small)
        self.pos_emb  = SinusoidalPosEmb(d_model)        # xyz + z_table (4 coords)
        # Normals get their OWN linear projection — NOT the sinusoidal pos emb. Unit
        # direction vectors pushed through high freqs (2^0..2^7) become aliased noise
        # that competes with the xyz signal (the past "normals interfere" failure).
        # Zero-init so this branch starts as a no-op: the model learns xyz first and
        # only folds normals in as gradients warrant → cannot hurt other channels early.
        self.normal_emb = nn.Linear(3, d_model)
        nn.init.zeros_(self.normal_emb.weight)
        nn.init.zeros_(self.normal_emb.bias)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True,
            norm_first=True,   # pre-LN: trains deep (6L) from scratch at cold 1e-3
        )                      # without warmup; post-LN stalled at chance at this depth
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        self.head_u = nn.Linear(d_model, k)
        self.head_v = nn.Linear(d_model, k)

    def encode(self, pts):
        """Return per-point features (B, N, d_model) after transformer+MLP — for RL policy."""
        xyz     = pts[:, :, :3]
        coords  = pts[:, :, :4]
        normals = pts[:, :, 4:7]
        f = self.encoder(xyz) + self.pos_emb(coords) + self.normal_emb(normals)
        f = self.transformer(f)
        return self.mlp(f)

    def forward(self, pts):
        # pts: (B, N, 7) — [x, y, z, z_from_table, nx, ny, nz]
        f     = self.encode(pts)                     # (B, N, d_model)
        phi_u = self.head_u(f)                       # (B, N, K)
        phi_v = self.head_v(f)                       # (B, N, K)
        return phi_u, phi_v

    @torch.no_grad()
    def predict_uv(self, pts):
        """Inference: argmax bins → UV in [0,1] + per-point confidence (max softmax prob)."""
        phi_u, phi_v = self.forward(pts)
        ku   = phi_u.argmax(dim=2).float()
        kv   = phi_v.argmax(dim=2).float()
        u    = ku / (self.k - 1)
        v    = kv / (self.k - 1)
        conf = (phi_u.softmax(dim=2).max(dim=2).values *
                phi_v.softmax(dim=2).max(dim=2).values)   # (B, N)
        return torch.stack([u, v], dim=2), conf            # (B, N, 2), (B, N)


def uv_mapper_loss(phi_u, phi_v, gt_u_bin, gt_v_bin):
    """
    phi_u, phi_v : (B, N, K)
    gt_u_bin     : (B, N)  long, values in [0, K-1]
    gt_v_bin     : (B, N)  long
    """
    B, N, K = phi_u.shape
    loss_u = nn.functional.cross_entropy(
        phi_u.reshape(B * N, K), gt_u_bin.reshape(B * N))
    loss_v = nn.functional.cross_entropy(
        phi_v.reshape(B * N, K), gt_v_bin.reshape(B * N))
    return loss_u + loss_v, loss_u, loss_v


if __name__ == "__main__":
    model = UVMapper().cuda()
    x = torch.randn(2, 4096, 7).cuda()   # [xyz, z_from_table, normal_xyz]
    x[:, :, 4:7] = F.normalize(x[:, :, 4:7], dim=-1)
    phi_u, phi_v = model(x)
    print("phi_u:", phi_u.shape)
    print("phi_v:", phi_v.shape)
    n = sum(p.numel() for p in model.parameters())
    print(f"params: {n/1e6:.1f}M")
