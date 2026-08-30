# lcunet/pose_net.py

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# the six supervised entries of the 12-dim z vector, under the a4c anchor
Z_SLICE = slice(6, 12)
Z_DIM = 6
N_CLASSES = 6
N_VIEWS = 2


def tril_size(n: int) -> int:
    """Free parameters of an n x n lower-triangular matrix: n(n+1)/2."""
    return n * (n + 1) // 2


class ConvBlock2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.SiLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class PoseNet(nn.Module):
    """
    patches -> (mu, L) of a Gaussian over the standardised z.

    - Input channels: N_VIEWS * N_CLASSES one-hot label planes + N_VIEWS val
    - Output: mu [B,6] and L [B,6,6] in standardised space. L is lower-triangular
    """

    def __init__(self, dims: Sequence[int] = (32, 64, 128, 256),
                 n_landmarks: int = 0, cov: str = 'full',
                 hidden: int = 256, min_sigma: float = 0.05,
                 z_mean: Optional[np.ndarray] = None,
                 z_sd: Optional[np.ndarray] = None):
        super().__init__()
        if cov not in ('full', 'diag'):
            raise ValueError(f"cov must be 'full' or 'diag', got {cov!r}")
        self.cov = cov
        self.n_landmarks = int(n_landmarks)
        self.min_log_sigma = math.log(float(min_sigma))

        c_in = N_VIEWS * N_CLASSES + N_VIEWS
        blocks, c = [], c_in
        for d in dims:
            blocks.append(ConvBlock2d(c, d))
            c = d
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(c + self.n_landmarks, hidden), 
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden), 
            nn.SiLU(inplace=True)
        )
        self.mu = nn.Linear(hidden, Z_DIM)
        n_scale = tril_size(Z_DIM) if cov == 'full' else Z_DIM
        self.scale = nn.Linear(hidden, n_scale)

        # start near the marginal: mu = 0 and L = I in standardised space
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)

        # standardisation constants travel with the weights
        self.register_buffer('z_mean', torch.zeros(Z_DIM) if z_mean is None
                             else torch.as_tensor(z_mean, dtype=torch.float32))
        self.register_buffer('z_sd', torch.ones(Z_DIM) if z_sd is None
                             else torch.as_tensor(z_sd, dtype=torch.float32))
        self.register_buffer('lm_mean', torch.zeros(max(self.n_landmarks, 1)))
        self.register_buffer('lm_sd', torch.ones(max(self.n_landmarks, 1)))

        idx = torch.tril_indices(Z_DIM, Z_DIM)
        self.register_buffer('tril_idx', idx)
        self.register_buffer('is_diag', (idx[0] == idx[1]))

    # -- standardisation ------------
    def standardise_z(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.z_mean) / self.z_sd

    def unstandardise_z(self, zs: torch.Tensor) -> torch.Tensor:
        return zs * self.z_sd + self.z_mean

    def set_standardisation(self, z_mean, z_sd, lm_mean=None, lm_sd=None,
                            eps: float = 1e-6) -> None:
        """write the standardisation constants into the buffers"""
        self.z_mean.copy_(torch.as_tensor(z_mean, dtype=torch.float32))
        # clamped to avoid divide-by-zero in standardisation
        self.z_sd.copy_(torch.clamp(torch.as_tensor(z_sd, dtype=torch.float32), min=eps))
        if self.n_landmarks and lm_mean is not None:
            self.lm_mean.copy_(torch.as_tensor(lm_mean, dtype=torch.float32))
            self.lm_sd.copy_(torch.clamp(torch.as_tensor(lm_sd, dtype=torch.float32), min=eps))

    # -- forward ----------------------
    def encode(self, patch: torch.Tensor, valid: torch.Tensor,
               landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """patch [B,V,R,C] int64 labels, valid [B,V,R,C] bool."""
        B, V, R, C = patch.shape
        oh = F.one_hot(patch.long(), N_CLASSES)                 # [B,V,R,C,6]
        oh = oh.permute(0, 1, 4, 2, 3).reshape(B, V * N_CLASSES, R, C).float()
        x = torch.cat([oh, valid.float()], dim=1)               # [B, V*6+V, R, C]
        f = self.pool(self.encoder(x)).flatten(1)
        if self.n_landmarks:
            lm = (landmarks - self.lm_mean) / self.lm_sd
            f = torch.cat([f, lm], dim=1)
        return self.head(f)

    def forward(self, patch: torch.Tensor, valid: torch.Tensor,
                landmarks: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (mu [B,6], L [B,6,6]) in standardised space"""
        h = self.encode(patch, valid, landmarks)
        mu = self.mu(h)
        raw = self.scale(h)
        B = mu.shape[0]
        L = mu.new_zeros(B, Z_DIM, Z_DIM)
        if self.cov == 'full':
            # the diagonal is exponentiated to ensure positivity
            diag = raw[:, self.is_diag].clamp(min=self.min_log_sigma)
            off = raw[:, ~self.is_diag]
            i, j = self.tril_idx
            L[:, i[self.is_diag], j[self.is_diag]] = diag.exp()
            L[:, i[~self.is_diag], j[~self.is_diag]] = off
        else:
            d = raw.clamp(min=self.min_log_sigma).exp()
            L[:, torch.arange(Z_DIM), torch.arange(Z_DIM)] = d
        return mu, L

    # -- loss and sampling ----------------------
    def nll(self, mu: torch.Tensor, L: torch.Tensor, z_std: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Multivariate Gaussian NLL
        0.5 * ||L^-1 (z - mu)||^2 + sum_i log L_ii
        """
        d = (z_std - mu).unsqueeze(-1)                          # [B,6,1]
        u = torch.linalg.solve_triangular(L, d, upper=False).squeeze(-1)
        logdet = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        per = 0.5 * (u ** 2).sum(-1) + logdet                   # [B]
        if mask is None:
            return per.mean()
        keep = mask.all(dim=1) if mask.dim() > 1 else mask
        if not keep.any():
            return per.sum() * 0.0
        return per[keep].mean()

    @torch.no_grad()
    def sample(self, mu: torch.Tensor, L: torch.Tensor, n: int,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """sample n """
        B = mu.shape[0]
        eps = torch.randn(B, n, Z_DIM, dtype=mu.dtype, generator=generator)
        eps = eps.to(mu.device)
        zs = mu.unsqueeze(1) + torch.einsum('bij,bnj->bni', L, eps)
        return self.unstandardise_z(zs)

    @torch.no_grad()
    def sigma(self, L: torch.Tensor) -> torch.Tensor:
        """Marginal sd per dimension in original space"""
        var = (L ** 2).sum(-1)
        return var.sqrt() * self.z_sd


def build_pose_net(cov: str = 'full', n_landmarks: int = 0,
                   dims: Sequence[int] = (32, 64, 128, 256),
                   hidden: int = 256, min_sigma: float = 0.05) -> PoseNet:
    return PoseNet(dims=dims, n_landmarks=n_landmarks, cov=cov,
                   hidden=hidden, min_sigma=min_sigma)