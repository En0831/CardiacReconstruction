# implicit/data.py

from __future__ import annotations

import os
from typing import List, Sequence, Tuple
 
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
 
import nibabel as nib
 
from common.splits import Split
from common import canonical as C

# ==============================
# helpers
# ==============================
def resample_labels(seg: np.ndarray, grid_size: Sequence[int], num_classes: int, device: str = 'cpu') -> np.ndarray:
    """Trilinear one-hot resample + argmax"""
    t = torch.as_tensor(seg[None], dtype=torch.long, device=device)
    t = F.one_hot(t, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    t = F.interpolate(t, size=list(grid_size), mode='trilinear', align_corners=True)
    return t.argmax(1)[0].cpu().numpy().astype(np.uint8)


def voxel_ids_to_coords(voxel_ids: torch.Tensor, spacing: torch.Tensor) -> torch.Tensor:
    """[N, 3] int voxel ids -> [N, 3] physical mm (align_corners=False convention)."""
    return voxel_ids.to(spacing.dtype) * spacing + spacing / 2.0


def full_grid_coords(grid_size: Sequence[int], spacing: torch.Tensor) -> torch.Tensor:
    """Dense [X, Y, Z, 3] coordinate grid in mm."""
    axes = [torch.arange(s) for s in grid_size]
    ids = torch.stack(torch.meshgrid(axes, indexing='ij'), -1)
    return voxel_ids_to_coords(ids, spacing)


# ==============================
# Training dense prior
# ==============================
class DenseVolumeDataset(Dataset):
    def __init__(self, files: Sequence[str], num_points: int = 64 ** 3, num_classes: int = 6, grid_size: Sequence[int] = (96, 96, 128),
                 voxel_size: float = 2.0, fg_fraction: float = 0.5, device: str = 'cpu', canonical_cfg=None):
        super().__init__()
        self.files = list(files)
        self.num_points = num_points
        self.num_classes = num_classes
        self.grid_size = list(grid_size)
        self.fg_fraction = fg_fraction
 
        self.spacing = torch.full((3,), float(voxel_size), dtype=torch.float32)
        self.image_size = torch.tensor(self.grid_size, dtype=torch.float32) * self.spacing
 
        self.volumes: List[torch.Tensor] = []
        self.casenames: List[str] = []
        for path in self.files:
            seg = nib.load(path).get_fdata().astype(np.uint8)
            seg = resample_labels(seg, self.grid_size, num_classes, device)
            if canonical_cfg is not None:
                gauge = C.build_gauge(seg, canonical_cfg)
                seg = C.resample_to_canonical(seg, gauge, canonical_cfg)
            self.volumes.append(torch.from_numpy(seg).long())
            self.casenames.append(os.path.basename(path))
 
        self.fg_index: List[torch.Tensor] = [torch.nonzero(v > 0, as_tuple=False) for v in self.volumes]
 
    def __len__(self) -> int:
        return len(self.volumes)
 
    def __getitem__(self, item: int):
        vol = self.volumes[item]
        n = self.num_points
        n_fg = int(n * self.fg_fraction)
        n_rand = n - n_fg
 
        parts = []
        if n_rand > 0:
            rand_ids = torch.stack([torch.randint(0, s, (n_rand,)) for s in self.grid_size], dim=-1)
            parts.append(rand_ids)
        if n_fg > 0:
            fg = self.fg_index[item]
            sel = torch.randint(0, fg.shape[0], (n_fg,))
            parts.append(fg[sel])
        voxel_ids = torch.cat(parts, 0)
 
        labels = vol[voxel_ids[:, 0], voxel_ids[:, 1], voxel_ids[:, 2]]
        coords = voxel_ids_to_coords(voxel_ids, self.spacing)
        return {'coords': coords, 'labels': labels, 'caseids': item, 'casenames': self.casenames[item]}
 
    def full_volume(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense GT + dense coordinate grid, for validation / metric computation."""
        return self.volumes[item], full_grid_coords(self.grid_size, self.spacing)


def build_prior_trainset(split: Split, num_points: int = 32 ** 3, num_classes: int = 6, grid_size: Sequence[int] = (96, 96, 128), voxel_size: float = 2.0,
                         fg_fraction: float = 0.5, device: str = 'cpu', canonical_cfg=None) -> DenseVolumeDataset:
    """Build a dataset for training the implicit prior."""
    return DenseVolumeDataset(split.paths('train'), num_points=num_points, num_classes=num_classes, grid_size=grid_size, 
                              voxel_size=voxel_size, fg_fraction=fg_fraction, device=device, canonical_cfg=canonical_cfg)