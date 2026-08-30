# common/io.py

"""
Writing reconstructed volumes out as NIfTI.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import nibabel as nib
from .views import STRUCT

LABELS = {0: 'background', **{i: n for n, i in STRUCT}}


def save_nifti(vol: np.ndarray, path: str, voxel_size: float = 2.0,
               affine: Optional[np.ndarray] = None) -> str:
    """Write a label volume to `path`"""
    if affine is None:
        affine = np.diag([float(voxel_size)] * 3 + [1.0])
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    data = np.asarray(vol).astype(np.uint8)
    img = nib.Nifti1Image(data, affine)
    h = img.header
    h.set_xyzt_units('mm')
    h['cal_min'] = 0.0
    h['cal_max'] = float(max(1, int(data.max())))
    h['intent_code'] = 0

    nib.save(img, path)
    return path


def recon_path(root: str, group: str, sub: str, tag: str) -> str:
    """root/group/group[_SUB]_TAG.nii.gz"""
    name = f"{group}_{sub}_{tag}" if sub else f"{group}_{tag}"
    return os.path.join(root, group, f"{name}.nii.gz")