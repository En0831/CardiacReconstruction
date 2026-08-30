# camus/data.py

from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib

from common.views import View2D, frame_2d, inplane_coords, px_per_voxel

# CAMUS label -> 6-class model label
CAMUS_REMAP: Dict[int, int] = {1: 1, 2: 2, 3: 4}
CAMUS_OBSERVED = frozenset(CAMUS_REMAP.values())
FALLBACK_MM = 0.308     # mm/px

LATERAL_FLIP = True
TAG_TO_VIEW = {'a4c': '4CH', 'a2c': '2CH'}


def list_patients(root: str) -> List[str]:
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, 'patient*')) if os.path.isdir(p))


def load_mask(path: str, fallback_mm: float = FALLBACK_MM) -> Tuple[np.ndarray, float]:
    """(mask [H,W] uint8 in CAMUS labels, mm_per_px). Spacing from the file."""
    img = nib.load(path)
    m = np.squeeze(img.get_fdata()).astype(np.uint8)
    if m.ndim != 2:
        raise ValueError(f"{os.path.basename(path)}: expected a 2D mask, got {m.shape}")
    sp = [float(z) for z in img.header.get_zooms()[:2] if float(z) > 0]
    mm = sp[0] if (sp and not all(abs(s - 1.0) < 1e-6 for s in sp)) else float(fallback_mm)
    return m, mm


def view2d_from_mask(mask: np.ndarray, mm_per_px: float, voxel_size: float = 2.0,
                     flip: bool = LATERAL_FLIP) -> View2D:
    """CAMUS mask -> View2D, via the same frame_2d / inplane_coords as synthetic data"""
    remapped = np.zeros_like(mask)
    for src, dst in CAMUS_REMAP.items():
        remapped[mask == src] = dst
    if flip:
        remapped = remapped[:, ::-1].copy()     # flip to match with training data

    ppv = px_per_voxel(voxel_size, mm_per_px)
    apex, e_v_2d, lv_len = frame_2d(remapped)
    alpha, beta = inplane_coords(remapped, apex, e_v_2d, ppv)
    return View2D(mask=remapped, alpha=alpha, beta=beta, apex2d=apex, e_v_2d=e_v_2d,
                  mm_per_px=float(mm_per_px), lv_len_px=lv_len,
                  observed_classes=CAMUS_OBSERVED, valid=None, apex3d=None)


def case_paths(root: str, pid: str, phase: str, suffix: str = '_pred') -> Dict[str, str]:
    """{'a4c': .../pid_4CH_<phase>_pred.nii.gz, 'a2c': ...}. phase in {ED, ES}"""
    return {tag: os.path.join(root, pid, f"{pid}_{v}_{phase}{suffix}.nii.gz")
            for tag, v in TAG_TO_VIEW.items()}


def load_case(root: str, pid: str, phase: str, voxel_size: float = 2.0,
              fallback_mm: float = FALLBACK_MM, suffix: str = '_pred') -> Dict[str, View2D]:
    """Both views of one patient at one phase. Raises if either view is missing."""
    out = {}
    for tag, path in case_paths(root, pid, phase, suffix).items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{pid} {phase}: missing {os.path.basename(path)}")
        m, mm = load_mask(path, fallback_mm)
        out[tag] = view2d_from_mask(m, mm, voxel_size)
    return out


def read_reference_ef(csv_path: str) -> Dict[str, dict]:
    import csv
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pid = row.get('patient', '').strip()
            if not pid:
                continue
            if not pid.startswith('patient'):
                pid = f"patient{int(pid):04d}"

            def num(k):
                try:
                    return float(row[k])
                except (KeyError, TypeError, ValueError):
                    return float('nan')

            out[pid] = {'ef_ref': num('EF'), 'ef_2ch': num('EF_2CH'),
                        'ef_4ch': num('EF_4CH'),
                        'quality_2ch': row.get('ImageQuality_2CH', ''),
                        'quality_4ch': row.get('ImageQuality_4CH', '')}
    return out