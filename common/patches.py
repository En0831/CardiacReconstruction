# common/patches.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PatchConfig:
    size: Tuple[int, int] = (192, 192)
    mm_per_px: Optional[float] = 1.0
    apex_rc: Tuple[float, float] = (8.0, 96.0)
    normalize: bool = False
    target_lv_len_mm: float = 100.0      # if normalize=True


def _sampling_pitch(view, patch: PatchConfig) -> float:
    """mm per patch pixel for this view."""
    if patch.mm_per_px is not None:
        base = float(patch.mm_per_px)
    else:
        base = float(view.mm_per_px)
    if not patch.normalize:
        return base
    lv_mm = float(view.lv_len_px) * float(view.mm_per_px)
    if lv_mm <= 1e-6:
        return base
    return base * (lv_mm / patch.target_lv_len_mm)


def patch_coords(view, patch: PatchConfig) -> np.ndarray:
    """[rows, cols, 2] float64 of patch pixel coordinates in the source image"""
    rows, cols = patch.size
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    pitch = _sampling_pitch(view, patch)
    beta_mm = (r - patch.apex_rc[0]) * pitch          # apex -> base
    alpha_mm = (c - patch.apex_rc[1]) * pitch         # lateral

    ppx = float(view.mm_per_px)
    e_v = np.asarray(view.e_v_2d, np.float64)
    e_u = np.array([-e_v[1], e_v[0]], np.float64)
    return (np.asarray(view.apex2d, np.float64)[None, None, :]
            + (beta_mm / ppx)[..., None] * e_v[None, None, :]
            + (alpha_mm / ppx)[..., None] * e_u[None, None, :])


def view_patch(view, patch: PatchConfig = PatchConfig()) -> np.ndarray:
    """One View2D -> [rows, cols] uint8 labels in the detected 2D frame"""
    rows, cols = patch.size
    out = np.zeros((rows, cols), np.uint8)
    if getattr(view, 'is_empty', False):
        return out
    yx = patch_coords(view, patch)
    rr = np.round(yx[..., 0]).astype(np.int32)
    cc = np.round(yx[..., 1]).astype(np.int32)
    H, W = view.mask.shape
    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    out[ok] = view.mask[rr[ok], cc[ok]]
    return out


def view_patches(views: Dict[str, object], tags: Sequence[str],
                 patch: PatchConfig = PatchConfig()) -> np.ndarray:
    """-> [n_views, rows, cols] uint8, in `tags` order."""
    return np.stack([view_patch(views[t], patch) for t in tags]).astype(np.uint8)


def view_patch_with_mask(view, patch: PatchConfig = PatchConfig()
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """-> ([rows, cols] uint8, [rows, cols] bool)"""
    rows, cols = patch.size
    lab = np.zeros((rows, cols), np.uint8)
    ok = np.zeros((rows, cols), bool)
    if getattr(view, 'is_empty', False):
        return lab, ok
    yx = patch_coords(view, patch)
    rr = np.round(yx[..., 0]).astype(np.int32)
    cc = np.round(yx[..., 1]).astype(np.int32)
    H, W = view.mask.shape
    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    lab[ok] = view.mask[rr[ok], cc[ok]]
    return lab, ok


def view_patches_with_mask(views, tags, patch: PatchConfig = PatchConfig()):
    """-> ([n_views, rows, cols] uint8, [n_views, rows, cols] bool)"""
    pairs = [view_patch_with_mask(views[t], patch) for t in tags]
    return (np.stack([p[0] for p in pairs]).astype(np.uint8),
            np.stack([p[1] for p in pairs]))


def patch_escape(view, patch: PatchConfig = PatchConfig(),
                 n_classes: int = 6) -> dict:
    
    out: Dict[str, object] = {'empty': bool(getattr(view, 'is_empty', False))}
    if out['empty']:
        return out
    pts = np.stack(np.where(view.mask > 0)).T.astype(np.float64)
    if pts.size == 0:
        out['empty'] = True
        return out

    ppx = float(view.mm_per_px)
    e_v = np.asarray(view.e_v_2d, np.float64)
    e_u = np.array([-e_v[1], e_v[0]], np.float64)
    rel = (pts - np.asarray(view.apex2d, np.float64)) * ppx
    beta = rel @ e_v          # mm along apex -> base
    alpha = rel @ e_u         # mm lateral

    rows, cols = patch.size
    pitch = _sampling_pitch(view, patch)
    lo_b = -patch.apex_rc[0] * pitch
    hi_b = (rows - 1 - patch.apex_rc[0]) * pitch
    lo_a = -patch.apex_rc[1] * pitch
    hi_a = (cols - 1 - patch.apex_rc[1]) * pitch

    inside = (beta >= lo_b) & (beta <= hi_b) & (alpha >= lo_a) & (alpha <= hi_a)
    out.update({
        'escaped_frac': float(1.0 - inside.mean()),
        'beta_min_mm': float(beta.min()), 'beta_max_mm': float(beta.max()),
        'alpha_min_mm': float(alpha.min()), 'alpha_max_mm': float(alpha.max()),
        'margin_beta_lo': float(beta.min() - lo_b),
        'margin_beta_hi': float(hi_b - beta.max()),
        'margin_alpha_lo': float(alpha.min() - lo_a),
        'margin_alpha_hi': float(hi_a - alpha.max()),
        'pitch_mm': float(pitch),
    })
    for lab in range(1, n_classes):
        sel = view.mask[view.mask > 0] == lab
        if sel.any():
            out[f'escaped_{lab}'] = float(1.0 - inside[sel].mean())
    return out


# ===================
# landmark vectorization
# ===================
LANDMARK_FIELDS = ('is_empty', 'lv_len_mm', 'mm_per_px',
                   'area_1_mm2', 'area_2_mm2', 'area_3_mm2',
                   'area_4_mm2', 'area_5_mm2')
LANDMARK_DIM_VIEW = len(LANDMARK_FIELDS)


def landmark_vector(views: Dict[str, object], tags: Sequence[str],
                    n_classes: int = 6) -> np.ndarray:
    rows = []
    for t in tags:
        v = views[t]
        empty = bool(getattr(v, 'is_empty', False))
        ppx = float(getattr(v, 'mm_per_px', 1.0))
        row = [float(empty),
               0.0 if empty else float(v.lv_len_px) * ppx,
               ppx]
        for lab in range(1, n_classes):
            row.append(0.0 if empty else float((v.mask == lab).sum()) * ppx * ppx)
        rows.append(row)
    return np.asarray(rows, np.float32).reshape(-1)