# common/metrics.py

from __future__ import annotations
 
from typing import Dict, List, Optional, Sequence, Tuple
 
import numpy as np
from scipy import ndimage, stats
from .views import STRUCT
 
 
# 6-class whole-heart convention
DEFAULT_STRUCTS: List[Tuple[str, int]] = list(STRUCT)
 
LEFT_HEART = ('LV', 'MY', 'LA')
FG_LABELS: Tuple[int, ...] = tuple(i for _, i in DEFAULT_STRUCTS)
STATUS_OK, STATUS_PRED_EMPTY, STATUS_GT_EMPTY, STATUS_BOTH_EMPTY = ('ok', 'pred_empty', 'gt_empty', 'both_empty')
 
# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def dice(a: np.ndarray, b: np.ndarray) -> float:
    """2 * |A ∩ B| / (|A| + |B|)"""
    sa, sb = float(a.sum()), float(b.sum())
    if sa + sb == 0:
        return float('nan')
    return 2.0 * float(np.logical_and(a, b).sum()) / (sa + sb)


def mean_dice(pred: np.ndarray, gt: np.ndarray, labels: Sequence[int] = FG_LABELS) -> float:
    return float(np.nanmean([dice(pred == k, gt == k) for k in labels]))


def majority_vote(samples: Sequence[np.ndarray], n_classes: int = 6) -> np.ndarray:
    counts = np.zeros((n_classes,) + samples[0].shape, np.uint16)
    for s in samples:
        for k in range(n_classes):
            counts[k] += (s == k)
    return counts.argmax(0).astype(np.uint8)


def _surface(mask: np.ndarray, border_is_background: bool = True) -> np.ndarray:
    """Binary mask -> binary surface mask (1 voxel thick)."""
    if ndimage is None:
        raise ImportError("scipy is required for surface metrics")
    eroded = ndimage.binary_erosion(mask, ndimage.generate_binary_structure(mask.ndim, 1),
                                    border_value=0 if border_is_background else 1)
    return np.logical_and(mask, np.logical_not(eroded))
 
 
def surface_distances(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    sp = tuple(float(s) for s in spacing)
    sp_pred, sp_gt = _surface(pred), _surface(gt)
    if sp_pred.sum() == 0 or sp_gt.sum() == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty
    dt_to_gt = ndimage.distance_transform_edt(np.logical_not(sp_gt), sampling=sp)
    dt_to_pred = ndimage.distance_transform_edt(np.logical_not(sp_pred), sampling=sp)
    return dt_to_gt[sp_pred], dt_to_pred[sp_gt]
 
 
def assd_hd(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], percentile: float = 95.0) -> Tuple[float, float]:
    """(ASSD, HD_percentile) in mm"""
    d_pg, d_gp = surface_distances(pred, gt, spacing)
    if d_pg.size == 0 or d_gp.size == 0:
        return float('nan'), float('nan')
    assd = float((d_pg.sum() + d_gp.sum()) / (d_pg.size + d_gp.size))
    hd = float(max(np.percentile(d_pg, percentile), np.percentile(d_gp, percentile)))
    return assd, hd
 
 
def volume_ml(mask: np.ndarray, spacing: Sequence[float]) -> float:
    """Voxel count -> mL"""
    vox_mm3 = float(np.prod([float(s) for s in spacing]))
    return float(mask.sum()) * vox_mm3 / 1000.0


def volumes_ml_by_struct(vol: np.ndarray, vox_ml: float,
                         structs: Sequence[Tuple[str, int]] = tuple(DEFAULT_STRUCTS)) -> Dict[str, float]:
    """label volume -> {'vol_LV_ml': ..., ...}"""
    return {f'vol_{name}_ml': float((vol == lab).sum()) * vox_ml for name, lab in structs}
 
 
# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------
class MulticlassEvaluator:
    """Accumulates per-structure metrics over a set of cases."""
 
    def __init__(self,
                 spacing: Sequence[float] = (2.0, 2.0, 2.0),
                 structs: Sequence[Tuple[str, int]] = tuple(DEFAULT_STRUCTS),
                 hd_percentile: float = 95.0,
                 surface: bool = True):
        self.spacing = tuple(float(s) for s in spacing)
        self.structs = list(structs)
        self.hd_percentile = float(hd_percentile)
        self.hd_key = f"hd{int(round(hd_percentile))}"
        self.surface = surface and (ndimage is not None)
        if surface and ndimage is None:
            print("  (scipy missing: ASSD/HD disabled, Dice and volumes still reported)")
        self.cases: List[dict] = []
 
    # -- one case ----------------------------------------------------------
    def add(self, pred: np.ndarray, gt: np.ndarray, case_id: Optional[str] = None, extra: Optional[dict] = None) -> dict:
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
 
        res: Dict[str, dict] = {}
        for name, lab in self.structs:
            p, g = (pred == lab), (gt == lab)
            has_p, has_g = bool(p.any()), bool(g.any())
            if has_p and has_g:
                status = STATUS_OK
            elif has_g:
                status = STATUS_PRED_EMPTY
            elif has_p:
                status = STATUS_GT_EMPTY
            else:
                status = STATUS_BOTH_EMPTY
 
            # gt_empty / both_empty -> the case cannot score this structure at all
            d = float('nan') if not has_g else (dice(p, g) if has_p else 0.0)
            a = h = float('nan')
            if self.surface and has_p and has_g:
                a, h = assd_hd(p, g, self.spacing, self.hd_percentile)
 
            vp = volume_ml(p, self.spacing)
            vg = volume_ml(g, self.spacing)
            res[name] = {'dice': d, 'assd': a, self.hd_key: h, 'status': status, 'vol_pred': vp, 'vol_gt': vg, 'vol_err': vp - vg}
 
        entry = {'case': case_id, 'structures': res}
        entry['mean_dice'] = _nanmean([res[n]['dice'] for n, _ in self.structs])
        entry['mean_dice_left'] = _nanmean(
            [res[n]['dice'] for n, _ in self.structs if n in LEFT_HEART])
        if extra:
            entry.update(extra)
        self.cases.append(entry)
        return res
 
    # -- aggregate ---------------------------------------------------------
    def per_struct(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for name, _ in self.structs:
            vals = [c['structures'][name] for c in self.cases]
            d = np.array([v['dice'] for v in vals], float)
            a = np.array([v['assd'] for v in vals], float)
            h = np.array([v[self.hd_key] for v in vals], float)
            ve = np.array([v['vol_err'] for v in vals], float)
            statuses = [v['status'] for v in vals]
            out[name] = {
                'dice': _nanmean(d), 'dice_std': _nanstd(d),
                'assd': _nanmean(a), 'assd_std': _nanstd(a),
                self.hd_key: _nanmean(h),
                'vol_err': _nanmean(ve), 'vol_err_std': _nanstd(ve),
                'n_dice': int(np.isfinite(d).sum()),
                'n_surf': int(np.isfinite(a).sum()),
                'n_pred_empty': int(sum(s == STATUS_PRED_EMPTY for s in statuses)),
                'n_gt_empty': int(sum(s in (STATUS_GT_EMPTY, STATUS_BOTH_EMPTY)
                                      for s in statuses)),
            }
        return out
 
    def summary(self, verbose: bool = True) -> Dict[str, dict]:
        t = self.per_struct()
        names = [n for n, _ in self.structs]
        left = [n for n in names if n in LEFT_HEART]
 
        t['_all'] = {
            'dice': _nanmean([t[n]['dice'] for n in names]),
            'assd': _nanmean([t[n]['assd'] for n in names]),
            self.hd_key: _nanmean([t[n][self.hd_key] for n in names]),
            'dice_case_mean': _nanmean([c['mean_dice'] for c in self.cases]),
            'dice_case_std': _nanstd(np.array([c['mean_dice'] for c in self.cases], float)),
            'n_cases': len(self.cases),
        }
        t['_left'] = {
            'dice': _nanmean([t[n]['dice'] for n in left]),
            'assd': _nanmean([t[n]['assd'] for n in left]),
            self.hd_key: _nanmean([t[n][self.hd_key] for n in left]),
            'dice_case_mean': _nanmean([c['mean_dice_left'] for c in self.cases]),
            'dice_case_std': _nanstd(np.array([c['mean_dice_left'] for c in self.cases],
                                              float)),
            'n_cases': len(self.cases),
        }
        if verbose:
            print(self.format_table(t))
        return t
 
    def format_table(self, t: Optional[Dict[str, dict]] = None) -> str:
        t = t or self.per_struct()
        hk = self.hd_key
        w = 76
        lines = ["=" * w,
                 f"{'struct':<8}{'Dice':>8}{'±sd':>7}{'ASSD':>8}{hk:>8}"
                 f"{'dVol':>9}{'n':>5}{'nSurf':>7}{'nPE':>5}",
                 "-" * w]
        for name, _ in self.structs:
            r = t[name]
            lines.append(
                f"{name:<8}{_f(r['dice'],3):>8}{_f(r['dice_std'],3):>7}"
                f"{_f(r['assd'],2):>8}{_f(r[hk],2):>8}{_f(r['vol_err'],1):>9}"
                f"{r['n_dice']:>5}{r['n_surf']:>7}{r['n_pred_empty']:>5}")
        lines.append("-" * w)
        for key, label in (('_all', 'ALL'), ('_left', 'LEFT')):
            if key in t:
                r = t[key]
                lines.append(f"{label:<8}{_f(r['dice'],3):>8}{'':>7}"
                             f"{_f(r['assd'],2):>8}{_f(r[hk],2):>8}")
        lines.append("=" * w)
        lines.append(f"spacing={self.spacing} mm | ASSD/{hk} in mm | dVol in mL "
                     f"(pred-gt) | nPE = pred_empty")
        return "\n".join(lines)

 
 
# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _nanmean(x) -> float:
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')
 
 
def _nanstd(x) -> float:
    x = np.asarray(x, float)
    return float(np.nanstd(x, ddof=1)) if np.isfinite(x).sum() > 1 else float('nan')
 
 
def _f(v, nd) -> str:
    return '-' if v is None or not np.isfinite(v) else f"{v:.{nd}f}"
