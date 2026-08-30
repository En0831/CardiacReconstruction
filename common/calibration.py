# common/calibration.py


from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats


def normalized_residuals(Z: np.ndarray, MU: np.ndarray, SIGMA: np.ndarray) -> np.ndarray:
    return (np.asarray(Z, float) - np.asarray(MU, float)) / np.asarray(SIGMA, float)


def per_dim_report(Z: np.ndarray, MU: np.ndarray, SIGMA: np.ndarray,
                   names: Optional[Sequence[str]] = None) -> Dict[str, dict]:
    R = normalized_residuals(Z, MU, SIGMA)
    S = np.asarray(SIGMA, float)
    names = names or [f'z{i}' for i in range(R.shape[1])]
    out = {}
    for i, name in enumerate(names):
        r = R[:, i]
        nll = float(np.mean(0.5 * r ** 2 + np.log(S[:, i]) + 0.5 * np.log(2 * np.pi)))
        kur = float(stats.kurtosis(r))
        out[name] = {
            'mean': float(r.mean()),
            'std': float(r.std(ddof=1)),
            'skew': float(stats.skew(r)),
            'excess_kurtosis': kur,
            'cover68': float(np.mean(np.abs(r) <= 1.0)),
            'cover95': float(np.mean(np.abs(r) <= 1.959964)),
            'nll': nll,
            'flag_overconfident': bool(r.std(ddof=1) > 1.25),
            'flag_underconfident': bool(r.std(ddof=1) < 0.8),
            'flag_bimodal': bool(kur < -1.0),
        }
    return out


def residual_correlation(Z: np.ndarray, MU: np.ndarray, SIGMA: np.ndarray) -> np.ndarray:
    return np.corrcoef(normalized_residuals(Z, MU, SIGMA).T)


def correlated_pairs(C: np.ndarray, names: Sequence[str], thresh: float = 0.3):
    """Off-diagonal |corr| above thresh -- the full-covariance upgrade trigger."""
    out = []
    for i in range(C.shape[0]):
        for j in range(i + 1, C.shape[1]):
            if np.isfinite(C[i, j]) and abs(C[i, j]) >= thresh:
                out.append((names[i], names[j], float(C[i, j])))
    return sorted(out, key=lambda t: -abs(t[2]))

