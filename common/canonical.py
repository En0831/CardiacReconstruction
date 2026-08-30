# common/canonical.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from common.views import LV, MY, TAGS, EmptyObservationError, PlaneParams, View2D, ViewConfig, ViewGeometry, build_geometry, canonical_frames, full_grid, original_frames, place
from common.perturb import apply_to_frame, geodesic_deg, log_so3, rodrigues

# 6 DoF for each view, plus one shared log-scale DoF
Z_VIEW_DOF = ('rot_longaxis_deg', 'rot_tilt_deg', 'rot_inplane_deg', 'trans_long_mm', 'trans_lat_mm', 'trans_n_mm')
Z_DIM_VIEW = len(Z_VIEW_DOF)
Z_DIM = 2 * Z_DIM_VIEW                      # a4c(6) + a2c(6) = 12
Z_NAMES = tuple(f'{tag}_{d}' for tag in TAGS for d in Z_VIEW_DOF)

GAUGE_ANCHORS = ('anatomical', 'a4c')

class DegenerateResidualError(EmptyObservationError):
    """Raised when the residual between canonical and true frames is too large"""

_CANON_CACHE: dict = {}
def get_canonical_frames(config: ViewConfig):
    """Get cached canonical frames for the given view configuration."""
    key = (tuple(config.canonical_apex), config.canonical_a2c_angle_deg)
    if key not in _CANON_CACHE:
        _CANON_CACHE[key] = canonical_frames(config)
    return _CANON_CACHE[key]


def _project_so3(R, tol_deg: float = 0.1):
    """Project a 3x3 matrix onto SO(3) by SVD"""
    R = np.asarray(R, np.float64)
    orthogonal = float(np.abs(R @ R.T - np.eye(3)).max())
    if not np.isfinite(orthogonal) or orthogonal > np.deg2rad(tol_deg):
        raise ValueError(f"_project_so3: input is not near-orthogonal (max|R R^T - I| = {orthogonal:.3g})")
    U, _, Vt = np.linalg.svd(R)
    if np.linalg.det(U @ Vt) < 0:
        U = U.copy()
        U[:, -1] *= -1.0            # flip the last column to ensure det=+1 (= rotation)
    Rp = U @ Vt                     # Closest SO(3) matrix in Frobenius norm 
    return Rp


# ===============================
# gauge
# ===============================
@dataclass
class CanonicalGauge:
    """x_canonical = s * R @ (x - apex) + a0 (voxel units)"""
    R: np.ndarray           # [3,3], maps anatomical A4C basis to canonical basis
    apex: np.ndarray        # [3], anatomical apex in the volume frame
    a0: np.ndarray          # [3], canonical apex
    s: float = 1.0          # scale (1.0 when no normalisation requested)
    lv_len_mm: float = float('nan') 

    def point(self, x: np.ndarray) -> np.ndarray:
        """Vectorised forward for resampling: [..., 3] volume -> canonical."""
        x = np.asarray(x, np.float64)
        return (self.s * (x - self.apex) @ self.R.T + self.a0).astype(np.float32)

    def inverse_points(self, y: np.ndarray) -> np.ndarray:
        """Vectorised inverse for resampling: [..., 3] canonical -> volume."""
        y = np.asarray(y, np.float64)
        return ((y - self.a0) @ self.R / self.s + self.apex).astype(np.float32)

    def frame(self, frame: PlaneParams) -> PlaneParams:
        """Canonical frame corresponding to a volume frame"""
        R = self.R.astype(np.float64)
        return PlaneParams(normal=(R @ frame.normal).astype(np.float32),
                           origin=self.point(frame.origin),
                           e_u=(R @ frame.e_u).astype(np.float32),
                           e_v=(R @ frame.e_v).astype(np.float32))


def lv_extent_vox(seg_3d: np.ndarray, long_axis: np.ndarray) -> float:
    """Compute the LV+MY extent along the long axis in voxels. Returns NaN if empty."""
    pts = np.stack(np.where((seg_3d == LV) | (seg_3d == MY))).T.astype(np.float64)
    if pts.shape[0] == 0:
        return float('nan')
    proj = pts @ np.asarray(long_axis, np.float64)
    return float(proj.max() - proj.min())


def build_gauge(seg_3d: np.ndarray, config: ViewConfig, anchor: str = 'anatomical',
                target_lv_len_mm: Optional[float] = None, frame_a4c=None) -> CanonicalGauge:
    """Compute the canonical gauge for each case"""
    if config.canonical_a2c_angle_deg is None:
        raise ValueError("canonical pipeline requires config.canonical_a2c_angle_deg")
    G4 = get_canonical_frames(config)['a4c']
    if anchor == 'anatomical':
        if frame_a4c is not None:
            raise ValueError("a4c frame is for anchor='a4c', not 'anatomical'")
        geometry = build_geometry(seg_3d, config)
        F4 = geometry['a4c']
        long_axis = geometry.long_axis
    elif anchor == 'a4c':
        if frame_a4c is None:
            raise ValueError("anchor='a4c' requires frame_a4c to be specified")
        F4 = frame_a4c
        long_axis = np.asarray(F4.e_v, np.float64)
    else:
        raise ValueError(f"Unknown anchor: {anchor}")

    R = _project_so3(G4.basis() @ F4.basis().T)
    lv_len_mm = lv_extent_vox(seg_3d, long_axis) * config.mm_per_voxel

    return CanonicalGauge(R=R, apex=F4.origin.astype(np.float64).copy(),
                          a0=np.asarray(config.canonical_apex, np.float64),
                          s=1.0, lv_len_mm=float(lv_len_mm))


def resample_to_canonical(seg_3d: np.ndarray, gauge: CanonicalGauge, config: ViewConfig) -> np.ndarray:
    """Resample a 3D volume to the canonical grid using the gauge (inverse mapping to fill in all canonical voxels)"""
    grid = full_grid(config.grid_size)      # [X,Y,Z,3] canonical ids
    src = gauge.inverse_points(grid)        # [X,Y,Z,3] volume ids of canonical grid
    idx = np.round(src).astype(np.int32)    # [X,Y,Z,3] nearest-neighbour volume indices
    shape = np.array(seg_3d.shape)
    inside = np.all((idx >= 0) & (idx < shape), axis=-1)
    out = np.zeros(config.grid_size, np.uint8)
    ii = idx[inside]
    out[inside] = seg_3d[ii[:, 0], ii[:, 1], ii[:, 2]]
    return out


def to_original(vol_c: np.ndarray, gauge: CanonicalGauge, out_shape: Sequence[int]) -> np.ndarray:
    """Canonical-frame labels -> original frame (inverse gauge, NN)."""
    grid = full_grid(tuple(out_shape))
    y = (gauge.s * (grid - gauge.apex) @ gauge.R.T + gauge.a0)
    idx = np.round(y).astype(np.int32)
    inside = np.all((idx >= 0) & (idx < np.array(vol_c.shape)), axis=-1)
    out = np.zeros(tuple(out_shape), np.uint8)
    ii = idx[inside]
    out[inside] = vol_c[ii[:, 0], ii[:, 1], ii[:, 2]]
    return out

# ================================================================================
# z: residual parameterisation
# ================================================================================
@dataclass
class ZResidual:
    """
    Continuous residual between canonical frames and true frames
    """
    R: Dict[str, np.ndarray]
    t: Dict[str, np.ndarray]
    valid: Dict[str, bool]

    def dof(self, config: ViewConfig) -> Dict[str, Dict[str, float]]:
        G = get_canonical_frames(config)
        out = {}
        for tag in TAGS:
            g = G[tag]
            omega = log_so3(self.R[tag])    # axis-angle vector (radians)
            t_mm = np.asarray(self.t[tag], np.float64) * config.mm_per_voxel
            out[tag] = {
                'rot_longaxis_deg': float(np.degrees(omega @ g.e_v.astype(np.float64))),
                'rot_tilt_deg': float(np.degrees(omega @ g.e_u.astype(np.float64))),
                'rot_inplane_deg': float(np.degrees(omega @ g.normal.astype(np.float64))),
                'trans_long_mm': float(t_mm @ g.e_v.astype(np.float64)),
                'trans_lat_mm': float(t_mm @ g.e_u.astype(np.float64)),
                'trans_n_mm': float(t_mm @ g.normal.astype(np.float64)),
            }
        return out


def z_from_frames(frames_true: Dict[str, PlaneParams], config: ViewConfig, valid: Optional[Dict[str, bool]] = None) -> ZResidual:
    """Residual carrying the fixed canonical frames onto the true (gauged) frames"""
    G = get_canonical_frames(config)
    R, t, ok = {}, {}, {}
    for tag in TAGS:
        v = True if valid is None else bool(valid.get(tag, True))
        ok[tag] = v
        if not v:
            R[tag], t[tag] = np.eye(3), np.zeros(3)
            continue
        F = frames_true[tag]
        R[tag] = _project_so3(F.basis().astype(np.float64) @ G[tag].basis().astype(np.float64).T)
        t[tag] = (F.origin.astype(np.float64) - G[tag].origin.astype(np.float64))
        ang = geodesic_deg(R[tag])  # angle between canonical and true frames
        if ang >= 120.0:    # if the residual is too large
            raise DegenerateResidualError(f"z_from_frames: view {tag} has degenerate residual (geodesic {ang:.1f} deg) -- check canonical_a2c_angle_deg or anatomy", F)
    return ZResidual(R=R, t=t, valid=ok)


def frames_from_z(z: ZResidual, config: ViewConfig) -> Dict[str, PlaneParams]:
    """Compute the true frames from the canonical frames and the residual z (inverse of z_from_frames)"""
    G = get_canonical_frames(config)
    return {tag: (apply_to_frame(G[tag], z.R[tag], z.t[tag]) if z.valid[tag] else G[tag]) for tag in TAGS}


def place_at_z(views: Dict[str, View2D], z: ZResidual, config: ViewConfig):
    """Place the views at the true frames corresponding to z"""
    return place(views, config, frames=frames_from_z(z, config))


# ----- flat vector form for the regression net ----------------------------------
def z_pack(z: ZResidual, config: ViewConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Z residual (R, t) -> (vec [12] in Z_NAMES order, valid [2] per view)."""
    dof = z.dof(config)
    vec = np.array([dof[tag][d] for tag in TAGS for d in Z_VIEW_DOF], np.float64)
    return vec, np.array([z.valid[t] for t in TAGS], bool)


def z_unpack(vec: np.ndarray, config: ViewConfig,
             valid: Optional[np.ndarray] = None) -> ZResidual:
    """Inverse of z_pack: (vec [12], valid [2]) -> ZResidual (R, t)"""
    vec = np.asarray(vec, np.float64).reshape(Z_DIM)
    G = get_canonical_frames(config)
    R, t, ok = {}, {}, {}
    for k, tag in enumerate(TAGS):
        v = vec[k * Z_DIM_VIEW:(k + 1) * Z_DIM_VIEW]
        g = G[tag]
        axes = (g.e_v.astype(np.float64), g.e_u.astype(np.float64),
                g.normal.astype(np.float64))
        omega = sum(np.deg2rad(v[i]) * axes[i] for i in range(3))
        R[tag] = rodrigues(omega)
        t[tag] = sum(v[3 + i] * axes[i] for i in range(3)) / config.mm_per_voxel
        ok[tag] = True if valid is None else bool(valid[k])
    return ZResidual(R=R, t=t, valid=ok)


# ===============================
# one-stop truth extraction for the dataset / oracle evaluation
# ===============================
@dataclass
class CanonicalTruth:
    gauge: CanonicalGauge
    frames_true: Dict[str, PlaneParams]     # gauged original_frames
    z: ZResidual                            # z* (target z)
    target: Optional[np.ndarray] = None     # canonical GT (only if requested)


def compute_truth(seg_3d: np.ndarray, views: Dict[str, View2D], geometry_obs: ViewGeometry, 
                  config: ViewConfig,with_target: bool = True, gauge: Optional[CanonicalGauge] = None,
                  target: Optional[np.ndarray] = None, anchor: str = 'anatomical') -> CanonicalTruth:
    """Compute the canonical truth for a given 3D segmentation and 2D views."""
    if gauge is None:
        frame_a4c = (original_frames(views, geometry_obs)['a4c'] if anchor == 'a4c' else None)
        gauge = build_gauge(seg_3d, config, anchor=anchor, frame_a4c=frame_a4c)
    T = original_frames(views, geometry_obs)                    # original frames in volume space (apex3d, detected_basis)
    frames_true = {tag: gauge.frame(T[tag]) for tag in TAGS}    # original frames pushed through the gauge to canonical space
    valid = {tag: (not views[tag].is_empty and views[tag].apex3d is not None) for tag in TAGS}
    z = z_from_frames(frames_true, config, valid=valid)
    if target is None and with_target:
        target = resample_to_canonical(seg_3d, gauge, config)
    return CanonicalTruth(gauge=gauge, frames_true=frames_true, z=z, target=target)