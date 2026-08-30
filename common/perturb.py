# common/perturb.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .views import PlaneParams, ViewConfig, ViewGeometry, TAGS

# ==========================
# specs
# ==========================
@dataclass
class AugmentSpec:
    sigma_longaxis_deg: float = 0.0
    sigma_tilt_deg: float = 0.0
    sigma_inplane_deg: float = 0.0
    sigma_trans_n_mm: float = 0.0
    sigma_trans_long_mm: float = 0.0
    sigma_trans_lat_mm: float = 0.0
    per_view: bool = True

    def is_identity(self) -> bool:
        return all(
            getattr(self, f'sigma_{mode}') == 0.0
            for mode in ['longaxis_deg', 'tilt_deg', 'inplane_deg',
                         'trans_n_mm', 'trans_long_mm', 'trans_lat_mm']
        )

def arm_specs(sigma_rot_deg: float = 15.0, sigma_trans_mm: float = 8.0) -> Dict[str, Tuple[AugmentSpec, AugmentSpec]]:
    zero = AugmentSpec()
    six = AugmentSpec(
        sigma_longaxis_deg = sigma_rot_deg,
        sigma_tilt_deg = sigma_rot_deg,
        sigma_inplane_deg = sigma_rot_deg,
        sigma_trans_n_mm = sigma_trans_mm,
        sigma_trans_long_mm = sigma_trans_mm,
        sigma_trans_lat_mm = sigma_trans_mm,
    )
    return {'A0': (zero, zero), 'A1': (six, zero), 'A2': (six, six)}


# =========================
# SO(3)
# =========================
def rodrigues(omega: np.ndarray) -> np.ndarray:
    """Axis-angle vector -> rotation matrix"""
    omega = np.asarray(omega, np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))    # rotation angle
    if theta < 1e-8:
        return np.eye(3)
    k = omega / theta       # unit rotation axis
    k = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)

def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector (inverse of rodrigues)"""
    R = np.asarray(R, np.float64)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = np.arccos(c)
    if theta < 1e-8:                # if theta is near 0
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-8:   # if theta is near pi
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(k))
        k = A[:, i] / (k[i] + 1e-12)
        return theta * k / (np.linalg.norm(k) + 1e-12)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return (theta / (2 * np.sin(theta))) * v

def geodesic_deg(R: np.ndarray) -> float:
    """Rotation angle of R"""
    R = np.asarray(R, np.float64)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = np.arccos(c)
    return float(np.degrees(theta))


# ========================
# sampling a rigid perturbation
# ========================
def sample_rigid_gaussian(spec: AugmentSpec, frame: PlaneParams, config: ViewConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a rigid perturbation (R, t) from independent per-DoF Gaussian, for training-time augmentation"""
    omega = (np.deg2rad(spec.sigma_longaxis_deg) * rng.standard_normal() * frame.e_v +
             np.deg2rad(spec.sigma_tilt_deg) * rng.standard_normal() * frame.e_u +
             np.deg2rad(spec.sigma_inplane_deg) * rng.standard_normal() * frame.normal)
    R = rodrigues(omega)
    t_mm = (spec.sigma_trans_n_mm * rng.standard_normal() * frame.normal + 
            spec.sigma_trans_long_mm * rng.standard_normal() * frame.e_v +
            spec.sigma_trans_lat_mm * rng.standard_normal() * frame.e_u)
    t = t_mm / float(config.mm_per_voxel)
    return R, t

def apply_to_frame(frame:PlaneParams, R: np.ndarray, t: np.ndarray) -> PlaneParams:
    """Apply a rigid perturbation to a plane frame"""
    R = np.asarray(R, np.float64)
    t = np.asarray(t, np.float64)
    new_normal = (R @ frame.normal).astype(np.float32)
    new_e_u = (R @ frame.e_u).astype(np.float32)
    new_e_v = (R @ frame.e_v).astype(np.float32)
    new_origin = (frame.origin + t).astype(np.float32)
    return PlaneParams(origin=new_origin, normal=new_normal, e_u=new_e_u, e_v=new_e_v)


# =========================
# observation side perturbation
# =========================
def observation_perturbation(spec, config: ViewConfig, rng: Optional[np.random.Generator] = None):
    """Sample a rigid perturbation for the observation side, or return None if not meaningful"""
    rng = rng or np.random.default_rng()
    state = {'shared': None, 'seen': set()}

    def perturb(tag: str, plane: PlaneParams, geometry, _rng: np.random.Generator) -> PlaneParams:
        if spec.is_identity():
            return plane
        if tag in state['seen']:
            state['seen'].clear()
            state['shared'] = None
        state['seen'].add(tag)

        if spec.per_view: # per-view perturbation
            R, t = sample_rigid_gaussian(spec, plane, config, rng)
        else:
            if state['shared'] is None:     # shared perturbation across a2c and a4c
                state['shared'] = sample_rigid_gaussian(spec, plane, config, rng)
            R, t = state['shared']
        return apply_to_frame(plane, R, t)
 
    return perturb


# =========================
# placement side perturbation
# =========================
def placement_perturbation(config: ViewConfig, geometry: Optional[ViewGeometry], spec, rng: Optional[np.random.Generator] = None,
                           a4c_fixed: bool = True, base_frames: Optional[Dict[str, PlaneParams]] = None
                           ) -> Tuple[Dict[str, PlaneParams], Dict[str, dict]]:
    rng = rng or np.random.default_rng()
    if base_frames is None:
        raise ValueError("base_frames must be provided for placement perturbation")    
    frames = dict(base_frames)
    truth = {}
    for tag in TAGS:
        if spec.is_identity() or (a4c_fixed and tag == 'a4c'):
            R, t = np.eye(3), np.zeros(3)
        else:
            R, t = sample_rigid_gaussian(spec, frames[tag], config, rng)
        truth[tag] = {'R': R, 't': np.asarray(t, np.float64), 'frame_canonical': frames[tag]}
        frames[tag] = apply_to_frame(frames[tag], R, t)
    return frames, truth



# def pose_error(R_est: np.ndarray, t_est: np.ndarray, R_true: np.ndarray, t_true: np.ndarray, frame: PlaneParams, config: ViewConfig) -> dict:
#     """Compute pose error metrics between estimated and ground truth poses"""
#     R_res = np.asarray(R_est, np.float64) @ np.asarray(R_true, np.float64)
#     t_res = (np.asarray(t_est, np.float64) + np.asarray(t_true, np.float64)) * float(config.mm_per_voxel)

#     omega = log_so3(R_res)
#     e_u = frame.e_u.astype(np.float64)
#     e_v = frame.e_v.astype(np.float64)
#     n = frame.normal.astype(np.float64)

#     return {
#         'rot_deg': geodesic_deg(R_res),
#         'trans_mm': float(np.linalg.norm(t_res)),
#         'rot_longaxis_deg': float(np.degrees(omega @ e_v)),
#         'rot_tilt_deg': float(np.degrees(omega @ e_u)),
#         'rot_inplane_deg': float(np.degrees(omega @ n)),
#         'trans_long_mm': float(t_res @ e_v),
#         'trans_lat_mm': float(t_res @ e_u),
#         'trans_n_mm': float(t_res @ n),
#         'injected_rot_deg': geodesic_deg(R_true),
#         'injected_trans_mm': float(np.linalg.norm(t_true) * config.mm_per_voxel),
#     }

# def relative_pose_error(est: Dict[str, dict], truth: Dict[str, dict], config: ViewConfig) -> dict:
#     """Compute relative pose error between a4c and a2c"""
#     def total(tag):
#         R = np.asarray(est[tag]['R'], np.float64) @ np.asarray(truth[tag]['R'], np.float64)
#         t = (np.asarray(est[tag]['t'], np.float64) + np.asarray(truth[tag]['t'], np.float64))
#         return R, t

#     R4, t4 = total('a4c')
#     R2, t2 = total('a2c')
#     R_rel = R4.T @ R2
#     t_rel = R4.T @ (t2 - t4) * float(config.mm_per_voxel)
#     frame = truth['a2c']['frame_canonical']
#     omega = log_so3(R_rel)
#     return {
#         'rel_rot_deg': geodesic_deg(R_rel),
#         'rel_trans_mm': float(np.linalg.norm(t_rel)),
#         'rel_rot_longaxis_deg': float(np.degrees(omega @ frame.e_v.astype(np.float64))),
#         'rel_rot_tilt_deg': float(np.degrees(omega @ frame.e_u.astype(np.float64))),
#         'rel_rot_inplane_deg': float(np.degrees(omega @ frame.normal.astype(np.float64))),
#     }
