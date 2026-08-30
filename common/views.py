# common/views.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import logging

import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)

#----- structure labeling -----
STRUCT = (('LV', 1), ('MY', 2), ('RV', 3), ('LA', 4), ('RA', 5))
LV, MY, RV, LA, RA = (i for _, i in STRUCT)
N_CLASSES = len(STRUCT) + 1  # include background

VIEW_CLASSES = {
    'left': {'a4c': {LV, MY, LA}, 'a2c': {LV, MY, LA}},
    'rv': {'a4c': {LV, MY, RV, LA}, 'a2c': {LV, MY, LA}},
    'whole': {'a4c': {LV, MY, RV, LA, RA}, 'a2c': {LV, MY, LA}},
}

TAGS = ('a4c', 'a2c')


# ====================================
# config
# ====================================

@dataclass
class ViewConfig:
    grid_size: Tuple[int, int, int] = (96, 96, 128)
    mm_per_voxel: float = 2.0       # mm per grid voxel
    thickness_vox: float = 1.0      # voxels per slice
    pitch_mm: float = 1.0           # mm per pixel
    view_config: str = 'left'

    canonical_apex: Tuple[float, float, float] = (35.0, 44.0, 18.0)
    canonical_a2c_angle_deg : Optional[float] = None

    apex_sigma_vox: float = 0.0

    def spacing(self) -> np.ndarray:
        return np.full(3, self.mm_per_voxel, dtype=np.float32)

@dataclass
class PlaneParams:
    normal: np.ndarray      # [3] unit
    origin: np.ndarray      # [3] voxel unit
    e_u: np.ndarray         # [3] unit (short-axis)
    e_v: np.ndarray         # [3] unit (long-axis (apex->base))

    def basis(self) -> np.ndarray:
        """[3, 3] with columns (e_u, e_v, normal)"""
        return np.stack([self.e_u, self.e_v, self.normal], axis=1)

@dataclass
class ViewGeometry:
    long_axis: np.ndarray   # [3] unit, apex->base direction
    pca_axis: np.ndarray    # only used to detect apex in 3D (long-axis of LV and MY)
    apex: np.ndarray
    planes: Dict[str, PlaneParams]
    inter_plane_angle_deg : float
    signed_a2c_angle_deg: float =  float('nan')
    
    def __getitem__(self, tag: str) -> PlaneParams:
        return self.planes[tag]

@dataclass
class View2D:
    mask: np.ndarray        # [H, W] uint8, binary mask of the view (6 classes)
    alpha: np.ndarray       # [H, W] float32, lateral offset [voxels] 
    beta: np.ndarray        # [H, W] float32, longitudinal offset [voxels]
    apex2d: np.ndarray      # [2] float32, detected apex in 2D view coordinates
    e_v_2d: np.ndarray      # [2] float32, detected unit vector of long-axis in 2D view coordinates
    mm_per_px: float
    lv_len_px: float
    observed_classes: frozenset     # feeds the per-view observed map
    valid: Optional[np.ndarray] = None  # [H, W] bool, if the 2d view point is in the 3d grid
    apex3d: Optional[np.ndarray] = None
    is_empty: bool = False

    @staticmethod
    def empty(mask_shape, observed_classes) -> 'View2D':
        z2 = np.zeros(mask_shape, np.float32)
        return View2D(mask=np.zeros(mask_shape, np.uint8), alpha=z2, beta=z2.copy(), apex2d=np.zeros(2, np.float32), 
                      e_v_2d=np.array([0., 1.], np.float32), mm_per_px=1.0, lv_len_px=0.0, observed_classes=frozenset(), 
                      valid=np.zeros(mask_shape, bool), is_empty=True)

def px_per_voxel(mm_per_voxel: float, mm_per_px: float) -> float:
    mm = float(mm_per_px)
    if not np.isfinite(mm) or mm <= 0.0:
        raise ValueError(f"invalid mm_per_px: {mm_per_px}")
    return float(mm_per_voxel) / mm

class DegenerateGeometryError(ValueError):
    def __init__(self, message: str, case_id: Optional[str] = None):
        self.message = message
        self.case_id = case_id
        super().__init__(f"[{case_id or 'unknown case'}]{message}")

class EmptyObservationError(ValueError):
    def __init__(self, message: str, case_id: Optional[str] = None):
        self.message = message
        self.case_id = case_id
        super().__init__(f"[{case_id or 'unknown case'}]{message}")
 
 
# Minimum LV+MY pixels needed to detect a long axis in 2D (PCA + apex argmin).
MIN_FRAME_PTS = 10

# ====================================
# geometry
# ====================================
def _centroid(mask: np.ndarray) -> np.ndarray:
    """Compute the centroid of a binary mask."""
    coords = np.stack(np.where(mask))
    if coords.shape[1] == 0:
        return None
    return coords.mean(-1).astype(np.float32)

def _principal_axis(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the principal axis of a set of coordinates."""
    mean = coords.mean(0)
    _, eigvecs = np.linalg.eigh(np.cov((coords - mean).T))
    axis = eigvecs[:, -1]
    return axis / (np.linalg.norm(axis) + 1e-8), mean

def _perp_to_axis(v: np.ndarray, axis: np.ndarray) -> Optional[np.ndarray]:
    """Project vector v onto the plane perpendicular to axis."""
    v = v - (v @ axis) * axis
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None

def rotate_about_axis(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotate vector v about a given axis by angle theta (in radians)."""
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return (v * np.cos(theta)
            + np.cross(axis, v) * np.sin(theta)
            + axis * (axis @ v) * (1.0 - np.cos(theta)))

def build_geometry(seg_3d: np.ndarray, config: ViewConfig,
                   rng: Optional[np.random.Generator] = None,
                   obs_perturb=None, case_id: Optional[str] = None) -> ViewGeometry:
    """
    A4C and A2C view geometry from 3D segmentation.
    A4C: the plane through {LA centroid, RA centroid, apex}.
    A2C: the plane which has normal vector running towards the RV centroid direction with the long axis component removed.
    """
    if rng is None:
        rng = np.random.default_rng()
    L, W, H = seg_3d.shape

    seg_lv, seg_my = (seg_3d == LV), (seg_3d == MY)
    com_rv, com_la, com_ra = (_centroid(seg_3d == RV), _centroid(seg_3d == LA), _centroid(seg_3d == RA))

    if com_la is None:
        com_la = np.array([L/2, W/2, H/2], dtype=np.float32)

    # ----- Apex -----
    coords = np.stack(np.where(seg_lv | seg_my)).T.astype(np.float32)
    if coords.shape[0] < MIN_FRAME_PTS:
        raise _degenerate(f"only {coords.shape[0]} voxels in LV+MY mask", case_id=case_id)
    pca_axis, mean = _principal_axis(coords)
    if (com_la - mean) @ pca_axis < 0:
        pca_axis = -pca_axis
    proj = (coords - mean) @ pca_axis
    apex = coords[np.argmin(proj)].copy()
    if config.apex_sigma_vox > 0.0:
        apex += config.apex_sigma_vox * rng.normal(size=3).astype(np.float32)

    # ----- Long axis -----
    lv = com_la - apex
    n_lv = np.linalg.norm(lv)
    if n_lv < 1e-6:
        raise _degenerate(f"degenerate long axis (apex {apex} and LA centroid {com_la} are too close)", case_id=case_id)
    long_axis = (lv / n_lv).astype(np.float32)

    # ----- A4C plane -----
    n_a4c, o_a4c = None, apex.copy()
    if com_ra is not None:
        n = np.cross(com_ra - com_la, apex - com_la)
        nn = np.linalg.norm(n)
        if nn > 1e-6:
            n_a4c = (n / nn).astype(np.float32)
            o_a4c = ((com_la + com_ra + apex) / 3.0).astype(np.float32)
    if n_a4c is None:
        ref = _perp_to_axis(np.array([0., 1., 0.], np.float32), long_axis)
        if ref is None:
            ref = _perp_to_axis(np.array([1., 0., 0.], np.float32), long_axis)
        n_a4c = np.cross(long_axis, ref).astype(np.float32)
        n_a4c /= (np.linalg.norm(n_a4c) + 1e-8)

    # ---- A2C plane -----
    n_a2c, o_a2c = None, apex.copy()
    if com_rv is not None:
        foot = apex + (com_rv - apex) @ long_axis * long_axis
        v = com_rv - foot
        nn = np.linalg.norm(v)
        if nn > 1e-6:
            n_a2c = (v / nn).astype(np.float32)
            o_a2c = foot.astype(np.float32)
    if n_a2c is None:
        lat = _perp_to_axis(n_a4c, long_axis)
        if lat is None:
            lat = np.array([1., 0., 0.], np.float32)
        n_a2c = rotate_about_axis(np.cross(long_axis, lat), long_axis, np.deg2rad(60))
        n_a2c /= (np.linalg.norm(n_a2c) + 1e-8).astype(np.float32)

    # ----- plane params -----
    planes = {}
    for tag, n, o in [('a4c', n_a4c, o_a4c), ('a2c', n_a2c, o_a2c)]:
        planes[tag] = _plane_from_normal(n, o, long_axis, apex, tag, case_id)

    geometry = _finalize(long_axis, pca_axis, apex, planes)
    if obs_perturb is not None:
        geometry = _finalize(long_axis, pca_axis, apex, {tag: obs_perturb(tag, p, geometry, rng) for tag, p in planes.items()})

    return geometry

def _finalize(long_axis, pca_axis, apex, planes) -> ViewGeometry:
    geometry = ViewGeometry(
        long_axis=long_axis.astype(np.float32),
        pca_axis=pca_axis.astype(np.float32),
        apex=apex.astype(np.float32),
        planes=planes,
        inter_plane_angle_deg=_angle_between_planes(planes['a4c'].normal, planes['a2c'].normal))
    geometry.signed_a2c_angle_deg = signed_a2c_angle(geometry)
    return geometry

def _plane_from_normal(normal: np.ndarray, origin: np.ndarray, long_axis: np.ndarray, 
                       apex: np.ndarray, tag: str, case_id: Optional[str] = None) -> PlaneParams:
    """Construct a plane from a normal vector and an origin point."""
    n = normal / (np.linalg.norm(normal) + 1e-8)
    e_v = _perp_to_axis(long_axis, n)
    if e_v is None:
        raise _degenerate(f"degenerate plane {tag} (long axis is parallel to normal)", case_id=case_id)
    e_u = np.cross(e_v, n)
    e_u /= (np.linalg.norm(e_u) + 1e-8)
    anchor = apex - ((apex - origin) @ n) * n
    return PlaneParams(
        normal=n.astype(np.float32),
        origin=anchor.astype(np.float32),
        e_u=e_u.astype(np.float32),
        e_v=e_v.astype(np.float32))

def _angle_between_planes(n1: np.ndarray, n2: np.ndarray) -> float:
    """Compute the angle between two planes given their normal vectors."""
    cos_theta = float(n1 @ n2) / (float(np.linalg.norm(n1)) * float(np.linalg.norm(n2)) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(abs(cos_theta), 0.0, 1.0))))

def signed_a2c_angle(geometry: ViewGeometry) -> float:
    ax = geometry.long_axis / (np.linalg.norm(geometry.long_axis) + 1e-8)
    u4 = _perp_to_axis(geometry.planes['a4c'].e_u, ax)
    u2 = _perp_to_axis(geometry.planes['a2c'].e_u, ax)
    if u4 is None or u2 is None:
        return float('nan')
    s = float(np.cross(u4, u2) @ ax)
    c = float(u4 @ u2)
    return float(np.degrees(np.arctan2(s, c)))

def _degenerate(message:str, case_id: Optional[str] = None) -> DegenerateGeometryError:
    log.warning("degenerate geometry [%s]: %s", case_id, message)
    return DegenerateGeometryError(message, case_id=case_id)


# ====================================
# cut planes
# ====================================

def _image_extent(plane: PlaneParams, grid_size: Tuple[int, int, int], pitch_vox: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the extent of the image in 3D space for a given plane."""
    L, W, H = grid_size
    corners = np.array([[x, y, z] for x in (0, L - 1) for y in (0, W - 1) for z in (0, H - 1)], dtype=np.float32)
    rel = corners - plane.origin
    row_coords = rel @ row_direction(plane)
    col_coords = rel @ plane.e_v
    a_ax = np.arange(np.floor(row_coords.min() / pitch_vox), np.ceil(row_coords.max() / pitch_vox) + 1, dtype=np.float32) * pitch_vox
    b_ax = np.arange(np.floor(col_coords.min() / pitch_vox), np.ceil(col_coords.max() / pitch_vox) + 1, dtype=np.float32) * pitch_vox
    return a_ax, b_ax

def row_direction(plane: PlaneParams) -> np.ndarray:
    """Compute the row direction (e_u) of the plane."""
    return -plane.e_u

def cut_planes(seg_3d: np.ndarray, geometry: ViewGeometry, config: ViewConfig) -> Dict[str, dict]:
    keep = VIEW_CLASSES[config.view_config]
    pitch_vox = config.pitch_mm / config.mm_per_voxel
    shape = np.array(seg_3d.shape)
    out = {}
    for tag in TAGS:
        plane = geometry[tag]
        a_ax, b_ax = _image_extent(plane, seg_3d.shape, pitch_vox)
        A, B = np.meshgrid(a_ax, b_ax, indexing='ij')
        e_row = row_direction(plane)
        x = (plane.origin[None, None, :] + A[..., None] * e_row[None, None] + B[..., None] * plane.e_v[None, None])     # [L, W, 3]
        idx = np.round(x).astype(np.int32)
        inside = np.all((idx >= 0) & (idx < shape), axis=-1)
        lab = np.zeros(A.shape, np.uint8)
        ii = idx[inside]
        v = seg_3d[ii[:, 0], ii[:, 1], ii[:, 2]]
        allowed = np.isin(v, list(keep[tag]))
        lab[inside] = np.where(allowed, v, 0).astype(np.uint8)
        out[tag] = {'mask': lab, 'eu_coord': (-A).astype(np.float32),
                    'ev_coord': B.astype(np.float32), 'valid': inside, 'plane': plane,
                    'observed_classes': frozenset(keep[tag])}
    return out

# ====================================
# 2D framing
# ====================================

def frame_2d(m2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Frame the 2D view by detecting apex and long-axis direction."""
    pts = np.stack(np.where((m2d==LV) | (m2d==MY))).T.astype(np.float32)
    if pts.shape[0] < MIN_FRAME_PTS:
        raise ValueError("Not enough points to frame 2D view")
    mean = pts.mean(0)
    _, eigvecs = np.linalg.eigh(np.cov((pts - mean).T))
    axis = eigvecs[:, -1]
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    la = np.stack(np.where(m2d==LA)).T.astype(np.float32)
    base_ref = la.mean(0) if la.shape[0] > 0 else (mean + axis)
    if (base_ref - mean) @ axis < 0:
        axis = -axis
    proj = (pts - mean) @ axis
    apex = pts[np.argmin(proj)]
    lv_len = np.max(proj) - np.min(proj)
    return apex.astype(np.float32), axis.astype(np.float32), float(lv_len)

def inplane_coords(m2d: np.ndarray, apex: np.ndarray, e_v_2d: np.ndarray, ppv: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute in-plane coordinates (alpha, beta) for each pixel in the 2D view."""
    H, W = m2d.shape
    row, col = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    d_rc = np.stack([row - apex[0], col - apex[1]], axis=-1).astype(np.float32)
    e_u_2d = np.array([-e_v_2d[1], e_v_2d[0]], dtype=np.float32)
    alpha = (d_rc @ e_u_2d) / ppv
    beta = (d_rc @ e_v_2d) / ppv
    return alpha.astype(np.float32), beta.astype(np.float32)

def to_2d(cuts: Dict[str, dict], config: ViewConfig, mm_per_px: Optional[Dict[str, float]] = None, 
          case_id: Optional[str] = None) -> Dict[str, View2D]:
    """Convert cut planes to 2D views with framing and in-plane coordinates."""
    out = {}
    for tag, cut in cuts.items():
        mm = (mm_per_px or {}).get(tag, config.pitch_mm)
        ppv = px_per_voxel(config.mm_per_voxel, mm)
        try:
            apex, e_v_2d, lv_len = frame_2d(cut['mask'])
        except ValueError:
            out[tag] = View2D.empty(cut['mask'].shape, cut['observed_classes'])
            continue
        alpha, beta = inplane_coords(cut['mask'], apex, e_v_2d, ppv)
        # invert the detected 2D apex back into 3D via the coordinates cut_planes
        # recorded for every pixel: X = origin + eu_coord*e_u + ev_coord*e_v
        apex3d = None
        pl = cut.get('plane')
        if pl is not None:
            r0 = int(np.clip(round(float(apex[0])), 0, cut['mask'].shape[0] - 1))
            c0 = int(np.clip(round(float(apex[1])), 0, cut['mask'].shape[1] - 1))
            apex3d = (pl.origin
                      + float(cut['eu_coord'][r0, c0]) * pl.e_u
                      + float(cut['ev_coord'][r0, c0]) * pl.e_v).astype(np.float32)
        out[tag] = View2D(mask=cut['mask'], alpha=alpha, beta=beta, apex2d=apex, 
                          e_v_2d=e_v_2d, mm_per_px=float(mm), lv_len_px=lv_len, 
                          observed_classes=cut['observed_classes'], valid=cut['valid'],
                          apex3d=apex3d)
    if all(out[t].is_empty for t in out):
        raise EmptyObservationError(
            "both views empty: no plane retained enough LV+MY to frame", case_id)
    return out


# ====================================
# placement of 2D views in 3D
# ====================================
def canonical_frames(config: ViewConfig) -> Dict[str, PlaneParams]:
    """Return canonical frames for A4C and A2C views based on the configuration."""
    if config.canonical_a2c_angle_deg is None:
            raise ValueError("canonical_a2c_angle_deg must be provided")

    a0 = np.array(config.canonical_apex, np.float32)
    ev4 = np.array([0., 0., 1.], np.float32)        # long axis, apex -> base (+z)
    n4 = np.array([0., 1., 0.], np.float32)         # plane normal (+y)
    eu4 = np.cross(ev4, n4).astype(np.float32)      # = (-1, 0, 0)
    G4 = PlaneParams(normal=n4, origin=a0, e_u=eu4, e_v=ev4)    # where to place the A4C view in 3D  

    theta = np.deg2rad(config.canonical_a2c_angle_deg)
    eu2 = rotate_about_axis(eu4, ev4, theta).astype(np.float32)
    n2 = np.cross(eu2, ev4).astype(np.float32)   # so that cross(e_v, n) == e_u
    G2 = PlaneParams(normal=n2, origin=a0.copy(), e_u=eu2, e_v=ev4.copy())
    return {'a4c': G4, 'a2c': G2}


def place_view(view: View2D, frame: PlaneParams, config: ViewConfig, flip: bool = False, 
               out: Optional[np.ndarray] = None, grid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Place a 2D view into a 3D grid based on the given frame and configuration."""

    if grid is None:
        grid = full_grid(config.grid_size)
    L, W, H = config.grid_size
    vol = out if out is not None else np.zeros((L, W, H), np.uint8)

    rel = grid - frame.origin
    slab = np.abs(rel @ frame.normal) <= config.thickness_vox
    idx = np.where(slab)
    P = rel[idx]

    ppv = px_per_voxel(config.mm_per_voxel, view.mm_per_px)
    alpha_px = (P @ frame.e_u) * ppv        # lateral    [px]
    beta_px = (P @ frame.e_v) * ppv         # apex->base [px]
    if flip:
        alpha_px = -alpha_px
 
    e_u_2d = np.array([-view.e_v_2d[1], view.e_v_2d[0]], np.float32)
    yx = (view.apex2d[None] + beta_px[:, None] * view.e_v_2d[None] + alpha_px[:, None] * e_u_2d[None])
    r = np.round(yx[:, 0]).astype(int)
    c = np.round(yx[:, 1]).astype(int)
    Hh, Ww = view.mask.shape
    ok = (r >= 0) & (r < Hh) & (c >= 0) & (c < Ww)
    labs = np.zeros(P.shape[0], np.uint8)
    labs[ok] = view.mask[r[ok], c[ok]]
    vol[idx] = labs

    return vol, slab

def detected_basis(view: View2D, plane: PlaneParams) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the 3D basis vectors (e_u, e_v) for a detected 2D view based on the plane parameters."""
    ev2 = np.asarray(view.e_v_2d, np.float64)
    eu2 = np.array([-ev2[1], ev2[0]], np.float64)          # same +90 deg as inplane_coords
    e_u = plane.e_u.astype(np.float64)
    e_v = plane.e_v.astype(np.float64)
    e_v_3d = -ev2[0] * e_u + ev2[1] * e_v
    e_u_3d = -eu2[0] * e_u + eu2[1] * e_v
    return e_u_3d.astype(np.float32), e_v_3d.astype(np.float32)


def original_frames(views: Dict[str, View2D], geometry: ViewGeometry) -> Dict[str, PlaneParams]:
    """Return the original frames of the 2D views based on the detected apex and long-axis direction."""
    out = {}
    for tag in TAGS:
        p = geometry[tag]
        v = views[tag]
        if v.is_empty or v.apex3d is None:
            out[tag] = PlaneParams(normal=p.normal.copy(), origin=p.origin.copy(), e_u=p.e_u.copy(), e_v=p.e_v.copy())
            continue
        e_u_3d, e_v_3d = detected_basis(v, p)
        out[tag] = PlaneParams(normal=p.normal.copy(), origin=np.asarray(v.apex3d, np.float32).copy(), e_u=e_u_3d, e_v=e_v_3d)
    return out


def place(views: Dict[str, View2D], config: ViewConfig, frames: Optional[Dict[str, PlaneParams]] = None,
          geometry: Optional[ViewGeometry] = None, flips: Optional[Dict[str, bool]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Place 2D views into a 3D grid and return the merged volume and metadata."""
    frames = frames or canonical_frames(config)
    flips = flips or {}
    grid = full_grid(config.grid_size)
    per_view, slabs = {}, {}
    for tag in TAGS:
        if views[tag].is_empty:
            per_view[tag] = np.zeros(config.grid_size, np.uint8)
            slabs[tag] = np.zeros(config.grid_size, bool)
            continue
        vol, slab = place_view(views[tag], frames[tag], config, flip=bool(flips.get(tag, False)), grid=grid)
        per_view[tag] = vol 
        slabs[tag] = slab
    merged = np.where(per_view['a4c'] > 0, per_view['a4c'], per_view['a2c'])
    return merged.astype(np.uint8), {'views': per_view, 'slabs': slabs, 'frames': frames}
 
def full_grid(grid_size) -> np.ndarray:
    """[X,Y,Z,3] voxel-index grid"""
    return np.stack(np.meshgrid(*[np.arange(s) for s in grid_size], indexing='ij'), -1).astype(np.float32)

# ====================================
# sampling for the implicit view
# ====================================
def distance_weights(mask: np.ndarray, mm_per_px: float, d0_mm: float = 20.0) -> np.ndarray:
    """Compute distance-based weights for a binary mask."""
    fg = mask > 0
    if not fg.any() or ndimage is None:
        return np.ones(mask.shape, np.float32)
    d = ndimage.distance_transform_edt(~fg, sampling=(mm_per_px, mm_per_px))
    return (1.0 / (1.0 + (d / float(d0_mm)) ** 2)).astype(np.float32)

def weighted_sample(view: View2D, n_points: int, d0_mm: float = 20.0, 
                    rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """Sample points from the 2D view with distance-based weighting for background."""
    rng = rng or np.random.default_rng()
    if view.is_empty:
        z = np.zeros(0)
        return {
            'labels': z.astype(np.int64), 
            'alpha': z.astype(np.float32),
            'beta': z.astype(np.float32), 
            'n_fg': 0, 
            'n_bg': 0
        }
    
    lab = view.mask.reshape(-1)
    a = view.alpha.reshape(-1)
    b = view.beta.reshape(-1)

    ok = (view.valid.reshape(-1) if view.valid is not None else np.ones(lab.size, bool))
    fg = np.flatnonzero((lab > 0) & ok)
    bg = np.flatnonzero((lab == 0) & ok)
 
    n_bg = max(int(n_points) - fg.size, 0)
    if n_bg and bg.size:
        if n_bg < bg.size:
            w = distance_weights(view.mask, view.mm_per_px, d0_mm).reshape(-1)[bg]
            p = w / w.sum()
            bg = rng.choice(bg, size=n_bg, replace=False, p=p)
    else:
        bg = bg[:0]
    keep = np.concatenate([fg, bg])
    return {
        'labels': lab[keep].astype(np.int64),
        'alpha': a[keep].astype(np.float32),
        'beta': b[keep].astype(np.float32), 
        'n_fg': int(fg.size), 
        'n_bg': int(bg.size)
    }



# ====================================
# diagnostics
# ====================================
# def measure_inter_plane_angles(volumes, config: ViewConfig, rng: Optional[np.random.Generator] = None):
#     """Measure inter-plane angles for a list of 3D volumes."""
#     out, signed, skipped = [], [], []
#     for i, seg in enumerate(volumes):
#         try:
#             g = build_geometry(seg, config, rng, case_id=f"volume[{i}]")
#             out.append(g.inter_plane_angle_deg); signed.append(g.signed_a2c_angle_deg)
#         except DegenerateGeometryError as e:
#             skipped.append({'index': i, 'case_id': e.case_id, 'reason': e.reason})
#             out.append(float('nan')); signed.append(float('nan'))
#     a = np.asarray(out, float)
#     ok = a[np.isfinite(a)]
#     return {'angles': a, 'signed': np.asarray(signed, float),
#             'median': float(np.median(ok)) if ok.size else float('nan'),
#             'mean': float(ok.mean()) if ok.size else float('nan'),
#             'std': float(ok.std(ddof=1)) if ok.size > 1 else float('nan'),
#             'p05': float(np.percentile(ok, 5)) if ok.size else float('nan'),
#             'p95': float(np.percentile(ok, 95)) if ok.size else float('nan'),
#             'signed_median': float(np.median(np.asarray(signed, float)[
#                 np.isfinite(signed)])) if np.isfinite(signed).any() else float('nan'),
#             'n': int(ok.size), 'skipped': skipped}
