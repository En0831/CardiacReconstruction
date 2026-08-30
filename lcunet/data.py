# lcunet/data.py

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

import nibabel as nib

from common.views import TAGS, ViewConfig, build_geometry, cut_planes, to_2d, place, original_frames, EmptyObservationError
from common.perturb import arm_specs, observation_perturbation, placement_perturbation
from common.splits import Split
from common.patches import PatchConfig, view_patches_with_mask, landmark_vector
from common import canonical
from implicit.data import resample_labels

N_CLASSES = 6


class LCUNetDataset(Dataset):
    def __init__(self, files: Sequence[str], data_type: str, arm: str = 'A0',
                 sigma_rot_deg: float = 15.0, sigma_trans_mm: float = 8.0,
                 grid_size: Sequence[int] = (96, 96, 128), voxel_size: float = 2.0,
                 view_config: str = 'left', seed: int = 12345,
                 max_retries: int = 10, device: str = 'cpu',
                 canonical_a2c_angle_deg: Optional[float] = None,
                 canonical_apex: Optional[Sequence[float]] = None,
                 anchor: str = 'anatomical',
                 patch_size: Sequence[int] = (192, 192),
                 patch_apex_rc: Sequence[float] = (8.0, 96.0),
                 patch_mm_per_px: Optional[float] = 1.0):
        super().__init__()
        assert data_type in ('train', 'valid', 'test')
        assert arm in ('A0', 'A1', 'A2', 'C', 'A2c')
        if arm in ('C', 'A2c') and canonical_a2c_angle_deg is None:
            raise ValueError("arm C or A2c requires canonical_a2c_angle_deg to be specified")
        self.data_type = data_type
        self.arm = arm
        self.seed = seed
        self.max_retries = max_retries
        self.anchor = anchor
        # 
        self.patch_cfg = PatchConfig(size=tuple(int(v) for v in patch_size), 
                                     apex_rc=tuple(float(v) for v in patch_apex_rc),
                                     mm_per_px=patch_mm_per_px)
        cfg_kw = {}
        if canonical_apex is not None:
            cfg_kw['canonical_apex'] = tuple(float(x) for x in canonical_apex)
        self.cfg = ViewConfig(grid_size=tuple(grid_size), mm_per_voxel=voxel_size, view_config=view_config, 
                              canonical_a2c_angle_deg=(canonical_a2c_angle_deg if arm in ('C', 'A2c') else None), **cfg_kw)

        # Add observation perturbation for A1/C, placement perturbation for A2/A2c, and no perturbation for A0
        spec_arm = 'A1' if arm == 'C' else ('A2' if arm == 'A2c' else arm)
        self.obs_spec, self.place_spec = arm_specs(sigma_rot_deg, sigma_trans_mm)[spec_arm]

        self.volumes = []
        self.casenames = []
        for path in files:
            seg = nib.load(path).get_fdata().astype(np.uint8)
            self.volumes.append(resample_labels(seg, grid_size, N_CLASSES, device))
            self.casenames.append(path)

        # ----- arm C precomputation (per volume) -----
        # gauge: volume -> canonical, target: 3d seg in canonical
        # if the anchor is a4c, the gauge/target are computed in _synth_c
        self.gauges, self.targets_c = [], [] 
        if arm in ('C', 'A2c') and self.anchor == 'anatomical':
            for seg in self.volumes:
                gauge = canonical.build_gauge(seg, self.cfg)
                self.gauges.append(gauge)
                self.targets_c.append(canonical.resample_to_canonical(seg, gauge, self.cfg))

    def __len__(self) -> int:
        return len(self.volumes)

    def _rng(self, item: int) -> np.random.Generator:
        if self.data_type == 'train':
            return np.random.default_rng()
        return np.random.default_rng(self.seed + item)

    # --------- A0-A2 -------------------------
    def _synth(self, seg_3d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        obs_rng = np.random.default_rng(rng.integers(1 << 32))
        place_rng = np.random.default_rng(rng.integers(1 << 32))
        geom_rng = np.random.default_rng(rng.integers(1 << 32))

        obs_turb = (None if self.obs_spec.is_identity() else observation_perturbation(self.obs_spec, self.cfg, obs_rng))
        geometry = build_geometry(seg_3d, self.cfg, geom_rng, obs_perturb=obs_turb)
        views = to_2d(cut_planes(seg_3d, geometry, self.cfg), self.cfg)

        base = original_frames(views, geometry)
        if self.place_spec.is_identity():
            frames = base
        else:
            frames, _ = placement_perturbation(self.cfg, geometry, self.place_spec, place_rng, a4c_fixed=False, base_frames=base)
        placed, _ = place(views, self.cfg, frames=frames)
        return placed

    # --------- C and A2c -------------------------
    def _synth_c(self, item: int, rng: np.random.Generator) -> dict:
        seg_3d = self.volumes[item]
        obs_rng = np.random.default_rng(rng.integers(1 << 32))
        geom_rng = np.random.default_rng(rng.integers(1 << 32))

        obs_hook = (None if self.obs_spec.is_identity()
                    else observation_perturbation(self.obs_spec, self.cfg, obs_rng))
        geometry_obs = build_geometry(seg_3d, self.cfg, geom_rng,
                                      obs_perturb=obs_hook,
                                      case_id=self.casenames[item])
        views = to_2d(cut_planes(seg_3d, geometry_obs, self.cfg), self.cfg,
                      case_id=self.casenames[item])

        # compute gauge and target for each case
        if self.anchor == 'a4c':
            base = original_frames(views, geometry_obs)
            gauge = canonical.build_gauge(seg_3d, self.cfg, anchor='a4c', frame_a4c=base['a4c'])
            target_c = canonical.resample_to_canonical(seg_3d, gauge, self.cfg)
        else:
            gauge = self.gauges[item]
            target_c = self.targets_c[item]
        truth = canonical.compute_truth(seg_3d, views, geometry_obs, self.cfg, gauge=gauge, target=target_c)
        z_vec, z_valid = canonical.z_pack(truth.z, self.cfg)
        
        z_place = truth.z
        # Add placement perturbation
        if not self.place_spec.is_identity():
            place_rng = np.random.default_rng(rng.integers(1 << 32))
            per = [self.place_spec.sigma_longaxis_deg, self.place_spec.sigma_tilt_deg,
                self.place_spec.sigma_inplane_deg, self.place_spec.sigma_trans_long_mm,
                self.place_spec.sigma_trans_lat_mm, self.place_spec.sigma_trans_n_mm]
            d = np.array([place_rng.normal(0.0, s) for _ in canonical.TAGS for s in per], np.float64)
            if self.anchor == 'a4c':
                d[:canonical.Z_DIM_VIEW] = 0.0
            z_place = canonical.z_unpack(z_vec + d, self.cfg, z_valid)
        placed, info = canonical.place_at_z(views, z_place, self.cfg)


        z_mask = np.repeat(z_valid, canonical.Z_DIM_VIEW).astype(bool)
        if self.anchor == 'a4c':    # The first 6 dim (a4c) are not supervised
            z_mask[:canonical.Z_DIM_VIEW] = False
        return {
            'placed': placed,
            'target': target_c,
            'z_vec': z_vec.astype(np.float32),          # [12] supervision for 部品2
            'z_valid': z_valid,                         # [2]  per-view validity
            'z_mask': z_mask,                           # [12] NLL supervision mask
            'views': views,                             # raw View2D dict (masks,
            'slabs': info['slabs'],                     # apex2d, e_v_2d, lv_len_px)
        }

    def _draw(self, item: int):
        """Draw one sample from the dataset, with retries for empty observations in training"""
        if self.data_type == 'train':
            for _ in range(self.max_retries):
                try:
                    if self.arm in ('C', 'A2c'):
                        return self._synth_c(item, self._rng(item))
                    return self._synth(self.volumes[item], self._rng(item))
                except EmptyObservationError:
                    continue
            raise EmptyObservationError(f"max_retries={self.max_retries} exceeded for {self.casenames[item]}")
        # valid/test: deterministic single attempt; caller handles the exception
        if self.arm in ('C', 'A2c'):
            return self._synth_c(item, self._rng(item))
        return self._synth(self.volumes[item], self._rng(item))

    def __getitem__(self, item: int):
        out = self._draw(item)
        if self.arm in ('C', 'A2c'):
            return (torch.from_numpy(out['placed'].astype(np.int64)),
                    torch.from_numpy(out['target'].astype(np.int64)))
        return (torch.from_numpy(out.astype(np.int64)),
                torch.from_numpy(self.volumes[item].astype(np.int64)))


    def pose_item(self, item: int) -> dict:
        assert self.arm == 'C', "pose supervision is defined by the canonical gauge"
        out = self._draw(item)
        if not out['z_mask'].any():
            raise EmptyObservationError("no supervised z dimensions", self.casenames[item])
        patch, valid = view_patches_with_mask(out['views'], TAGS, self.patch_cfg)
        lm = landmark_vector(out['views'], TAGS, N_CLASSES)
        return {
            'patch': torch.from_numpy(patch.astype(np.int64)),   # [2,R,C]
            'valid': torch.from_numpy(valid),                    # [2,R,C] bool
            'landmarks': torch.from_numpy(lm),                   # [16] float32
            'z': torch.from_numpy(out['z_vec']),                 # [13] float32
            'z_mask': torch.from_numpy(out['z_mask']),           # [13] bool
        }


class PoseDataset(Dataset):
    def __init__(self, base: LCUNetDataset, skip_empty: bool = True):
        assert base.arm == 'C', "pose supervision needs the canonical gauge"
        self.base = base
        self.skip_empty = bool(skip_empty)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> dict:
        if not self.skip_empty:
            return self.base.pose_item(item)
        try:
            return self.base.pose_item(item)
        except EmptyObservationError:
            return None

def collate_skip(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None

def build_lcunet_dataset(split: Split, subset: str, arm: str = 'A0', **kwargs) -> LCUNetDataset:
    return LCUNetDataset(split.paths(subset), data_type=subset, arm=arm, **kwargs)

def build_pose_dataset(split: Split, subset: str, skip_empty: bool = True, **kwargs) -> PoseDataset:
    return PoseDataset(build_lcunet_dataset(split, subset, arm='C', **kwargs), skip_empty=skip_empty)