# lcunet/sampler_pose.py

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from common.views import N_CLASSES, TAGS
from common.patches import PatchConfig, view_patches_with_mask, landmark_vector
from lcunet.pose_net import PoseNet, build_pose_net

GAUGE_KEYS = ('anchor', 'canonical_a2c_angle_deg', 'canonical_apex', 'target_lv_len_mm')


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return list(np.atleast_1d(a)) == list(np.atleast_1d(b))


def load_pose_net(path: str, device) -> Tuple[PoseNet, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = build_pose_net(cov=ck.get('cov', 'full'),
                         n_landmarks=int(ck.get('n_landmarks', 0)),
                         dims=ck.get('dims', [32, 64, 128, 256]),
                         hidden=int(ck.get('hidden', 256)),
                         min_sigma=float(ck.get('min_sigma', 0.05))).to(device)
    net.load_state_dict(ck['net'])       # brings z_mean / z_sd with it
    net.eval()
    if ck.get('anchor') != 'a4c':
        raise SystemExit(f"pose net checkpoint {path} is not anchored to the A4C frame, which is required for this sampler")
    return net, ck


def check_agree(pose_ck: dict, unet_ck: dict) -> dict:
    """check if the two checkpoints agree on the gauge constants"""
    bad = [k for k in GAUGE_KEYS if not _same(pose_ck.get(k), unet_ck.get(k))]
    if bad:
        detail = ', '.join(f"{k}: pose={pose_ck.get(k)} unet={unet_ck.get(k)}" for k in bad)
        raise SystemExit(f"gauge mismatch between the checkpoints ({detail})")
    return {k: pose_ck.get(k) for k in GAUGE_KEYS}


class PosePosterior:

    def __init__(self, pose_net: PoseNet, device, patch_cfg: PatchConfig, use_landmarks: bool = False):
        self.pose_net = pose_net
        self.device = device
        self.patch_cfg = patch_cfg
        self.use_landmarks = bool(use_landmarks)

    @torch.no_grad()
    def posterior(self, views: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        patch, valid = view_patches_with_mask(views, TAGS, self.patch_cfg)
        p = torch.from_numpy(patch.astype(np.int64))[None].to(self.device)
        v = torch.from_numpy(valid)[None].to(self.device)
        lm = None
        if self.use_landmarks:
            lm = torch.from_numpy(
                landmark_vector(views, TAGS, N_CLASSES))[None].to(self.device)
        return self.pose_net(p, v, lm)


def build_pose_posterior(pose_ckpt: str, device) -> Tuple[PosePosterior, dict]:
    pose_net, pose_ck = load_pose_net(pose_ckpt, device)
    patch_cfg = PatchConfig(size=tuple(pose_ck.get('patch_size', (192, 192))),
                            apex_rc=tuple(pose_ck.get('patch_apex_rc', (8.0, 96.0))),
                            mm_per_px=pose_ck.get('patch_mm_per_px', 1.0))
    pose = PosePosterior(pose_net, device, patch_cfg, use_landmarks=bool(pose_ck.get('n_landmarks', 0)))
    return pose, pose_ck