# eval/case.py

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from common.canonical import to_original
from common.metrics import mean_dice
from common.perturb import AugmentSpec, observation_perturbation
from common.views import ViewConfig, build_geometry, cut_planes, to_2d
import common.canonical as C


@dataclass
class CaseBundle:
    case_id: str
    seg: np.ndarray                      # original-frame GT
    views: dict                          # 2D observations
    geometry_obs: object                 # perturbed (cut) geometry
    truth: C.CanonicalTruth              # gauge, z*, canonical target
    cfg: ViewConfig


def build_case(seg: np.ndarray, cfg: ViewConfig, case_id: str,
               rng: np.random.Generator, obs_spec: Optional[AugmentSpec] = None, anchor: str="anatomical") -> CaseBundle:
    geometry_clean = build_geometry(seg, cfg)
    perturbation = observation_perturbation(obs_spec, cfg, rng) if obs_spec is not None else None
    geometry_obs = (build_geometry(seg, cfg, rng, obs_perturb=perturbation)
                    if perturbation else geometry_clean)
    views = to_2d(cut_planes(seg, geometry_obs, cfg), cfg, case_id=case_id)
    truth = C.compute_truth(seg, views, geometry_obs, cfg, anchor=anchor)

    return CaseBundle(case_id=case_id, seg=seg, views=views, geometry_obs=geometry_obs, truth=truth, cfg=cfg)
    

def roundtrip_floor(bundle: CaseBundle) -> dict:
    """GT -> canonical -> back vs GT: the canonical methods' unified ceiling."""
    back = to_original(bundle.truth.target, bundle.truth.gauge, bundle.seg.shape)
    return mean_dice(back, bundle.seg)
