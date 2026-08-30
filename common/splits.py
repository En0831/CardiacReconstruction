# common/splits.py

"""Splitting WHS++ dataset into train/valid/test dataset"""

from __future__ import annotations
 
import argparse
import glob
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence
 
import numpy as np
 
DEFAULT_SEED = 12345
DEFAULT_RATIOS = (0.6, 0.1)     # (train, valid)
PATTERN = '*.nii.gz'
 
 
def case_id(path: str) -> str:
    """Filename without the .nii.gz"""
    base = os.path.basename(path)
    for ext in ('.nii.gz', '.nii'):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]
 
 
def split_indices(n: int, seed: int = DEFAULT_SEED, ratios: Sequence[float] = DEFAULT_RATIOS) -> Dict[str, np.ndarray]:
    """Randomly split n cases into train/valid/test indices"""
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        perm = np.random.permutation(n)
    finally:
        np.random.set_state(state)
    n_train = int(ratios[0] * n)
    n_valid = int(ratios[1] * n)
    return {
        'train': perm[:n_train],
        'valid': perm[n_train:n_train + n_valid],
        'test': perm[n_train + n_valid:],
    }
 

##############################
# split data class
##############################
@dataclass
class Split:
    data_dir: str
    files: List[str]
    indices: Dict[str, np.ndarray]
    seed: int = DEFAULT_SEED
    ratios: Sequence[float] = DEFAULT_RATIOS
    fingerprint: str = field(init=False)

    def __post_init__(self):
        """keep fingerprints to use the same split for all the training"""
        payload = json.dumps(
            {'seed': self.seed, 'ratios': list(self.ratios),
             'splits': {k: [case_id(self.files[i]) for i in v] for k, v in self.indices.items()}}, sort_keys=True)
        self.fingerprint = hashlib.sha1(payload.encode()).hexdigest()[:16]

    def paths(self, subset: str) -> List[str]:
        return [self.files[i] for i in self.indices[subset]]
 
    def ids(self, subset: str) -> List[str]:
        return [case_id(p) for p in self.paths(subset)]
  
    def __len__(self) -> int:
        return len(self.files)
 
    def assert_matches(self, other_fingerprint: str, what: str = 'checkpoint') -> None:
        """check fingerprint and raise an error if it does not match"""
        if other_fingerprint and other_fingerprint != self.fingerprint:
            raise RuntimeError(f"{what} fingerprint {other_fingerprint} does not match split fingerprint {self.fingerprint}")
 
    def describe(self) -> str:
        n = {k: len(v) for k, v in self.indices.items()}
        return (f"split[{self.fingerprint}] {len(self)} cases "
                f"(train {n['train']} / valid {n['valid']} / test {n['test']}) "
                f"seed={self.seed} from {self.data_dir}")

  
def load_split(data_dir: str, seed: int = DEFAULT_SEED) -> Split:
    files = sorted(glob.glob(os.path.join(data_dir, PATTERN)))
    if not files:
        raise FileNotFoundError(f"no files matching {PATTERN!r} under {data_dir!r}")
    return Split(data_dir=data_dir, files=files, indices=split_indices(len(files), seed, DEFAULT_RATIOS), seed=seed)
 

def add_split_args(ap) -> None:
    ap.add_argument('--data_dir', default='../../data/whs/')
    ap.add_argument('--split_seed', type=int, default=DEFAULT_SEED)


def split_from_args(args) -> Split:
    return load_split(args.data_dir, args.split_seed)