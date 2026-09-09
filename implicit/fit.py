# implicit/fit.py

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from common.views import View2D, PlaneParams, weighted_sample, TAGS
from common import perturb as _pt
from implicit.implicits_echo import MultiClassAutoDecoder, PartialLabelLoss


# ==============================
# pose
# ==============================
def rodrigues_torch(alpha: torch.Tensor) -> torch.Tensor:
    """axis-angle [3] -> rotation matrix [3,3] (differentiable)."""
    theta = torch.linalg.norm(alpha) + 1e-8
    k = alpha / theta
    K = torch.zeros(3, 3, dtype=alpha.dtype, device=alpha.device)
    K[0, 1], K[0, 2] = -k[2], k[1]
    K[1, 0], K[1, 2] = k[2], -k[0]
    K[2, 0], K[2, 1] = -k[1], k[0]
    eye = torch.eye(3, dtype=alpha.dtype, device=alpha.device)
    return eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

class ViewPose(torch.nn.Module):
    """Per-view rigid pose: axis-angle rotation + translation, both zero-initialised."""

    def __init__(self, device):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(3, device=device))
        self.t = torch.nn.Parameter(torch.zeros(3, device=device))

    def forward(self, anchor, eu, ev, a_ij, b_ij, reflect=False):
        """
        anchor [3], eu/ev [3], a_ij/b_ij [N] -> x [N,3] in grid voxel units.
        """
        R = rodrigues_torch(self.alpha)
        e_u_r = R @ eu
        e_v_r = R @ ev
        if reflect: # descrested reflection across the plane of the view
            e_u_r = -e_u_r
        origin = anchor + self.t
        return origin[None] + a_ij[:, None] * e_u_r[None] + b_ij[:, None] * e_v_r[None]

def _identity_place(origin, e_u, e_v, a_ij, b_ij, reflect=False):
    """A4C: identity pose (no rotation, no translation, but can reflect)."""
    if reflect:
        e_u = -e_u
    return origin[None] + a_ij[:, None] * e_u[None] + b_ij[:, None] * e_v[None]


# ==============================
# turning 2D views into tensors for fitting
# ==============================
def _identity_observed_map(observed_classes: Sequence[int]) -> Dict[int, int]:
    return {int(c): int(c) for c in observed_classes}

def views_to_tensors(views: Dict[str, View2D], frames: Dict[str, PlaneParams], device, n_points: int,
                     d0_mm: float=20.0, rng: Optional[np.random.Generator]=None) -> Dict[str, dict]:
    """Convert a dict of View2D to a dict of tensors for fitting."""
    rng = rng or np.random.default_rng()
    out = {}
    for tag in TAGS:
        v = views[tag]
        s = weighted_sample(v, n_points, d0_mm=d0_mm, rng=rng)
        f = frames[tag]
        out[tag] = dict(
            labels = torch.from_numpy(s['labels']).to(device),
            a = torch.from_numpy(s['alpha']).to(device),
            b = torch.from_numpy(s['beta']).to(device),
            origin = torch.from_numpy(np.asarray(f.origin)).float().to(device),
            e_u = torch.from_numpy(np.asarray(f.e_u)).float().to(device),
            e_v = torch.from_numpy(np.asarray(f.e_v)).float().to(device),
            observed_classes = sorted(int(c) for c in v.observed_classes),
            n_fg=s['n_fg'], 
            n_bg=s['n_bg'],
        )
    return out

def voxel_to_mm(x_vox: torch.Tensor, mm_per_voxel: float) -> torch.Tensor:
    """Convert voxel coordinates to mm coordinates."""
    return x_vox * mm_per_voxel + mm_per_voxel / 2.0


# ==============================
# fitting
# ==============================
class FitConfig:
    def __init__(self, n_iter_latent=100, n_iter_total=1000, lr=1e-2, 
                 lat_reg_lambda=None, lam_t=0.0, lam_r=0.0, num_classes=6):
        self.n_iter_latent = n_iter_latent
        self.n_iter_total = n_iter_total
        self.n_iter_joint = max(0, n_iter_total - n_iter_latent)
        self.lr = lr
        self.lat_reg_lambda = lat_reg_lambda
        self.lam_t = lam_t
        self.lam_r = lam_r
        self.num_classes = num_classes

def _make_criteria(tensor, num_classes, device):
    """Make a loss criterion for the given tensor."""
    crit = {}
    for tag, v in tensor.items():
        obs = v['observed_classes']
        crit[tag] = PartialLabelLoss(_identity_observed_map(obs), num_classes=num_classes, observed_classes=obs).to(device)
    return crit

def fit_once(net, tensors, z_init, config, mm_per_voxel, lat_reg_lambda, device, reflect=False, use_pose=True, fit_a4c=False, verbose=False):
    z = z_init.clone().detach().to(device).requires_grad_(True)
    poses = {}
    if use_pose:
        poses['a2c'] = ViewPose(device).to(device)
        if fit_a4c:
            poses['a4c'] = ViewPose(device).to(device)
    pose_params = [p for pose in poses.values() for p in pose.parameters()]

    crit = _make_criteria(tensors, config.num_classes, device)

    def place(tag, v):
        if tag in poses:
            return poses[tag](v['origin'], v['e_u'], v['e_v'], v['a'], v['b'], reflect=reflect)
        else:
            return _identity_place(v['origin'], v['e_u'], v['e_v'], v['a'], v['b'], reflect=reflect)

    def data_loss():
        total = z.new_zeros(())
        for tag, v in tensors.items():
            x_mm = voxel_to_mm(place(tag, v), mm_per_voxel)
            logits = net(z[None], x_mm[None])
            mask = torch.ones_like(v['labels'], dtype=torch.bool)[None]
            total += crit[tag](logits, v['labels'][None], mask)
        return total / max(len(tensors), 1)

    def pose_reg():
        r = z.new_zeros(())
        for pose in poses.values():
            r = r + config.lam_t * (pose.t ** 2).sum() + config.lam_r * (pose.alpha ** 2).sum()
        return r

    def total_loss(step):
        loss = data_loss() + pose_reg()
        if lat_reg_lambda and lat_reg_lambda > 0:
            loss = loss + min(1.0, step / 100.0) * lat_reg_lambda * (z ** 2).sum()
        return loss

    def snapshot(step, loss):
        return {
            'loss': float(loss),
            'z': z.detach().clone(),
            'pose': _snapshot_pose(poses),
            'step': step,
            }

    with torch.no_grad():
        best = snapshot(0, total_loss(0))

    # ------ phase 1: latent-only optimization ------
    optA = torch.optim.Adam([z], lr=config.lr)
    for i in range(config.n_iter_latent):
        optA.zero_grad()
        loss = total_loss(i)
        loss.backward()
        optA.step()
        if verbose and (i % 100 == 0 or i == config.n_iter_latent - 1):
            print(f"[latent] step {i:04d} loss {loss.item():.4f}")
        if loss.item() < best['loss']:
            best = snapshot(i, loss.item())

    # ------ phase 2: joint optimization ------
    groups = [{'params': [z], 'lr': config.lr}]
    if pose_params:
        groups.append({'params': pose_params, 'lr': config.lr})
    optB = torch.optim.Adam(groups)
    for j in range(config.n_iter_joint):
        step = config.n_iter_latent + j
        optB.zero_grad()
        loss = total_loss(step)
        loss.backward()
        optB.step()
        if verbose and (j % 100 == 0 or j == config.n_iter_joint - 1):
            print(f"[joint] step {step:04d} loss {loss.item():.4f}")
        if loss.item() < best['loss']:
            best = snapshot(step, loss.item())

    best['reflect'] = reflect
    best['z_norm'] = float(torch.linalg.norm(best['z']))
    return best

def _snapshot_pose(poses: Dict[str, ViewPose]) -> Dict[str, dict]:
    out = {}
    for tag in TAGS:
        if tag in poses:
            al = poses[tag].alpha.detach().cpu().numpy()
            t = poses[tag].t.detach().cpu().numpy()
            out[tag] = {
                'R': _pt.rodrigues(al),
                't': t.astype(np.float64),
                'alpha': al.astype(np.float64),
                }
        else:
            out[tag] = {
                'R': np.eye(3),
                't': np.zeros(3),
                'alpha': np.zeros(3),
            }
    return out

def fit_latent(net, views: Dict[str, View2D], frames: Dict[str, PlaneParams], latents_train: torch.Tensor, config: FitConfig, mm_per_voxel: float,
               device, lat_reg_lambda: Optional[float] = None, n_points: int = 40000, d0_mm: float = 20.0, n_restart: int = 2, use_pose: bool = True, 
               fit_a4c: bool = False, chirality: bool = False, rng: Optional[np.random.Generator] = None) -> dict:
    """Fit latent code and optionally pose to 2D views."""
    rng = rng or np.random.default_rng()
    tensors = views_to_tensors(views, frames, device, n_points, d0_mm, rng)
 
    reg = lat_reg_lambda if lat_reg_lambda is not None else config.lat_reg_lambda
 
    mu = latents_train.mean(0)
    extra = max(0, n_restart - 1)
    idx = rng.choice(len(latents_train), extra, replace=False) if extra else []
    inits = [mu] + [latents_train[j] for j in idx]
    reflects = [False, True] if chirality else [False]
 
    best = None
    for reflect in reflects:
        for init in inits:
            r = fit_once(net, tensors, init, config, mm_per_voxel, reg, device,
                         reflect=reflect, use_pose=use_pose, fit_a4c=fit_a4c)
            if best is None or r['loss'] < best['loss']:
                best = r
    best['n_fg'] = {t: tensors[t]['n_fg'] for t in TAGS}
    best['n_bg'] = {t: tensors[t]['n_bg'] for t in TAGS}
    best['lat_reg_lambda'] = reg
    return best

def latent_reg_from_ckpt(ckpt: dict, default: float=1e-4) -> float:
    args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    val = args.get('lat_reg_lambda', None) if isinstance(args, dict) else None
    return float(val) if val is not None else float(default)


def load_prior(path: str, device):
    """Prior checkpoint -> (net, latents_train, lat_reg_lambda, fingerprint, args)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = dict(ck['args'])
    for k in ('gauge_space', 'gauge_anchor', 'canonical_apex', 'canonical_a2c_angle_deg'):
        if k in ck:
            a[k] = ck[k]
    net = MultiClassAutoDecoder(
        lat_dim=a['latent_dim'], spatial_dim=3, image_size=ck['image_size'].clone(),
        occnet_num_layers=a['op_num_layers'], occnet_layers_with_coords=a['op_coord_layers'],
        num_classes=a['num_classes']).to(device)
    net.load_state_dict(ck['net'])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)          # the prior is frozen at test time
    return (net, ck['latents_train'].to(device),
            latent_reg_from_ckpt(ck), ck.get('split_fingerprint', ''), a)
