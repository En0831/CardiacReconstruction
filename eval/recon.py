# eval/recon.py

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import nibabel as nib
import implicit.fit as F

from common.splits import add_split_args, split_from_args, case_id
from common.views import N_CLASSES, TAGS, ViewConfig, place, original_frames, DegenerateGeometryError, EmptyObservationError
from common.perturb import isotropic_spec, placement_perturbation
from common.metrics import MulticlassEvaluator, DEFAULT_STRUCTS, volumes_ml_by_struct, mean_dice, majority_vote
from common.io import recon_path, save_nifti
from common import canonical as C
from lcunet.infer import load_unet, complete
from lcunet.sampler_pose import build_pose_posterior, check_agree
from implicit.data import resample_labels
from implicit.decode import decode_volume
from camus.data import list_patients, load_case, read_reference_ef
from eval.case import build_case, roundtrip_floor

STRUCTS = [n for n, _ in DEFAULT_STRUCTS]
LEFT = ('LV', 'MY', 'LA')
PLACEMENTS = ('original', 'canonical_oracle', 'canonical_zero', 'canonical_posterior')
CANONICAL = {'canonical_oracle', 'canonical_zero', 'canonical_posterior'}


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def parse_model(spec: str) -> Tuple[str, str]:
    if ':' not in spec:
        raise SystemExit("--model must be kind:path, e.g. lcunet:ckpts/a1.pth")
    kind, path = spec.split(':', 1)
    if kind not in ('lcunet', 'implicit'):
        raise SystemExit(f"unknown model kind {kind!r}")
    return kind, path


def load_implicit(path: str, device, args):
    net, lat, reg, fp, a = F.load_prior(path, device)
    cfg = F.FitConfig(n_iter_latent=args.n_iter_latent, n_iter_total=args.n_iter_total, lr=args.fit_lr)
    return SimpleNamespace(net=net, latents=lat, reg=reg, cfg=cfg, fp=fp, args=a,
                           use_pose=not args.pose_off, gauge_space=a.get('gauge_space', 'original'))


def implicit_reconstruct(m, views, frames, cfg: ViewConfig, device, rng: np.random.Generator) -> np.ndarray:
    res = F.fit_latent(m.net, views, frames, m.latents, m.cfg, cfg.mm_per_voxel, device, lat_reg_lambda=m.reg, 
                       use_pose=m.use_pose, fit_a4c=True, chirality=False, rng=rng)
    return decode_volume(m.net, res['z'], cfg.grid_size, cfg.mm_per_voxel, device)


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------
def z_vectors(mode: str, views, cfg, truth, pose, n: int, temperature: float, rng: np.random.Generator) -> np.ndarray:
    """[n, 12] full z vectors for a canonical placement mode."""
    if mode == 'canonical_zero':
        return np.zeros((1, C.Z_DIM), np.float64)
    if mode == 'canonical_oracle':
        vec, _ = C.z_pack(truth.z, cfg)
        return vec[None].astype(np.float64)
    mu, L = pose.posterior(views)
    if temperature <= 0:
        z6 = pose.pose_net.unstandardise_z(mu).cpu().numpy()
    else:
        g = torch.Generator(device='cpu').manual_seed(int(rng.integers(1 << 31)))
        z6 = pose.pose_net.sample(mu, L * temperature, n, generator=g)[0].cpu().numpy()
    out = np.zeros((z6.shape[0], C.Z_DIM), np.float64)
    out[:, 6:12] = z6
    return out


def perturb_z(vec, spec, rng, anchor='a4c'):
    if spec is None or spec.is_identity():
        return vec
    per = [spec.sigma_longaxis_deg, spec.sigma_tilt_deg, spec.sigma_inplane_deg,
           spec.sigma_trans_long_mm, spec.sigma_trans_lat_mm, spec.sigma_trans_n_mm]
    d = np.array([rng.normal(0.0, s) for _ in TAGS for s in per], np.float64)
    if anchor == 'a4c':
        d[:C.Z_DIM_VIEW] = 0.0
    return vec + d


# ---------------------------------------------------------------------------
def whs_rows(kind, model, pose, bundle, seg, mode: str, anchor, spec, n: int, temperature: float, 
             rng, cfg, device, ev: MulticlassEvaluator) -> Tuple[List[dict], List[np.ndarray], List[np.ndarray]]:
    fl_mean = roundtrip_floor(bundle)
    rows, vols, ins = [], [], []

    if mode == 'original':
        base = original_frames(bundle.views, bundle.geometry_obs)
        imp_space = getattr(model, 'gauge_space', 'original')
        draws = range(max(n, 1)) if spec is not None else range(1)
        for k in draws:
            frames = base
            if spec is not None and not spec.is_identity():
                frames, _ = placement_perturbation(cfg, bundle.geometry_obs, spec, rng, a4c_fixed=False, base_frames=base)
            placed, _ = place(bundle.views, cfg, frames=frames)
            if kind == 'lcunet':
                vol = complete(model, placed, device)
                space = 'original'
            else:
                vol = implicit_reconstruct(model, bundle.views, frames, cfg, device, rng)
                space = imp_space
            rows.append(dict(sample_idx=k, space=space, target='canonical' if space == 'canonical' else 'original',
                             pred=vol, placed=placed, floor_mean=fl_mean)) 
    else:
        zs = z_vectors(mode, bundle.views, cfg, bundle.truth, pose, n, temperature, rng)
        _, z_valid = C.z_pack(bundle.truth.z, cfg)
        imp_space = getattr(model, 'gauge_space', 'original')
        for k in range(zs.shape[0]):
            vec = perturb_z(zs[k], spec, rng, anchor=anchor)
            z = C.z_unpack(vec, cfg, z_valid)
            placed, _ = C.place_at_z(bundle.views, z, cfg)
            if kind == 'lcunet':
                vol = complete(model, placed, device)
                space = 'canonical'
            else:
                vol = implicit_reconstruct(model, bundle.views, C.frames_from_z(z, cfg), cfg, device, rng)
                space = imp_space
            rows.append(dict(sample_idx=k, space=space, target='canonical' if space == 'canonical' else 'original',
                             pred=vol, placed=placed, z=vec[6:12].round(4).tolist(), floor_mean=fl_mean))

    out = []
    for r in rows:
        pred = r.pop('pred'); placed = r.pop('placed')
        uni = (C.to_original(pred, bundle.truth.gauge, seg.shape)
               if r['space'] == 'canonical' else pred)
        per = ev.add(uni, seg, case_id=bundle.case_id)
        row = dict(r)
        for s in STRUCTS:
            d = per.get(s, {})
            row[f'dice_{s}'] = d.get('dice', float('nan'))
            row[f'assd_{s}'] = d.get('assd', float('nan'))
            row[f'hd95_{s}'] = d.get('hd95', d.get('hd', float('nan')))
            row[f'vol_{s}_ml_gt'] = d.get('vol_gt', float('nan'))
        row['dice_left'] = float(np.nanmean([row[f'dice_{s}'] for s in LEFT]))
        row['dice_all'] = float(np.nanmean([row[f'dice_{s}'] for s in STRUCTS]))
        row['dice_unified'] = row['dice_all']
        row.update(volumes_ml_by_struct(uni, (cfg.mm_per_voxel ** 3) / 1000.0))
        out.append(row)
        vols.append(uni)
        ins.append(placed)
    return out, vols, ins


def ef_row(pid: str, kind: str, arm: str, mode: str, vols_by_phase: Dict[str, List[float]], ref: dict) -> dict:
    edv = np.array(vols_by_phase['ED'])
    esv = np.array(vols_by_phase['ES'])
    n_pair = min(edv.size, esv.size)
    e, s_ = edv[:n_pair], esv[:n_pair]
    ok_ = e > 0
    ef = 100.0 * (e[ok_] - s_[ok_]) / e[ok_]

    row = {'kind': 'case', 'dataset': 'camus', 'model': kind, 'arm': arm,
           'case': pid, 'placement': mode,
           'edv_median': float(np.median(edv)), 'esv_median': float(np.median(esv))}
    if ef.size:
        row.update(ef_median=float(np.median(ef)), ef_lo=float(ef.min()),
                   ef_hi=float(ef.max()), ef_width=float(ef.max() - ef.min()))
    meta = ref.get(pid, {})
    row.update(quality_4ch=meta.get('quality_4ch', ''), quality_2ch=meta.get('quality_2ch', ''))
    ref_ef = meta.get('ef_ref')
    if ref_ef is not None and not np.isnan(ref_ef) and ef.size:
        row.update(ef_ref=float(ref_ef), ef_err=float(np.median(ef)) - float(ref_ef),
                   ef_covered=bool(ef.min() <= ref_ef <= ef.max()))
    return row


# ---------------------------------------------------------------------------
# argument checks
# ---------------------------------------------------------------------------
def check_args(args, pose, a2c: Optional[float]) -> None:
    canon = [p for p in args.placement if p in CANONICAL]

    if canon and a2c is None:
        raise SystemExit("canonical placement requires --canonical_a2c_angle")
    if 'canonical_posterior' in args.placement and pose is None:
        raise SystemExit("canonical_posterior needs --pose_ckpt")
    if 'canonical_posterior' in args.placement and max(args.levels) > 0:
        raise SystemExit("no levels allowed for canonical_posterior")
    if args.dataset == 'camus' and any(p in ('original', 'canonical_oracle') for p in args.placement):
        raise SystemExit("CAMUS has no 3D volume, so original_frames and z* do not exist")


# ---------------------------------------------------------------------------
# per-dataset loops
# ---------------------------------------------------------------------------
def whs_loop(fh, args, cfg, R, obs_spec) -> int:
    """Rows for the WHS synthetic path. Returns how many cases were saved as NIfTI."""
    n_written = 0
    paths = R.split.paths(args.subset)
    if args.limit:
        paths = paths[:args.limit]

    for i, p in enumerate(paths):
        cid = case_id(p)
        try:
            seg = resample_labels(nib.load(p).get_fdata().astype(np.uint8), list(cfg.grid_size), N_CLASSES, 'cpu')
            bundle = build_case(seg, cfg, cid, np.random.default_rng(args.seed * 7919 + i),
                                obs_spec=obs_spec, anchor=R.anchor)
        except (EmptyObservationError, DegenerateGeometryError, ValueError) as e:
            fh.write(json.dumps({'kind': 'skip', 'case': cid, 'error': f"{type(e).__name__}: {e}"}) + "\n")
            continue

        keep = n_written < R.keep_n
        for mode in args.placement:
            for lv in args.levels:
                spec = isotropic_spec(lv, lv / 2)
                ev = MulticlassEvaluator(spacing=(cfg.mm_per_voxel,) * 3)
                rng = np.random.default_rng(args.seed * 104729 + i)
                t0 = time.time()
                rows, vols, ins = whs_rows(
                    R.kind, R.model, R.pose, bundle, seg, mode, R.anchor, spec,
                    args.n_samples, args.temperature, rng, cfg, R.device, ev)
                common = {'dataset': 'whs', 'model': R.kind, 'arm': R.arm, 'case': cid, 'placement': mode, 
                          'level_deg': lv, 'temperature': args.temperature}
                for r in rows:
                    fh.write(json.dumps({
                        'kind': 'sample', **common, 'level_mm': lv / 2, 'n_samples': len(rows),
                        'seconds': round((time.time() - t0) / max(len(rows), 1), 3), **r}) + "\n")
                if len(vols) > 1:
                    per = np.array([mean_dice(v, seg) for v in vols])
                    mv = majority_vote(vols, N_CLASSES)
                    fh.write(json.dumps({
                        'kind': 'sample_set', **common, 'n_samples': len(vols),
                        'one_sample_dice': float(per.mean()),
                        'best_of_n': float(per.max()),
                        'majority_dice': mean_dice(mv, seg)}) + "\n")
                if keep and vols:
                    tag = f"{R.arm}_{mode}_L{int(lv)}"
                    save_nifti(vols[0], recon_path(args.save_nifti, cid, 'x', f"{tag}_recon"), cfg.mm_per_voxel)
                    save_nifti(ins[0], recon_path(args.save_nifti, cid, 'x', f"{tag}_input"), cfg.mm_per_voxel)
                    save_nifti(seg, recon_path(args.save_nifti, cid, 'x', "gt"), cfg.mm_per_voxel)
        if keep:
            n_written += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(paths)}", flush=True)
    return n_written


def camus_loop(fh, args, cfg, R) -> int:
    """Rows for the CAMUS real-data path: no 3D GT, so LV volume and EF only."""
    n_written = 0
    ref = read_reference_ef(args.ef_csv) if args.ef_csv and os.path.exists(args.ef_csv) else {}
    pats = list_patients(args.camus_dir)
    if args.limit:
        pats = pats[:args.limit]
    vox_ml = (cfg.mm_per_voxel ** 3) / 1000.0

    for i, pid in enumerate(pats):
        try:
            cases = {ph: load_case(args.camus_dir, pid, ph, cfg.mm_per_voxel,
                                   args.fallback_mm, args.camus_suffix) for ph in ('ED', 'ES')}
        except Exception as e:
            fh.write(json.dumps({'kind': 'skip', 'case': pid, 'error': f"{type(e).__name__}: {e}"}) + "\n")
            continue

        keep = n_written < R.keep_n
        for mode in args.placement:
            rng = np.random.default_rng(args.seed + i)     # one stream per patient
            vols_by_phase = {}
            for ph, views in cases.items():
                valid = np.array([not views[t].is_empty for t in TAGS], bool)
                zs = z_vectors(mode, views, cfg, None, R.pose, args.n_samples, args.temperature, rng)
                lv_vols = []
                for k in range(zs.shape[0]):
                    z = C.z_unpack(zs[k], cfg, valid)
                    placed, _ = C.place_at_z(views, z, cfg)
                    vol = (complete(R.model, placed, R.device) if R.kind == 'lcunet'
                           else implicit_reconstruct(R.model, views, C.frames_from_z(z, cfg), cfg, R.device, rng))
                    vml = volumes_ml_by_struct(vol, vox_ml)
                    fh.write(json.dumps({
                        'kind': 'sample', 'dataset': 'camus', 'model': R.kind,
                        'arm': R.arm, 'case': pid, 'phase': ph,
                        'placement': mode, 'level_deg': 0.0,
                        'space': (getattr(R.model, 'gauge_space', 'original') if R.kind == 'implicit' else 'canonical'),
                        'temperature': args.temperature, 'sample_idx': k,
                        'n_samples': zs.shape[0],
                        'z': zs[k, 6:12].round(4).tolist(), **vml}) + "\n")
                    lv_vols.append(vml['vol_LV_ml'])
                    if keep and k == 0:
                        save_nifti(vol, recon_path(args.save_nifti, pid, ph, f"{R.arm}_{mode}_recon"), cfg.mm_per_voxel)
                        save_nifti(placed, recon_path(args.save_nifti, pid, ph, f"{R.arm}_{mode}_input"), cfg.mm_per_voxel)
                vols_by_phase[ph] = lv_vols

            fh.write(json.dumps(ef_row(pid, R.kind, R.arm, mode, vols_by_phase, ref)) + "\n")
        if keep:
            n_written += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pats)}", flush=True)
    return n_written


# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="reconstruct and evaluate one model")
    add_split_args(ap)
    ap.add_argument('--dataset', required=True, choices=['whs', 'camus'])
    ap.add_argument('--model', required=True, help="lcunet:<ckpt> or implicit:<ckpt>")
    ap.add_argument('--pose_ckpt', default='')
    ap.add_argument('--placement', nargs='+', default=['original'], choices=list(PLACEMENTS))
    ap.add_argument('--levels', type=float, nargs='+', default=[0.0])
    ap.add_argument('--n_samples', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--subset', default='test', choices=['train', 'valid', 'test'])
    ap.add_argument('--obs_sigma_rot', type=float, default=8.0)
    ap.add_argument('--obs_sigma_trans', type=float, default=4.0)
    ap.add_argument('--camus_dir', default='../../data/camus/camus_pred/')
    ap.add_argument('--ef_csv', default='')
    ap.add_argument('--fallback_mm', type=float, default=0.308)
    ap.add_argument('--grid_size', type=int, nargs=3, default=[96, 96, 128])
    ap.add_argument('--voxel_size', type=float, default=2.0)
    ap.add_argument('--view_config', default='left')
    ap.add_argument('--anchor', default='')
    ap.add_argument('--canonical_a2c_angle', type=float, default=None)
    ap.add_argument('--canonical_apex', type=float, nargs=3, default=None)
    ap.add_argument('--n_iter_latent', type=int, default=100)
    ap.add_argument('--n_iter_total', type=int, default=1000)
    ap.add_argument('--fit_lr', type=float, default=1e-2)
    ap.add_argument('--pose_off', action='store_true')
    ap.add_argument('--save_nifti', default='')
    ap.add_argument('--save_nifti_n', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    ap.add_argument('--camus_suffix', default='_pred', choices=['_pred', '_gt'])
    return ap


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    kind, path = parse_model(args.model)

    split = None
    if args.dataset == 'whs':
        split = split_from_args(args)
        print(split.describe(), flush=True)

    pose, pose_ck = None, None
    if args.pose_ckpt:
        pose, pose_ck = build_pose_posterior(args.pose_ckpt, device)
    if kind == 'lcunet':
        net, ck = load_unet(path, device)
        if pose_ck is not None:
            check_agree(pose_ck, ck)
        model, arm = net, ck.get('arm', '?')
        train_sigma = [(ck.get('args') or {}).get('sigma_rot_deg'), (ck.get('args') or {}).get('sigma_trans_mm')]
        gauge = {k: ck.get(k) for k in ('anchor', 'canonical_a2c_angle_deg', 'canonical_apex')}
    else:
        model = load_implicit(path, device, args)
        arm = 'pose_off' if args.pose_off else 'pose_on'
        train_sigma = [None, None]
        gauge = {'anchor': None, 'canonical_a2c_angle_deg': None, 'canonical_apex': None}
        if pose is None and any(p in CANONICAL for p in args.placement):
            gauge.update(anchor=args.anchor or 'a4c')

    anchor = args.anchor or gauge.get('anchor') or 'a4c'
    a2c = (args.canonical_a2c_angle if args.canonical_a2c_angle is not None
           else gauge.get('canonical_a2c_angle_deg'))
    apex = args.canonical_apex or gauge.get('canonical_apex')
    check_args(args, pose, a2c)

    kw = {}
    if apex is not None:
        kw['canonical_apex'] = tuple(float(v) for v in apex)
    cfg = ViewConfig(grid_size=tuple(args.grid_size), mm_per_voxel=args.voxel_size,
                     view_config=args.view_config, canonical_a2c_angle_deg=a2c, **kw)

    meta = dict(kind='meta', dataset=args.dataset, model=kind, arm=arm, ckpt=path,
                pose_ckpt=args.pose_ckpt, anchor=anchor,
                canonical_a2c_angle_deg=a2c,
                canonical_apex=list(cfg.canonical_apex),
                grid_size=list(cfg.grid_size),
                voxel_size=cfg.mm_per_voxel, view_config=args.view_config,
                obs_sigma=[args.obs_sigma_rot, args.obs_sigma_trans],
                placements=args.placement, levels=args.levels,
                implicit_gauge_space=(getattr(model, 'gauge_space', None) if kind == 'implicit' else None),
                n_samples=args.n_samples, temperature=args.temperature,
                split_fingerprint=(split.fingerprint if split is not None else None),
                train_sigma=train_sigma,
                args=vars(args))
    print(json.dumps({k: v for k, v in meta.items() if k != 'args'}, indent=2))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fh = open(args.out, 'w', buffering=1)
    fh.write(json.dumps(meta) + "\n")

    R = SimpleNamespace(kind=kind, model=model, pose=pose, arm=arm, anchor=anchor,
                        split=split, device=device,
                        keep_n=(args.save_nifti_n if args.save_nifti else 0))

    if args.dataset == 'whs':
        obs_spec = isotropic_spec(args.obs_sigma_rot, args.obs_sigma_trans)
        n_written = whs_loop(fh, args, cfg, R, obs_spec)
    else:
        n_written = camus_loop(fh, args, cfg, R)

    fh.close()
    print(f"\n  jsonl -> {args.out}")
    if args.save_nifti:
        print(f"  NIfTI  -> {args.save_nifti}   ({n_written} cases)")


if __name__ == '__main__':
    main()