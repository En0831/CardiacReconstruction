# lcunet/train_pose.py

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.splits import add_split_args, split_from_args
from common.canonical import Z_NAMES
from common.calibration import per_dim_report, residual_correlation, correlated_pairs
from common.patches import LANDMARK_FIELDS
from lcunet.data import build_pose_dataset, collate_skip
from lcunet.pose_net import build_pose_net, Z_SLICE

Z_SUP_NAMES = list(Z_NAMES)[Z_SLICE]          # the six supervised A2C entries


@torch.no_grad()
def fit_standardisation(net, loader, device, max_batches: int = 0):
    """get the mean and sd of the training set"""
    Z, L = [], []
    for i, b in enumerate(loader):
        if b is None:
            continue
        m = b['z_mask'][:, Z_SLICE].all(dim=1)
        if m.any():
            Z.append(b['z'][m][:, Z_SLICE].numpy())
            L.append(b['landmarks'][m].numpy())
        if max_batches and i + 1 >= max_batches:
            break
    if not Z:
        raise SystemExit("no supervised samples while fitting standardisation")
    Z = np.concatenate(Z); L = np.concatenate(L)
    net.set_standardisation(Z.mean(0), Z.std(0, ddof=1), L.mean(0), L.std(0, ddof=1))
    return Z, L


@torch.no_grad()
def evaluate(net, loader, device) -> dict:
    net.eval()
    Zs, MUs, SIGs, NLLs = [], [], [], []
    for b in loader:
        if b is None:
            continue
        keep = b['z_mask'][:, Z_SLICE].all(dim=1)
        if not keep.any():
            continue
        patch = b['patch'][keep].to(device)
        valid = b['valid'][keep].to(device)
        lm = b['landmarks'][keep].to(device) if net.n_landmarks else None
        z = b['z'][keep][:, Z_SLICE].to(device)
        mu, L = net(patch, valid, lm)
        NLLs.append(net.nll(mu, L, net.standardise_z(z)).item() * int(keep.sum()))
        Zs.append(z.cpu().numpy())
        MUs.append(net.unstandardise_z(mu).cpu().numpy())
        SIGs.append(net.sigma(L).cpu().numpy())
    if not Zs:
        return {}
    Z = np.concatenate(Zs); MU = np.concatenate(MUs); SIG = np.concatenate(SIGs)
    rep = per_dim_report(Z, MU, SIG, Z_SUP_NAMES)
    C = residual_correlation(Z, MU, SIG)
    return {'n': int(Z.shape[0]), 'per_dim': rep,
            'nll': float(np.sum(NLLs) / max(Z.shape[0], 1)),
            'correlation': C.tolist(),
            'pairs': [(a, b_, float(v)) for a, b_, v in correlated_pairs(C, Z_SUP_NAMES)],
            'prior_sd': Z.std(0, ddof=1).tolist(),
            'sigma_mean': SIG.mean(0).tolist()}


def print_report(rep: dict, epoch: int, n_epoch: int, train_nll: float) -> None:
    if not rep:
        print(f"[{epoch}/{n_epoch}] train NLL {train_nll:.4f} | no valid samples",
              flush=True)
        return
    print(f"\n[{epoch}/{n_epoch}] train NLL {train_nll:.4f}   "
          f"valid NLL {rep['nll']:.4f}   n {rep['n']}")
    print(f"  {'dof':<24}{'sigma':>9}{'prior':>9}{'ratio':>8}"
          f"{'r.std':>8}{'c68':>7}{'c95':>7}{'kurt':>8}  flags")
    for i, name in enumerate(Z_SUP_NAMES):
        d = rep['per_dim'][name]
        sig, pri = rep['sigma_mean'][i], rep['prior_sd'][i]
        flags = []
        if d['flag_overconfident']:
            flags.append('OVERCONF')
        if d['flag_underconfident']:
            flags.append('underconf')
        if d['flag_bimodal']:
            flags.append('BIMODAL')
        if pri > 0 and sig / pri > 0.95:
            flags.append('=prior')
        print(f"  {name:<24}{sig:>9.3f}{pri:>9.3f}{sig/max(pri,1e-9):>8.2f}"
              f"{d['std']:>8.2f}{d['cover68']:>7.2f}{d['cover95']:>7.2f}"
              f"{d['excess_kurtosis']:>8.2f}  {' '.join(flags)}")
    if rep['pairs']:
        print("  residual correlations above 0.3 (a diagonal head cannot fit these):")
        for a, b_, v in rep['pairs']:
            print(f"    {a:<24} {b_:<24} {v:+.2f}")
    print("  ratio ~1 means the observation carried no information for that DoF."
          "\n  At sigma 8/4 the translations are 93-99% injected noise, so that is"
          "\n  the correct answer there, not a failure.", flush=True)


def main():
    ap = argparse.ArgumentParser(description="train the residual posterior (部品2)")
    add_split_args(ap)
    ap.add_argument('--canonical_a2c_angle', type=float, required=True)
    ap.add_argument('--canonical_apex', type=float, nargs=3, default=None)
    ap.add_argument('--anchor', default='a4c', choices=['anatomical', 'a4c'])
    ap.add_argument('--sigma_rot_deg', type=float, default=8.0)
    ap.add_argument('--sigma_trans_mm', type=float, default=4.0)
    ap.add_argument('--view_config', default='left')
    ap.add_argument('--grid_size', type=int, nargs=3, default=[96, 96, 128])
    ap.add_argument('--voxel_size', type=float, default=2.0)
    ap.add_argument('--patch_size', type=int, nargs=2, default=[192, 192])
    ap.add_argument('--patch_apex_rc', type=float, nargs=2, default=[8.0, 96.0])
    ap.add_argument('--patch_mm_per_px', type=float, default=1.0)
    ap.add_argument('--cov', default='full', choices=['full', 'diag'])
    ap.add_argument('--use_landmarks', action='store_true')
    ap.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128, 256])
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--min_sigma', type=float, default=0.05)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--n_epoch', type=int, default=200)
    ap.add_argument('--val_every', type=int, default=5)
    ap.add_argument('--ckpt_every', type=int, default=25)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--tag', default='pose')
    ap.add_argument('--ckpt_dir', default='./ckpts/pose')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    if args.anchor != 'a4c':
        ap.error(f"--anchor {args.anchor}: Use --anchor a4c")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    split = split_from_args(args)
    print(split.describe(), flush=True)

    common = dict(grid_size=args.grid_size, voxel_size=args.voxel_size,
                  sigma_rot_deg=args.sigma_rot_deg,
                  sigma_trans_mm=args.sigma_trans_mm,
                  view_config=args.view_config,
                  canonical_a2c_angle_deg=args.canonical_a2c_angle,
                  canonical_apex=args.canonical_apex,
                  anchor=args.anchor,
                  patch_size=args.patch_size, 
                  patch_apex_rc=args.patch_apex_rc,
                  patch_mm_per_px=args.patch_mm_per_px)
    trainset = build_pose_dataset(split, 'train', **common)
    validset = build_pose_dataset(split, 'valid', **common)
    tl = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                    drop_last=True, num_workers=args.num_workers,
                    collate_fn=collate_skip)
    vl = DataLoader(validset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_skip)
    cfg = trainset.base.cfg
    print(f"anchor {args.anchor} | apex {tuple(cfg.canonical_apex)} | a2c "
          f"{cfg.canonical_a2c_angle_deg} | sigma {args.sigma_rot_deg}/"
          f"{args.sigma_trans_mm} | cov {args.cov} | patch {tuple(args.patch_size)}")
    print(f"{len(trainset)} train / {len(validset)} valid", flush=True)

    n_lm = len(LANDMARK_FIELDS) * 2 if args.use_landmarks else 0
    net = build_pose_net(cov=args.cov, n_landmarks=n_lm, dims=args.dims,
                         hidden=args.hidden, min_sigma=args.min_sigma).to(device)
    print(f"pose net: {sum(p.numel() for p in net.parameters()):,} params, "
          f"{n_lm} landmark dims", flush=True)

    print("\nfitting standardisation on one pass over the training set...", flush=True)
    Z0, _ = fit_standardisation(net, tl, device)
    print(f"  n {Z0.shape[0]}")
    for i, name in enumerate(Z_SUP_NAMES):
        print(f"    {name:<24} mean {Z0[:, i].mean():8.3f}  sd {Z0[:, i].std(ddof=1):8.3f}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_epoch)

    history = []
    for epoch in range(args.n_epoch + 1):
        net.train()
        losses = []
        for b in tl:
            if b is None:
                continue
            keep = b['z_mask'][:, Z_SLICE].all(dim=1)
            if not keep.any():
                continue
            patch = b['patch'][keep].to(device)
            valid = b['valid'][keep].to(device)
            lm = b['landmarks'][keep].to(device) if n_lm else None
            z = b['z'][keep][:, Z_SLICE].to(device)
            mu, L = net(patch, valid, lm)
            loss = net.nll(mu, L, net.standardise_z(z))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        train_nll = float(np.mean(losses)) if losses else float('nan')

        if epoch % args.val_every == 0:
            rep = evaluate(net, vl, device)
            print_report(rep, epoch, args.n_epoch, train_nll)
            history.append({'epoch': epoch, 'train_nll': train_nll, **rep})

        if epoch % args.ckpt_every == 0:
            torch.save({'net': net.state_dict(), 'optimizer': opt.state_dict(),
                        'epoch': epoch, 'cov': args.cov, 'n_landmarks': n_lm,
                        'dims': list(args.dims), 'hidden': args.hidden,
                        'min_sigma': args.min_sigma,
                        'anchor': args.anchor,
                        'canonical_apex': list(cfg.canonical_apex),
                        'canonical_a2c_angle_deg': cfg.canonical_a2c_angle_deg,
                        'patch_size': list(args.patch_size),
                        'patch_apex_rc': list(args.patch_apex_rc),
                        'patch_mm_per_px': args.patch_mm_per_px,
                        'grid_size': list(args.grid_size),
                        'voxel_size': args.voxel_size,
                        'split_fingerprint': split.fingerprint,
                        'args': vars(args)},
                       os.path.join(args.ckpt_dir, f'pose_{args.tag}_{epoch}.pth'))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump({'history': history, 'args': vars(args),
                       'z_names': Z_SUP_NAMES}, f, indent=2, default=float)
        print(f"\n  json -> {args.json}")


if __name__ == '__main__':
    main()