# implicit/train.py
 
from __future__ import annotations
 
import argparse
import math
import os
 
import numpy as np
import torch
 
from common.losses import CEWithDiceLoss
from common.splits import split_from_args, add_split_args
from common.views import ViewConfig
from implicit.data import build_prior_trainset
from implicit.implicits_echo import MultiClassAutoDecoder
from implicit.decode import decode_dice
 
 
 
def evaluate_dense(net, latent, dataset, item, device, chunk=64 ** 3):
    """Per-class Dice of one training latent, via the shared decoder."""
    gt, _ = dataset.full_volume(item)
    voxel_size = float(dataset.spacing[0])
    return decode_dice(net, latent, gt, voxel_size, device, chunk)
 
 
def main():
    ap = argparse.ArgumentParser(description="6-class implicit auto-decoder shape prior")
    add_split_args(ap) 
    ap.add_argument('--tag', default='echo6')
    ap.add_argument('--ckpt_dir', default='./ckpts/')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--num_classes', type=int, default=6)
    ap.add_argument('--latent_dim', type=int, default=128)
    ap.add_argument('--op_num_layers', type=int, default=8)
    ap.add_argument('--op_coord_layers', type=int, nargs='+', default=[0, 4])
    ap.add_argument('--num_points', type=int, default=32 ** 3)
    ap.add_argument('--fg_fraction', type=float, default=0.5)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_lat', type=float, default=1e-3)
    ap.add_argument('--lat_reg_lambda', type=float, default=1e-4)
    ap.add_argument('--n_epoch', type=int, default=4000)
    ap.add_argument('--log_every', type=int, default=20)
    ap.add_argument('--ckpt_every', type=int, default=200)
    ap.add_argument('--voxel_size', type=float, default=2.0)
    ap.add_argument('--grid_size', type=int, nargs=3, default=[96, 96, 128])
    ap.add_argument('--resume', default='', type=str)
    ap.add_argument('--canonical', action='store_true')
    ap.add_argument('--canonical_a2c_angle', type=float, default=-76.6)
    ap.add_argument('--canonical_apex', type=float, nargs=3, default=[35.0, 44.0, 18.0])
    args = ap.parse_args()
 
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
 
    # ---- shared split ----
    split = split_from_args(args)
    print(split.describe(), flush=True)
    canonical_cfg = None
    if args.canonical:
        canonical_cfg = ViewConfig(
            grid_size=tuple(args.grid_size),
            mm_per_voxel=args.voxel_size,
            canonical_a2c_angle_deg=args.canonical_a2c_angle,
            canonical_apex=tuple(args.canonical_apex)
        )
    trainset = build_prior_trainset(
        split, num_points=args.num_points, num_classes=args.num_classes,
        grid_size=args.grid_size, voxel_size=args.voxel_size,
        fg_fraction=args.fg_fraction, canonical_cfg=canonical_cfg)
    loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"{len(trainset)} training volumes, image_size={trainset.image_size.tolist()} mm",
          flush=True)
 
    # ---- model + external latent table ----
    net = MultiClassAutoDecoder(
        lat_dim=args.latent_dim, spatial_dim=3,
        image_size=trainset.image_size.clone(),
        occnet_num_layers=args.op_num_layers,
        occnet_layers_with_coords=args.op_coord_layers,
        num_classes=args.num_classes).to(device)
 
    latents = torch.nn.Parameter(
        torch.normal(0.0, 1.0 / math.sqrt(args.latent_dim),
                     [len(trainset), args.latent_dim], device=device))
 
    optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': args.lr},
        {'params': latents, 'lr': args.lr_lat},
    ])
    criterion = CEWithDiceLoss(num_classes=args.num_classes).to(device)
 
    # ---- optional resume ----
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # the latent rows are pinned to a split; refuse to resume across a drift
        if 'split_fingerprint' in ckpt:
            split.assert_matches(ckpt['split_fingerprint'], 'checkpoint')
        net.load_state_dict(ckpt['net'])
        saved_lat = ckpt['latents_train'].to(device)
        if saved_lat.shape != latents.shape:
            raise ValueError(f"latent table shape mismatch: checkpoint {saved_lat.shape} vs current {latents.shape}")
        with torch.no_grad():
            latents.copy_(saved_lat)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        else:
            print("  (no optimizer state; Adam moments restart from zero)", flush=True)
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)
 
    for epoch in range(start_epoch, args.n_epoch + 1):
        net.train()
        running = []
        for batch in loader:
            coords = batch['coords'].to(device)
            labels = batch['labels'].to(device)
            lat = latents[batch['caseids']]
 
            optimizer.zero_grad()
            logits = net(lat, coords)
            loss = criterion(logits, labels)
            if args.lat_reg_lambda > 0:                   # DeepSDF ramp over first 100 epochs
                reg = torch.mean(torch.sum(lat ** 2, dim=1))
                loss = loss + min(1.0, epoch / 100.0) * args.lat_reg_lambda * reg
            loss.backward()
            optimizer.step()
            running.append(loss.item())
 
        if epoch % args.log_every == 0:
            d = evaluate_dense(net, latents[0].detach(), trainset, 0, device)
            names = ['LV', 'MY', 'RV', 'LA', 'RA'][:args.num_classes - 1]
            msg = "  ".join(f"{n}:{v:.3f}" for n, v in zip(names, d))
            print(f"[{epoch}/{args.n_epoch}] loss {np.mean(running):.4f} | case0 {msg}",
                  flush=True)
 
        if epoch % args.ckpt_every == 0:
            torch.save({'net': net.state_dict(),
                        'latents_train': latents.detach().cpu(),
                        'gauge_space': 'canonical' if args.canonical else 'original',
                        'gauge_anchor': 'anatomical' if args.canonical else None,
                        'canonical_apex': list(args.canonical_apex) if args.canonical else None,
                        'canonical_a2c_angle_deg': args.canonical_a2c_angle if args.canonical else None,
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'image_size': trainset.image_size,
                        'split_fingerprint': split.fingerprint,
                        'args': vars(args)},
                       os.path.join(args.ckpt_dir, f'ad_mc_{args.tag}_{epoch}.pth'))
 
 
if __name__ == '__main__':
    main()