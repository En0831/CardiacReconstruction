# lcunet/train.py

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.losses import CEWithDiceLoss
from common.metrics import MulticlassEvaluator
from common.splits import add_split_args, split_from_args
from common.views import EmptyObservationError
from lcunet.data import build_lcunet_dataset, N_CLASSES
from lcunet.unet import UNet
from lcunet.infer import to_onehot, complete


def validate(net, dataset, device, spacing, hd_percentile=95.0):
    ev = MulticlassEvaluator(spacing=spacing, hd_percentile=hd_percentile)
    net.eval()
    excluded = []
    for i in range(len(dataset)):
        try:
            inp, tgt = dataset[i]
        except EmptyObservationError:
            excluded.append(dataset.casenames[i])
            continue
        pred = complete(net, inp, device)
        ev.add(pred, tgt.numpy(), case_id=os.path.basename(dataset.casenames[i]))
    return ev, excluded


def main():
    ap = argparse.ArgumentParser(description="LC-U-Net label completion, shared pipeline")
    add_split_args(ap)
    ap.add_argument('--arm', choices=['A0', 'A1', 'A2', 'C', 'A2c'], default='A0')
    ap.add_argument('--sigma_rot_deg', type=float, default=15.0)
    ap.add_argument('--sigma_trans_mm', type=float, default=8.0)
    ap.add_argument('--view_config', default='left', choices=['left', 'rv', 'whole'])
    ap.add_argument('--canonical_a2c_angle', type=float, default=None)
    ap.add_argument('--canonical_apex', type=float, nargs=3, default=None)
    ap.add_argument('--anchor', choices=['anatomical', 'a4c'], default='anatomical')
    ap.add_argument('--tag', default='lcunet')
    ap.add_argument('--ckpt_dir', default='./ckpts/')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dim_hid', type=int, nargs='+', default=[32, 64, 128, 256, 256])
    ap.add_argument('--drop_rate', type=float, default=0.0)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--n_epoch', type=int, default=500)
    ap.add_argument('--val_every', type=int, default=10)
    ap.add_argument('--ckpt_every', type=int, default=50)
    ap.add_argument('--grid_size', type=int, nargs=3, default=[96, 96, 128])
    ap.add_argument('--voxel_size', type=float, default=2.0)
    ap.add_argument('--resume', default='', type=str)
    ap.add_argument('--num_workers', type=int, default=2)
    args = ap.parse_args()
    if args.arm in ('C', 'A2c') and args.canonical_a2c_angle is None:
        ap.error("--canonical_a2c_angle is required for arm C and A2c")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    spacing = (args.voxel_size,) * 3    #(2, 2, 2)

    # load the split and print a summary
    split = split_from_args(args)
    print(split.describe(), flush=True)

    # build the datasets and dataloader
    common = dict(grid_size=args.grid_size, voxel_size=args.voxel_size,
                  sigma_rot_deg=args.sigma_rot_deg, sigma_trans_mm=args.sigma_trans_mm,
                  view_config=args.view_config, anchor=args.anchor,
                  canonical_a2c_angle_deg=args.canonical_a2c_angle,
                  canonical_apex=args.canonical_apex)
    trainset = build_lcunet_dataset(split, 'train', arm=args.arm, **common)
    validset = build_lcunet_dataset(split, 'valid', arm=args.arm, **common)
    loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    print(f"arm {args.arm}: {len(trainset)} train / {len(validset)} valid volumes", flush=True)

    net = UNet(dim_in=N_CLASSES, dim_hid=args.dim_hid, dim_out=N_CLASSES, drop_rate=args.drop_rate).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = CEWithDiceLoss(num_classes=N_CLASSES).to(device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if 'split_fingerprint' in ckpt:
            split.assert_matches(ckpt['split_fingerprint'], 'checkpoint')
        net.load_state_dict(ckpt['net'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)


    for epoch in range(start_epoch, args.n_epoch + 1):
        net.train()
        running = []
        for inp, tgt in loader:
            x = to_onehot(inp).to(device)                 # [B,6,X,Y,Z] input echo
            target = tgt.long().to(device)                # [B,X,Y,Z] dense target
            optimizer.zero_grad()
            logits = net(x)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            running.append(loss.item())

        if epoch % args.val_every == 0:
            ev, excluded = validate(net, validset, device, spacing)
            s = ev.summary(verbose=False)
            msg = (f"[{epoch}/{args.n_epoch}] loss {np.mean(running):.4f} | "
                   f"val Dice left {s['_left']['dice_case_mean']:.4f} "
                   f"all {s['_all']['dice_case_mean']:.4f}")
            if excluded:
                msg += f" | {len(excluded)} val cases empty at {args.arm}"
            print(msg, flush=True)

        if epoch % args.ckpt_every == 0:
            torch.save({'net': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'arm': args.arm,
                        'view_config': args.view_config,
                        'anchor': args.anchor,
                        'canonical_a2c_angle_deg': args.canonical_a2c_angle,
                        'canonical_apex': list(trainset.cfg.canonical_apex),
                        'split_fingerprint': split.fingerprint,
                        'args': vars(args)},
                       os.path.join(args.ckpt_dir, f'lcunet_{args.tag}_{epoch}.pth'))


if __name__ == '__main__':
    main()