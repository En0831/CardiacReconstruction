# implicit/decode.py

from __future__ import annotations
 
from typing import Optional, Sequence
 
import numpy as np
import torch
 
from implicit.data import full_grid_coords
 
 
@torch.no_grad()
def decode_volume(net, latent: torch.Tensor, grid_size: Sequence[int],
                  voxel_size: float, device, chunk: int = 64 ** 3) -> np.ndarray:

    spacing = torch.full((3,), float(voxel_size), dtype=torch.float32)
    coords = full_grid_coords(list(grid_size), spacing)          # [X,Y,Z,3] mm
    flat = coords.reshape(-1, 3).to(device)
    lat = latent.detach().to(device)
 
    preds = torch.empty(flat.shape[0], dtype=torch.long)
    net.eval()
    for i in range(0, flat.shape[0], chunk):
        logits = net(lat[None], flat[i:i + chunk][None])          # [1,C,n]
        preds[i:i + chunk] = logits.argmax(1)[0].cpu()
    return preds.reshape(tuple(grid_size)).numpy().astype(np.uint8)
 
 
@torch.no_grad()
def decode_dice(net, latent: torch.Tensor, gt: torch.Tensor, voxel_size: float,
                device, chunk: int = 64 ** 3) -> list:

    pred = decode_volume(net, latent, gt.shape, voxel_size, device, chunk)
    gt_np = gt.cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)
    out = []
    for k in range(1, net.num_classes):
        p, g = (pred == k), (gt_np == k)
        denom = int(p.sum()) + int(g.sum())
        out.append(2.0 * float(np.logical_and(p, g).sum()) / denom
                   if denom > 0 else float('nan'))
    return out