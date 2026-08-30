# lcunet/infer.py

from __future__ import annotations
 
from typing import Tuple
 
import numpy as np
import torch
 
from common.views import N_CLASSES
from lcunet.unet import UNet
 
DEFAULT_DIM_HID = [32, 64, 128, 256, 256]

def to_onehot(vol: torch.Tensor) -> torch.Tensor:
    oh = torch.nn.functional.one_hot(vol.long(), N_CLASSES)
    return oh.permute(0, 4, 1, 2, 3).float()


def load_unet(path: str, device) -> Tuple[UNet, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck.get('args') or {}
    net = UNet(dim_in=N_CLASSES, dim_out=N_CLASSES,
               dim_hid=a.get('dim_hid', DEFAULT_DIM_HID),
               drop_rate=0.0).to(device)
    net.load_state_dict(ck['net'])
    net.eval()
    return net, ck


@torch.no_grad()
def complete(net, vol, device) -> np.ndarray:
    x = torch.as_tensor(vol, dtype=torch.int64)[None]
    logits = net(to_onehot(x).to(device))
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)