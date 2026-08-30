# common/losses.py

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# ============================
# soft dice
# ============================

def soft_dice_multiclass(logits: torch.Tensor, target: torch.Tensor, num_classes: int,
                         include_background: bool = False, eps: float = 1e-3) -> torch.Tensor:
    """Mean soft Dice over foreground classes. logits [B,C,*ST], target [B,*ST] (long)."""
    probs = torch.softmax(logits, dim=1)
    target_1h = F.one_hot(target.long(), num_classes)          # [B, *ST, C]
    target_1h = target_1h.movedim(-1, 1).to(probs.dtype)       # [B, C, *ST]

    start = 0 if include_background else 1
    probs = probs[:, start:]
    target_1h = target_1h[:, start:]

    dims = tuple(range(2, probs.dim()))
    inter = (probs * target_1h).sum(dims)
    denom = probs.sum(dims) + target_1h.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)                 # [B, C']
    return dice.mean()

class CEWithDiceLoss(torch.nn.Module):
    """Multi-class replacement for the original `BCEWithDiceLoss`."""

    def __init__(self, num_classes: int = 6, ce_weight: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None,
                 include_background_in_dice: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.include_background_in_dice = include_background_in_dice
        self.register_buffer(
            'class_weights',
            class_weights if class_weights is not None else torch.ones(num_classes))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target.long(), weight=self.class_weights)
        dice = soft_dice_multiclass(logits, target, self.num_classes,
                                    self.include_background_in_dice)
        return self.ce_weight * ce + (1.0 - dice)


def masked_ce_dice(logits: torch.Tensor, labels: torch.Tensor, fg_mask: torch.Tensor, observed_classes: Sequence[int],
                   ce_weight: float = 1.0, eps: float = 1e-6, dice_eps: float = 1e-3) -> torch.Tensor:
    """CE + (1 - Dice) on the observed foreground only."""
    if not bool(fg_mask.any()):
        return logits.new_zeros(())

    probs = torch.softmax(logits, dim=1)        # [B, C, *ST]
    probs_pix = probs.movedim(1, -1)[fg_mask]   # [N, C]
    target = labels[fg_mask]                    # [N]

    # ---- CE over the observed foreground pixels ----
    ce = -torch.log(probs_pix.gather(1, target[:, None]).squeeze(1) + eps).mean()
 
    # ---- Dice over the observed classes, on those same pixels ----
    obs = sorted(int(c) for c in observed_classes)
    dices = []
    for c in obs:
        p_c = probs_pix[:, c]               # [N]
        g_c = (target == c).to(p_c.dtype)   # [N]
        inter = (p_c * g_c).sum()
        denom = p_c.sum() + g_c.sum()
        dices.append((2.0 * inter + dice_eps) / (denom + dice_eps))
    dice = torch.stack(dices).mean() if dices else probs.new_ones(())
 
    return ce_weight * ce + (1.0 - dice)