# utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-5, include_background=False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, C, H, W]
            targets: [B, H, W]
        """
        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(
            targets.long(),
            num_classes=self.num_classes,
        )  # [B, H, W, C]

        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        if self.include_background:
            class_start = 0
        else:
            class_start = 1

        probs = probs[:, class_start:, :, :]
        targets_one_hot = targets_one_hot[:, class_start:, :, :]

        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        union = torch.sum(probs, dim=dims) + torch.sum(targets_one_hot, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        loss = 1.0 - dice.mean()
        return loss


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes=4, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes, include_background=False)

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets.long())
        dice_loss = self.dice(logits, targets)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss