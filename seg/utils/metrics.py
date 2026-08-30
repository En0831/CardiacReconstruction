# utils/metrics.py

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


# ----------------------------------------------------------------------
# Surface-distance metrics (ASSD, HD95)
# ----------------------------------------------------------------------
def _surface_distances(pred, gt, spacing=(1.0, 1.0)):
    """
    Symmetric surface-to-surface distances between two binary masks.

    Args:
        pred: numpy bool [H, W]
        gt:   numpy bool [H, W]

    Returns:
        d_pred_to_gt: distances from each pred-boundary pixel to GT boundary
        d_gt_to_pred: distances from each GT-boundary pixel to pred boundary
    """
    pred_boundary = pred ^ binary_erosion(pred)
    gt_boundary = gt ^ binary_erosion(gt)

    # Distance from every pixel to the nearest boundary pixel of the other mask
    dt_gt = distance_transform_edt(~gt_boundary, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_boundary, sampling=spacing)

    d_pred_to_gt = dt_gt[pred_boundary]
    d_gt_to_pred = dt_pred[gt_boundary]

    return d_pred_to_gt, d_gt_to_pred


def assd_hd95_single(pred, gt, spacing=(1.0, 1.0)):
    """
    ASSD and HD95 for ONE binary mask pair.

    Args:
        pred: numpy bool [H, W]
        gt:   numpy bool [H, W]

    Returns:
        (assd, hd95). NaN if either mask is empty (metric undefined).
    """
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan"), float("nan")

    d_pg, d_gp = _surface_distances(pred, gt, spacing=spacing)

    if len(d_pg) == 0 or len(d_gp) == 0:
        return float("nan"), float("nan")

    assd = (d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp))
    hd95 = max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))

    return float(assd), float(hd95)


@torch.no_grad()
def full_metrics_per_class(logits, targets, num_classes=4, spacing=(1.0, 1.0)):
    """
    Per-image Dice / ASSD / HD95 for each foreground class.

    Args:
        logits:  Tensor [B, C, H, W]
        targets: Tensor [B, H, W]
        spacing: pixel spacing (dy, dx). (1, 1) -> distances in pixels.
    """
    preds = torch.argmax(logits, dim=1)  # [B, H, W]

    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()

    out = {
        "dice": {cls: [] for cls in range(1, num_classes)},
        "assd": {cls: [] for cls in range(1, num_classes)},
        "hd95": {cls: [] for cls in range(1, num_classes)},
    }

    batch_size = preds_np.shape[0]

    for b in range(batch_size):
        for cls in range(1, num_classes):
            pred_cls = preds_np[b] == cls
            gt_cls = targets_np[b] == cls

            # Dice (per image)
            inter = np.logical_and(pred_cls, gt_cls).sum()
            denom = pred_cls.sum() + gt_cls.sum()

            if denom == 0:
                dice = float("nan")  # class absent in both -> undefined
            else:
                dice = 2.0 * inter / denom

            out["dice"][cls].append(float(dice))

            # ASSD / HD95 (per image)
            assd, hd95 = assd_hd95_single(pred_cls, gt_cls, spacing=spacing)
            out["assd"][cls].append(assd)
            out["hd95"][cls].append(hd95)

    return out