# train.py

"""Train segmentation model on CAMUS dataset"""

import argparse
from pathlib import Path
import random
import json
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model_factory import build_model
from utils.dataset import CAMUSDataset
from utils.losses import CEDiceLoss
from utils.metrics import full_metrics_per_class
from utils.split import load_split


CLASS_NAMES = {1: "lv", 2: "myo", 3: "la"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)

    for images, masks in pbar:
        images = images.to(device)           # [B, 1, H, W]
        masks = masks.to(device).long()      # [B, H, W]

        logits = model(images)               # [B, 4, H, W]
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=4):
    model.eval()

    running_loss = 0.0

    acc = {
        "dice": {cls: [] for cls in CLASS_NAMES},
        "assd": {cls: [] for cls in CLASS_NAMES},
        "hd95": {cls: [] for cls in CLASS_NAMES},
    }

    pbar = tqdm(loader, desc="Valid", leave=False)

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device).long()

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)

        batch_metrics = full_metrics_per_class(
            logits=logits,
            targets=masks,
            num_classes=num_classes,
        )

        for metric in ["dice", "assd", "hd95"]:
            for cls in CLASS_NAMES:
                acc[metric][cls].extend(batch_metrics[metric][cls])

    valid_loss = running_loss / len(loader.dataset)

    metrics = {"valid_loss": valid_loss}

    for metric in ["dice", "assd", "hd95"]:
        class_means = []

        for cls, name in CLASS_NAMES.items():
            value = float(np.nanmean(acc[metric][cls]))
            metrics[f"{metric}_{name}"] = value
            class_means.append(value)

        metrics[f"{metric}_mean"] = float(np.mean(class_means))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model on CAMUS ED/ES frames")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./ckpts/camus_unet")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument( "--model", type=str, default="unet", choices=["unet", "resunet", "attention_unet"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--rotation_deg", type=float, default=15.0)
    parser.add_argument("--translate_frac", type=float, default=0.10)
    parser.add_argument("--scale_min", type=float, default=0.9)
    parser.add_argument("--scale_max", type=float, default=1.1)
    parser.add_argument("--use_sequence_frame", action="store_true") 

    args = parser.parse_args()

    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # -------------------------
    # Load fixed split from JSON
    # -------------------------
    split_info = load_split(args.split_json)

    train_ids = split_info["train"]
    valid_ids = split_info["valid"]
    test_ids = split_info["test"]

    # Keep a copy of the exact split used for this run next to the checkpoints
    shutil.copy(args.split_json, save_dir / "split.json")

    print(f"Split JSON: {args.split_json}")
    print(f"Train patients: {len(train_ids)}")
    print(f"Valid patients: {len(valid_ids)}")
    print(f"Test patients:  {len(test_ids)} (not used during training)")

    # -------------------------
    # Datasets
    # -------------------------
    image_size = (args.image_size, args.image_size)

    train_set = CAMUSDataset(
        data_root=args.data_root,
        patient_ids=train_ids,
        image_size=image_size,
        augment=not args.no_augment,
        rotation_deg=args.rotation_deg,
        translate_frac=args.translate_frac,
        scale_range=(args.scale_min, args.scale_max),
        use_sequence_frame=args.use_sequence_frame,
    )

    # Validation set: NO augmentation
    valid_set = CAMUSDataset(
        data_root=args.data_root,
        patient_ids=valid_ids,
        image_size=image_size,
        augment=False,
    )

    print(f"Augmentation: {'OFF' if args.no_augment else 'ON (train only)'}")

    print(f"Train samples: {len(train_set)}")
    print(f"Valid samples: {len(valid_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -------------------------
    # Model, loss, optimizer
    # -------------------------
    model = build_model(
        model_name=args.model,
        in_channels=1,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
    ).to(device)

    criterion = CEDiceLoss(num_classes=args.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dice = -1.0
    best_epoch = -1
    best_metrics = None
    best_record = None

    history = []

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        valid_metrics = validate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            num_classes=args.num_classes,
        )

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            **valid_metrics,
        }

        history.append(log)

        print(
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_metrics['valid_loss']:.4f} | "
            f"Dice mean={valid_metrics['dice_mean']:.4f} "
            f"(LV={valid_metrics['dice_lv']:.4f} "
            f"MYO={valid_metrics['dice_myo']:.4f} "
            f"LA={valid_metrics['dice_la']:.4f}) | "
            f"ASSD mean={valid_metrics['assd_mean']:.2f}px | "
            f"HD95 mean={valid_metrics['hd95_mean']:.2f}px"
        )

        # Save latest
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            },
            save_dir / "latest.pt",
        )

        # Save best
        if valid_metrics["dice_mean"] > best_dice:
            best_dice = valid_metrics["dice_mean"]
            best_epoch = epoch
            best_metrics = dict(valid_metrics)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "best_metrics": best_metrics,
                    "args": vars(args),
                },
                save_dir / "best.pt",
            )

            print(f"Saved best model: Dice mean={best_dice:.4f} (epoch {epoch})")

        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # -------------------------
    # Final summary
    # -------------------------
    best_summary = {
        "best_epoch": best_epoch,
        "selected_by": "dice_mean",
        "metrics": best_metrics,
        "note": "ASSD/HD95 are in PIXEL units at the resized resolution "
                f"({args.image_size}x{args.image_size}), not mm.",
    }

    history.append({"best_summary": best_summary})

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(save_dir / "best_val_metrics.json", "w") as f:
        json.dump(best_summary, f, indent=2)

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"  Dice  mean={best_metrics['dice_mean']:.4f} | "
          f"LV={best_metrics['dice_lv']:.4f} "
          f"MYO={best_metrics['dice_myo']:.4f} "
          f"LA={best_metrics['dice_la']:.4f}")
    print(f"  ASSD  mean={best_metrics['assd_mean']:.2f}px | "
          f"LV={best_metrics['assd_lv']:.2f} "
          f"MYO={best_metrics['assd_myo']:.2f} "
          f"LA={best_metrics['assd_la']:.2f}")
    print(f"  HD95  mean={best_metrics['hd95_mean']:.2f}px | "
          f"LV={best_metrics['hd95_lv']:.2f} "
          f"MYO={best_metrics['hd95_myo']:.2f} "
          f"LA={best_metrics['hd95_la']:.2f}")


if __name__ == "__main__":
    main()