# infer.py

"""Inference for CAMUS 2D segmentation"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.model_factory import build_model
from utils.metrics import assd_hd95_single
from utils.split import load_split

CLASS_NAMES = {1: "LV", 2: "MYO", 3: "LA"}
QUALITY_ORDER = ["Good", "Medium", "Poor", "Unknown"]


def load_nifti(path):
    nii = nib.load(str(path))
    arr = nii.get_fdata()
    arr = np.asarray(arr)
    return arr, nii.affine, nii.header


def normalize_image(image):
    """
    Robust normalization for ultrasound image.
    image: numpy array [H, W]
    """
    image = image.astype(np.float32)
    p1, p99 = np.percentile(image, (1, 99))
    image = np.clip(image, p1, p99)
    mean = image.mean()
    std = image.std()
    if std < 1e-8:
        image = image - mean
    else:
        image = (image - mean) / std
    return image


def prepare_frame(frame, image_size):
    """
    Convert one 2D frame to model input.
    Args:
        frame: numpy [H, W]
        image_size: tuple, e.g. (256, 256)
    Returns:
        x: torch.Tensor [1, 1, image_size[0], image_size[1]]
    """
    frame = normalize_image(frame)

    x = torch.from_numpy(frame).float()
    x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    x = F.interpolate(
        x,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )
    return x


@torch.no_grad()
def predict_frame(model, frame, device, image_size, original_size):
    """
    Predict segmentation for one 2D frame.
    Args:
        frame: numpy [H, W]
        original_size: original image size, e.g. (H, W)

    Returns:
        pred_original: numpy [H, W], values 0,1,2,3
    """
    x = prepare_frame(frame, image_size)
    x = x.to(device)

    logits = model(x)  # [1, C, h, w]

    # Resize logits back to original image size before argmax
    logits = F.interpolate(
        logits,
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )

    pred = torch.argmax(logits, dim=1)  # [1, H, W]
    pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)

    return pred


def save_mask_nifti(mask, affine, header, out_dir, save_name):
    """
    Save mask as .nii.gz.
    """
    save_path = out_dir / f"{save_name}.nii.gz"
    mask = mask.astype(np.uint8)
    new_header = header.copy()
    new_header.set_data_dtype(np.uint8)
    nii = nib.Nifti1Image(mask, affine=affine, header=new_header)
    nib.save(nii, str(save_path))


def squeeze_2d(image, path):
    """
    Ensure a loaded ED/ES image or mask is 2D [H, W].
    """
    if image.ndim == 3:
        if image.shape[-1] == 1:
            image = image[:, :, 0]
        else:
            raise ValueError(f"Expected 2D image, got {image.shape}: {path}")

    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got {image.shape}: {path}")

    return image


# ----------------------------------------------------------------------
# Image quality
# ----------------------------------------------------------------------
def load_quality_table(csv_path):
    """
    ImageQuality per (patient, view) from camus_ef.csv.

    columns: patient, ImageQuality_2CH, ImageQuality_4CH.

    Returns:
        {patient_id: {"2CH": "Good", "4CH": "Medium"}}
    """
    table = {}

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            patient_id = row.get("patient", "").strip()

            if not patient_id:
                continue

            if not patient_id.startswith("patient"):
                try:
                    patient_id = f"patient{int(patient_id):04d}"
                except ValueError:
                    continue

            table[patient_id] = {
                view: normalize_quality(row.get(f"ImageQuality_{view}", ""))
                for view in ("2CH", "4CH")
            }

    return table


def normalize_quality(value):
    """CAMUS writes Good / Medium / Poor; anything else is bucketed as Unknown."""
    value = (value or "").strip().capitalize()
    return value if value in ("Good", "Medium", "Poor") else "Unknown"


# ----------------------------------------------------------------------
# Per-case scoring
# ----------------------------------------------------------------------
def spacing_from_header(header):
    """(dy, dx) in mm from the NIfTI header."""
    zooms = header.get_zooms()[:2]
    return (float(zooms[0]), float(zooms[1]))


def score_case(pred, gt, spacing, num_classes=4, with_surface=True):
    """
    Dice / ASSD / HD95 for one ED or ES frame
    """
    record = {"dice": {}, "assd": {}, "hd95": {}}

    for cls in range(1, num_classes):
        name = CLASS_NAMES.get(cls, f"class{cls}")

        pred_cls = pred == cls
        gt_cls = gt == cls

        inter = np.logical_and(pred_cls, gt_cls).sum()
        denom = pred_cls.sum() + gt_cls.sum()
        record["dice"][name] = float("nan") if denom == 0 else float(2.0 * inter / denom)

        if with_surface:
            assd, hd95 = assd_hd95_single(pred_cls, gt_cls, spacing=spacing)
        else:
            assd, hd95 = float("nan"), float("nan")

        record["assd"][name] = assd
        record["hd95"][name] = hd95

    for metric in ("dice", "assd", "hd95"):
        values = [v for v in record[metric].values() if np.isfinite(v)]
        record[f"{metric}_mean"] = float(np.mean(values)) if values else float("nan")

    return record


def summarise(records, num_classes=4):
    """
    Aggregate a list of per-case records.
    """
    if not records:
        return {"n": 0}

    out = {"n": len(records)}

    for metric in ("dice", "assd", "hd95"):
        for cls in range(1, num_classes):
            name = CLASS_NAMES.get(cls, f"class{cls}")
            values = np.array([r[metric][name] for r in records], dtype=float)
            finite = values[np.isfinite(values)]

            out[f"{metric}_{name}_mean"] = float(finite.mean()) if finite.size else float("nan")
            out[f"{metric}_{name}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
            out[f"{metric}_{name}_undefined"] = int(values.size - finite.size)

        means = np.array([r[f"{metric}_mean"] for r in records], dtype=float)
        finite = means[np.isfinite(means)]
        out[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")

    return out


def group_summaries(records, key, order=None, num_classes=4):
    groups = defaultdict(list)

    for record in records:
        groups[record[key]].append(record)

    keys = order if order else sorted(groups)
    return {k: summarise(groups[k], num_classes) for k in keys if k in groups}


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
def infer_edes(model, data_root, patient_ids, gt_root, pred_root, device, image_size,
               quality_table=None, num_classes=4, with_surface=True):
    """
    Inference on ED/ES images:
        patientXXXX_2CH_ED.nii.gz
        patientXXXX_2CH_ES.nii.gz
        patientXXXX_4CH_ED.nii.gz
        patientXXXX_4CH_ES.nii.gz

    Returns a list of per-case metric records (empty when no GT is available).
    """
    records = []

    for patient_id in tqdm(patient_ids, desc="Infer ED/ES"):
        patient_dir = data_root / patient_id

        if not patient_dir.is_dir():
            print(f"Warning: patient folder not found, skipping: {patient_dir}")
            continue

        patient_gt_dir = gt_root / patient_id
        patient_pred_dir = pred_root / patient_id
        patient_gt_dir.mkdir(parents=True, exist_ok=True)
        patient_pred_dir.mkdir(parents=True, exist_ok=True)

        for view in ["2CH", "4CH"]:
            for phase in ["ED", "ES"]:
                image_path = patient_dir / f"{patient_id}_{view}_{phase}.nii.gz"
                if not image_path.exists():
                    continue

                image, affine, header = load_nifti(image_path)
                image = squeeze_2d(image, image_path)
                original_size = image.shape

                # ---- Prediction ----
                pred = predict_frame(model=model, frame=image, device=device, image_size=image_size, original_size=original_size)
                save_mask_nifti(pred, affine=affine, header=header, out_dir=patient_pred_dir,
                    save_name=f"{patient_id}_{view}_{phase}_pred")

                # ---- Ground truth ----
                gt_path = patient_dir / f"{patient_id}_{view}_{phase}_gt.nii.gz"
                if gt_path.exists():
                    gt, gt_affine, gt_header = load_nifti(gt_path)
                    gt = squeeze_2d(gt, gt_path)
                    save_mask_nifti(gt.astype(np.uint8), affine=gt_affine, header=gt_header, out_dir=patient_gt_dir,
                        save_name=f"{patient_id}_{view}_{phase}_gt")

                    # ---- Scoring ----
                    if gt.shape != pred.shape:
                        print(f"Warning: shape mismatch, not scored: {gt_path}")
                        continue

                    spacing = spacing_from_header(header)
                    record = score_case(pred, gt.astype(np.uint8), spacing,
                                        num_classes=num_classes, with_surface=with_surface)

                    quality = "Unknown"
                    if quality_table is not None:
                        quality = quality_table.get(patient_id, {}).get(view, "Unknown")

                    record.update(patient_id=patient_id, view=view, phase=phase,
                                  quality=quality, spacing_mm=list(spacing),
                                  image_shape=list(original_size))
                    records.append(record)
                else:
                    print(f"Warning: GT not found: {gt_path}")

    return records


def infer_sequence(model, data_root, patient_ids, gt_root, pred_root, device, image_size):
    """
    Inference on half_sequence images:
        patientXXXX_2CH_half_sequence.nii.gz
        patientXXXX_4CH_half_sequence.nii.gz
    Input sequence shape:  [H, W, T]
    Output sequence shape: [H, W, T]
    """
    for patient_id in tqdm(patient_ids, desc="Infer sequence"):
        patient_dir = data_root / patient_id

        if not patient_dir.is_dir():
            print(f"Warning: patient folder not found, skipping: {patient_dir}")
            continue

        patient_gt_dir = gt_root / patient_id
        patient_pred_dir = pred_root / patient_id
        patient_gt_dir.mkdir(parents=True, exist_ok=True)
        patient_pred_dir.mkdir(parents=True, exist_ok=True)

        for view in ["2CH", "4CH"]:
            seq_path = patient_dir / f"{patient_id}_{view}_half_sequence.nii.gz"

            if not seq_path.exists():
                continue

            seq, affine, header = load_nifti(seq_path)

            if seq.ndim != 3:
                raise ValueError(f"Expected sequence [H, W, T], got {seq.shape}: {seq_path}")

            h, w, t_len = seq.shape
            pred_seq = np.zeros((h, w, t_len), dtype=np.uint8)

            for t in range(t_len):
                frame = seq[:, :, t]
                pred = predict_frame(model=model, frame=frame, device=device, image_size=image_size, original_size=(h, w))
                pred_seq[:, :, t] = pred

            save_mask_nifti(pred_seq, affine=affine, header=header, out_dir=patient_pred_dir, save_name=f"{patient_id}_{view}_sequence_pred")

            # ---- Ground truth (sequence) ----
            gt_path = patient_dir / f"{patient_id}_{view}_half_sequence_gt.nii.gz"

            if gt_path.exists():
                gt_seq, gt_affine, gt_header = load_nifti(gt_path)

                if gt_seq.ndim != 3:
                    raise ValueError(f"Expected sequence GT [H, W, T], got {gt_seq.shape}: {gt_path}")

                save_mask_nifti(gt_seq.astype(np.uint8), affine=gt_affine, header=gt_header,
                    out_dir=patient_gt_dir, save_name=f"{patient_id}_{view}_sequence_gt",
                )


def load_model(ckpt_path, device, model_name="unet", num_classes=4, base_channels=32):
    model = build_model(model_name=model_name, in_channels=1, num_classes=num_classes, base_channels=base_channels)
    checkpoint = torch.load(str(ckpt_path), map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.floating, np.integer)):
        return json_safe(obj.item())
    return obj


def print_summary(title, table, num_classes=4):
    names = [CLASS_NAMES.get(c, f"class{c}") for c in range(1, num_classes)]

    print(f"\n{title}")
    header = f"  {'group':<10}{'n':>5}" + "".join(f"{n:>9}" for n in names) + f"{'mean':>9}"
    header += f"{'ASSD':>9}{'HD95':>9}"
    print(header)

    for key, s in table.items():
        if not s.get("n"):
            continue
        row = f"  {key:<10}{s['n']:>5}"
        row += "".join(f"{s[f'dice_{n}_mean']:>9.4f}" for n in names)
        row += f"{s['dice_mean']:>9.4f}"
        row += f"{s['assd_mean']:>9.2f}{s['hd95_mean']:>9.2f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Inference for CAMUS 2D segmentation (test split only)")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="../../../data/camus/camus_pred")
    parser.add_argument("--gt_dirname", type=str, default="gt")
    parser.add_argument("--pred_dirname", type=str, default="pred")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "resunet", "attention_unet"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="all", choices=["edes", "sequence", "all"])
    parser.add_argument("--ef_csv", type=str, default="")
    parser.add_argument("--metrics_json", type=str, default="")
    parser.add_argument("--no_surface_metrics", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    gt_root = output_root / args.gt_dirname
    pred_root = output_root / args.pred_dirname
    gt_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # -------------------------
    # Load split
    # -------------------------
    split_info = load_split(args.split_json)
    test_ids = sorted(split_info["test"])

    quality_table = None
    if args.ef_csv:
        quality_table = load_quality_table(args.ef_csv)
        covered = sum(1 for p in test_ids if p in quality_table)
        print(f"Quality table: {len(quality_table)} patients, {covered}/{len(test_ids)} test patients covered")

    print(f"Using device: {device}")
    print(f"Data root: {data_root}")
    print(f"Split JSON: {args.split_json}")
    print(f"Test patients: {len(test_ids)}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"GT root:   {gt_root}")
    print(f"Pred root: {pred_root}")
    print(f"Mode: {args.mode}")

    image_size = (args.image_size, args.image_size)

    model = load_model(ckpt_path=args.ckpt_path, device=device, model_name=args.model, 
                       num_classes=args.num_classes, base_channels=args.base_channels)

    records = []

    if args.mode in ["edes", "all"]:
        records = infer_edes(model=model, data_root=data_root, patient_ids=test_ids, gt_root=gt_root, pred_root=pred_root,
                   device=device, image_size=image_size, quality_table=quality_table,
                   num_classes=args.num_classes, with_surface=not args.no_surface_metrics)

    if args.mode in ["sequence", "all"]:
        infer_sequence(model=model, data_root=data_root, patient_ids=test_ids, gt_root=gt_root,
                       pred_root=pred_root, device=device, image_size=image_size)

    # -------------------------
    # ED/ES report
    # -------------------------
    if records:
        report = {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "ckpt_path": args.ckpt_path,
                "split_json": args.split_json,
                "n_test_patients": len(test_ids),
                "n_scored_frames": len(records),
                "distance_units": "mm (original resolution, spacing from NIfTI header)",
                "args": vars(args),
            },
            "overall": summarise(records, args.num_classes),
            "by_quality": group_summaries(records, "quality", QUALITY_ORDER, args.num_classes),
            "by_view": group_summaries(records, "view", ["2CH", "4CH"], args.num_classes),
            "by_phase": group_summaries(records, "phase", ["ED", "ES"], args.num_classes),
            "per_case": records,
        }

        metrics_json = Path(args.metrics_json) if args.metrics_json else output_root / "metrics_edes.json"
        metrics_json.parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_json, "w") as f:
            json.dump(json_safe(report), f, indent=2, allow_nan=False)

        print_summary("Overall", {"all": report["overall"]}, args.num_classes)
        print_summary("By image quality", report["by_quality"], args.num_classes)
        print_summary("By view", report["by_view"], args.num_classes)
        print_summary("By phase", report["by_phase"], args.num_classes)

        undefined = sum(report["overall"][f"assd_{CLASS_NAMES[c]}_undefined"]
                        for c in range(1, args.num_classes))
        if undefined:
            print(f"\n  {undefined} class instances had an empty prediction or GT "
                  f"(ASSD/HD95 undefined, excluded from those means)")

        print(f"\nMetrics -> {metrics_json}")

    print("Inference finished.")

if __name__ == "__main__":
    main()