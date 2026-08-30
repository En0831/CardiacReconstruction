# infer.py

"""Inference for CAMUS 2D segmentation"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.model_factory import build_model
from utils.split import load_split


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


def infer_edes(model, data_root, patient_ids, gt_root, pred_root, device, image_size):
    """
    Inference on ED/ES images:
        patientXXXX_2CH_ED.nii.gz
        patientXXXX_2CH_ES.nii.gz
        patientXXXX_4CH_ED.nii.gz
        patientXXXX_4CH_ES.nii.gz
    """
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
                else:
                    print(f"Warning: GT not found: {gt_path}")


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

    if args.mode in ["edes", "all"]:
        infer_edes(model=model, data_root=data_root, patient_ids=test_ids, gt_root=gt_root, pred_root=pred_root,
                   device=device, image_size=image_size,)

    if args.mode in ["sequence", "all"]:
        infer_sequence(model=model, data_root=data_root, patient_ids=test_ids, gt_root=gt_root,
                       pred_root=pred_root, device=device, image_size=image_size)

    print("Inference finished.")

if __name__ == "__main__":
    main()