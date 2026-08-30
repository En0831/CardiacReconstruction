# utils/dataset.py

from pathlib import Path
import random

import numpy as np
import nibabel as nib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CAMUSDataset(Dataset):
    def __init__(
        self,
        data_root,
        patient_ids,
        image_size=(256, 256),
        views=("2CH", "4CH"),
        phases=("ED", "ES"),
        augment=False,
        # --- extra mid-cycle frames from half_sequence ---
        use_sequence_frame=False,
        # --- geometric augmentation parameters ---
        p_affine=0.5,
        rotation_deg=15.0,
        translate_frac=0.10,
        scale_range=(0.9, 1.1),
        # --- intensity augmentation parameters ---
        p_gamma=0.5,
        gamma_range=(0.8, 1.2),
        p_jitter=0.5,
        brightness_std=0.1,   # additive shift in z-score units
        contrast_range=(0.9, 1.1),
        p_noise=0.3,
        noise_std=0.05,       # Gaussian noise std in z-score units
    ):
        """
        Returns:
            image: torch.FloatTensor [1, H, W]
            mask:  torch.LongTensor  [H, W]
        """
        self.data_root = Path(data_root)
        self.patient_ids = patient_ids
        self.image_size = image_size
        self.views = views
        self.phases = phases

        self.augment = augment
        self.use_sequence_frame = use_sequence_frame

        self.p_affine = p_affine
        self.rotation_deg = rotation_deg
        self.translate_frac = translate_frac
        self.scale_range = scale_range

        self.p_gamma = p_gamma
        self.gamma_range = gamma_range
        self.p_jitter = p_jitter
        self.brightness_std = brightness_std
        self.contrast_range = contrast_range
        self.p_noise = p_noise
        self.noise_std = noise_std

        self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.data_root}")

    def _collect_samples(self):
        samples = []

        for patient_id in self.patient_ids:
            patient_dir = self.data_root / patient_id

            for view in self.views:
                for phase in self.phases:
                    image_path = patient_dir / f"{patient_id}_{view}_{phase}.nii.gz"
                    mask_path = patient_dir / f"{patient_id}_{view}_{phase}_gt.nii.gz"

                    if image_path.exists() and mask_path.exists():
                        samples.append(
                            {
                                "image_path": image_path,
                                "mask_path": mask_path,
                                "patient_id": patient_id,
                                "view": view,
                                "phase": phase,
                                "frame": None,
                                "n_frames": None,
                            }
                        )

                if self.use_sequence_frame:
                    seq_sample = self._sequence_sample(patient_dir, patient_id, view)

                    if seq_sample is not None:
                        samples.append(seq_sample)
        return samples


    def _sequence_sample(self, patient_dir, patient_id, view):
        image_path = patient_dir / f"{patient_id}_{view}_half_sequence.nii.gz"
        mask_path = patient_dir / f"{patient_id}_{view}_half_sequence_gt.nii.gz"

        if not (image_path.exists() and mask_path.exists()):
            return None

        image_shape = nib.load(str(image_path)).shape
        mask_shape = nib.load(str(mask_path)).shape

        if len(image_shape) != 3 or len(mask_shape) != 3:
            print(f"Warning: not a [H, W, T] sequence, skipping: {image_path.name}")
            return None

        if image_shape[2] != mask_shape[2]:
            print(
                f"Warning: image has {image_shape[2]} frames but GT has "
                f"{mask_shape[2]}, skipping: {image_path.name}"
            )
            return None

        n_frames = mask_shape[2]

        if n_frames < 3:
            return None

        return {
            "image_path": image_path,
            "mask_path": mask_path,
            "patient_id": patient_id,
            "view": view,
            "phase": "SEQ",
            "frame": n_frames // 2,
            "n_frames": n_frames,
        }

    def __len__(self):
        return len(self.samples)

    def _load_nifti(self, path, frame=None):
        """
        Load one 2D array from a NIfTI file.
        """
        proxy = nib.load(str(path))

        if frame is not None:
            return np.asarray(proxy.dataobj[:, :, frame])
        arr = np.asarray(proxy.dataobj)

        if arr.ndim == 3:
            if arr.shape[-1] == 1:
                arr = arr[:, :, 0]
            else:
                raise ValueError(f"Expected 2D image, got shape {arr.shape}: {path}")

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {arr.shape}: {path}")

        return arr


    def _resolve_frame(self, sample):
        n_frames = sample["n_frames"]
        if n_frames is None:
            return None
        return random.randrange(1, n_frames - 1)


    # ------------------------------------------------------------------
    # Intensity augmentation
    # ------------------------------------------------------------------
    def _random_gamma(self, image):
        """
        Random gamma on RAW intensities.
        image: numpy [H, W] (raw, before z-score)
        """
        gamma = random.uniform(*self.gamma_range)

        lo = image.min()
        hi = image.max()

        if hi - lo < 1e-8:
            return image

        image01 = (image - lo) / (hi - lo)
        image01 = np.power(image01, gamma)
        return image01 * (hi - lo) + lo


    def _normalize_image(self, image):
        image = image.astype(np.float32)

        # Robust intensity normalization for ultrasound
        p1, p99 = np.percentile(image, (1, 99))
        image = np.clip(image, p1, p99)
        mean = image.mean()
        std = image.std()

        if std < 1e-8:
            image = image - mean
        else:
            image = (image - mean) / std
        return image


    def _random_intensity_jitter(self, image):
        """
        Brightness / contrast jitter on z-scored image.
        image: torch [1, H, W]
        """
        contrast = random.uniform(*self.contrast_range)
        brightness = random.gauss(0.0, self.brightness_std)
        return image * contrast + brightness


    def _random_noise(self, image):
        """
        Additive Gaussian noise on z-scored image.
        image: torch [1, H, W]
        """
        noise = torch.randn_like(image) * self.noise_std
        return image + noise

    # ------------------------------------------------------------------
    # Geometric augmentation (image and mask share the same transform)
    # ------------------------------------------------------------------
    def _random_affine(self, image, mask):
        """
        Random rotation + translation + scaling, applied identically
        to image (bilinear) and mask (nearest).

        image: torch [1, H, W]
        mask:  torch [H, W] (long)
        """
        angle = np.deg2rad(random.uniform(-self.rotation_deg, self.rotation_deg))
        scale = random.uniform(*self.scale_range)
        tx = random.uniform(-self.translate_frac, self.translate_frac) * 2.0
        ty = random.uniform(-self.translate_frac, self.translate_frac) * 2.0

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        inv_s = 1.0 / scale

        theta = torch.tensor(
            [
                [cos_a * inv_s, -sin_a * inv_s, tx],
                [sin_a * inv_s, cos_a * inv_s, ty],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)  # [1, 2, 3]

        img = image.unsqueeze(0)                    # [1, 1, H, W]
        msk = mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        grid = F.affine_grid(theta, size=img.shape, align_corners=False)
        fill = img.min()

        img = F.grid_sample(img - fill, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + fill
        msk = F.grid_sample(msk, grid, mode="nearest", padding_mode="zeros", align_corners=False)

        image = img.squeeze(0)                       # [1, H, W]
        mask = msk.squeeze(0).squeeze(0).long()      # [H, W]

        return image, mask

    # ------------------------------------------------------------------
    # Resizing
    # ------------------------------------------------------------------
    def _resize_image(self, image):
        """
        image: numpy [H, W]
        return: torch [1, image_size[0], image_size[1]]
        """
        x = torch.from_numpy(image).float()
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
        x = x.squeeze(0)  # [1, H, W]
        return x

    def _resize_mask(self, mask):
        """
        mask: numpy [H, W], values 0,1,2,3
        return: torch [image_size[0], image_size[1]]
        """
        x = torch.from_numpy(mask).float()
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        x = F.interpolate(x, size=self.image_size, mode="nearest")
        x = x.squeeze(0).squeeze(0).long()  # [H, W]
        return x

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame = self._resolve_frame(sample)
        image = self._load_nifti(sample["image_path"], frame)
        mask = self._load_nifti(sample["mask_path"], frame)

        # --- intensity aug on raw image (gamma) ---
        if self.augment and random.random() < self.p_gamma:
            image = self._random_gamma(image)

        image = self._normalize_image(image)
        mask = mask.astype(np.int64)
        image = self._resize_image(image)
        mask = self._resize_mask(mask)

        if self.augment:
            # --- geometric aug (shared transform) ---
            if random.random() < self.p_affine:
                image, mask = self._random_affine(image, mask)

            # --- intensity aug on normalized image ---
            if random.random() < self.p_jitter:
                image = self._random_intensity_jitter(image)

            if random.random() < self.p_noise:
                image = self._random_noise(image)

        return image, mask