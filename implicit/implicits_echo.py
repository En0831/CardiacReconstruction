# implicit/implicits_echo.py

from typing import List, Optional, Sequence
import torch
import torch.nn.functional as F

from common.losses import masked_ce_dice

# ==============================
# Model
# ==============================
class MultiClassOccupancyPredictor(torch.nn.Module):
    """DeepSDF-style coordinate MLP with `num_classes` output logits."""

    def __init__(self, latent_dim: int, spatial_dim: int, num_layers: int,
                 layers_with_coords: List[int], num_classes: int):
        def block(num_ch_in: int, num_ch_out: int):
            return torch.nn.Sequential(
                torch.nn.Linear(num_ch_in, num_ch_out),
                torch.nn.ReLU(True),
            )

        super().__init__()

        self.layers_with_coords = layers_with_coords
        in_channels = [latent_dim] * num_layers
        channels_with_coords = latent_dim + spatial_dim
        for lyr_id in self.layers_with_coords:
            in_channels[lyr_id] = channels_with_coords
        self.res_layers = torch.nn.ModuleList(
            [block(in_channels[i], latent_dim) for i in range(num_layers - 1)])
        # The last layer outputs `num_classes` logits instead of 1 (binary occupancy -> multi-class)
        self.last_layer = torch.nn.Linear(in_channels[-1], num_classes)

    def forward(self, closest_latents: torch.Tensor, local_coords: torch.Tensor) -> torch.Tensor:
        """[B, *ST, Z] -> [B, *ST, num_classes]"""
        features = closest_latents

        for i, layer in enumerate(self.res_layers):
            append_coords = i in self.layers_with_coords
            if append_coords:
                features = torch.cat([features, local_coords], dim=-1)

            out = layer(features)
            # No skip connection on layers where coordinates were concatenated
            features = out if append_coords else features + out

        features = self.last_layer(features)
        return features


class MultiClassAutoDecoder(torch.nn.Module):
    """Encoder-free implicit multi-class shape prior."""

    def __init__(self, lat_dim: int, spatial_dim: int, image_size: torch.Tensor,
                 occnet_num_layers: int, occnet_layers_with_coords: List[int],
                 num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.image_size: torch.Tensor
        self.register_buffer('image_size', image_size)
        latent_coords: torch.Tensor = image_size / 2  # noqa
        self.latent_coords: torch.Tensor
        self.register_buffer('latent_coords', latent_coords)
        self.occp_pred = MultiClassOccupancyPredictor(
            lat_dim, spatial_dim, occnet_num_layers, occnet_layers_with_coords, num_classes)

    def forward(self, latents: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """
        latents:     [B, Z]
        coordinates: [B, *ST, 3]   physical mm, same frame as during training
        returns:     [B, C, *ST]   logits
        """
        local_coords = coordinates - self.latent_coords
        n_spatial = coordinates.dim() - 2  # 1 for [B,N,3]; 3 for [B,X,Y,Z,3]
        lat = latents.reshape(latents.shape[0], *([1] * n_spatial), latents.shape[-1])
        lat = lat.expand(-1, *coordinates.shape[1:-1], -1)
        logits = self.occp_pred(lat, local_coords)   # [B, *ST, C]
        return logits.movedim(-1, 1)                 # [B, C, *ST]


# ==============================
# Losses
# ==============================

class PartialLabelLoss(torch.nn.Module):
    def __init__(self, observed_map: dict, num_classes: int = 6, eps: float = 1e-6,
                 neg_weight: float = 1.0, ce_weight: float = 1.0, observed_classes: Optional[Sequence[int]] = None):
        super().__init__()
        self.observed_map = dict(observed_map)
        self.num_classes = num_classes
        self.eps = eps
        self.neg_weight = neg_weight
        self.ce_weight = ce_weight
        self.observed_model_classes: Sequence[int] = (sorted(observed_classes) if observed_classes is not None else sorted(self.observed_map.values()))

    def forward(self, logits: torch.Tensor, echo_labels: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        logits:        [B, C, N]
        echo_labels:   [B, N]  values in the echo label space (0 = bg on the plane)
        observed_mask: [B, N]  bool; True where the voxel actually lies inside the
                               imaging plane/sector. Voxels outside carry no
                               information and are dropped entirely.
        """
        probs = torch.softmax(logits, dim=1)                       # [B, C, N]
        if observed_mask is None:
            observed_mask = torch.ones_like(echo_labels, dtype=torch.bool)

        # ---- remap echo labels -> model labels ----
        model_labels = torch.zeros_like(echo_labels, dtype=torch.long)
        for src, dst in self.observed_map.items():
            model_labels[echo_labels == src] = dst

        fg = observed_mask & (echo_labels > 0)
        bg = observed_mask & (echo_labels == 0)

        loss = logits.new_zeros(())

        # positive term: standard CE where the class is actually known
        loss = loss + masked_ce_dice(logits, model_labels, fg, self.observed_model_classes,
                                     ce_weight=self.ce_weight, eps=self.eps)

        # negative term: "none of the observed structures live here"
        if bg.any():
            p_obs = probs[:, self.observed_model_classes].sum(1)   # [B, N]
            p_obs = p_obs[bg].clamp(max=1.0 - self.eps)
            loss = loss - self.neg_weight * torch.log(1.0 - p_obs + self.eps).mean()

        return loss