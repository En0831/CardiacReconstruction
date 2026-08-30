# models/model_factory.py

from models.unet2d import UNet2D
from models.resunet2d import ResUNet2D
from models.attention_unet2d import AttentionUNet2D


def build_model(model_name, in_channels=1, num_classes=4, base_channels=32):
    model_name = model_name.lower()

    if model_name == "unet":
        return UNet2D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    if model_name == "resunet":
        return ResUNet2D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    if model_name == "attention_unet":
        return AttentionUNet2D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    raise ValueError(
        f"Unknown model_name: {model_name}. "
        "Choose from: unet, resunet, attention_unet"
    )