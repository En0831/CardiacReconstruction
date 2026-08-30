# models/attention_unet2d.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )

        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        """
        g: gating signal from decoder
        x: skip connection from encoder
        """
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        alpha = self.sigmoid(self.psi(self.relu(self.W_g(g) + self.W_x(x))))

        return x * alpha


class AttentionUNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_channels=32):
        super().__init__()

        filters = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_channels, filters[0])
        self.enc2 = DoubleConv(filters[0], filters[1])
        self.enc3 = DoubleConv(filters[1], filters[2])
        self.enc4 = DoubleConv(filters[2], filters[3])

        self.bottleneck = DoubleConv(filters[3], filters[4])

        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att4 = AttentionGate(filters[3], filters[3], filters[2])
        self.dec4 = DoubleConv(filters[4], filters[3])

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att3 = AttentionGate(filters[2], filters[2], filters[1])
        self.dec3 = DoubleConv(filters[3], filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att2 = AttentionGate(filters[1], filters[1], filters[0])
        self.dec2 = DoubleConv(filters[2], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.att1 = AttentionGate(filters[0], filters[0], max(filters[0] // 2, 1))
        self.dec1 = DoubleConv(filters[1], filters[0])

        self.out_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def _crop_or_pad(self, x, ref):
        diff_y = ref.size(2) - x.size(2)
        diff_x = ref.size(3) - x.size(3)

        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        return x

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self._crop_or_pad(d4, e4)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        d3 = self._crop_or_pad(d3, e3)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        d2 = self._crop_or_pad(d2, e2)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        d1 = self._crop_or_pad(d1, e1)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return self.out_conv(d1)