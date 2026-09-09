# lcunet/unet.py
# This architecture follows the one used in previous work (https://github.com/m-qiang/HeartSSM)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock3d(nn.Module):
    """
    3D convolutional block

    Args:
    - dim_in: dim of input channels
    - dim_out: dim of output channels
    - K: kernel size

    Input:
    - x: input features
    """
    def __init__(self, dim_in, dim_out, K=3, stride=1, drop_rate=0.0):
        super(ConvBlock3d, self).__init__()

        self.conv = nn.Conv3d(
            in_channels=dim_in, out_channels=dim_out,
            kernel_size=K, stride=stride, padding=K//2)
        self.norm = nn.InstanceNorm3d(dim_in, affine=False)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):
        # norm -> activation -> dropout -> conv -> norm
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    """
    3D U-Net

    Args:
    - dim_in: dim of input channels
    - dim_hid: dim of hidden channels
    - dim_out: dim of output channels
    - K: kernel size
    - drop_rate: dropout rate

    Inputs:
    - x: input volume (B,C_in,L,W,H)

    Returns:
    - x: output volume (B,C_out,L,W,H)
    """
    def __init__(
        self,
        dim_in=1,
        dim_hid=[16,32,64,128,256],
        dim_out=1,
        K=3,
        drop_rate=0.0
    ):
        super(UNet, self).__init__()

        self.conv0 = nn.Conv3d(in_channels=dim_in, out_channels=dim_hid[0], kernel_size=K, stride=1, padding=K//2)
        self.conv1 = ConvBlock3d(dim_in=dim_hid[0], dim_out=dim_hid[0], K=K, stride=1, drop_rate=drop_rate)
        self.conv2 = ConvBlock3d(dim_in=dim_hid[0], dim_out=dim_hid[1], K=K, stride=2, drop_rate=drop_rate)
        self.conv3 = ConvBlock3d(dim_in=dim_hid[1], dim_out=dim_hid[2], K=K, stride=2, drop_rate=drop_rate)
        self.conv4 = ConvBlock3d(dim_in=dim_hid[2], dim_out=dim_hid[3], K=K, stride=2, drop_rate=drop_rate)
        self.conv5 = ConvBlock3d(dim_in=dim_hid[3], dim_out=dim_hid[4], K=K, stride=1, drop_rate=drop_rate)
        
        self.deconv5 = ConvBlock3d(dim_in=dim_hid[4], dim_out=dim_hid[3], K=K, stride=1, drop_rate=drop_rate)
        self.deconv4 = ConvBlock3d(dim_in=dim_hid[3]*2, dim_out=dim_hid[2], K=K, stride=1, drop_rate=drop_rate)
        self.deconv3 = ConvBlock3d(dim_in=dim_hid[2]*2, dim_out=dim_hid[1], K=K, stride=1, drop_rate=drop_rate)
        self.deconv2 = ConvBlock3d(dim_in=dim_hid[1]*2, dim_out=dim_hid[0], K=K, stride=1, drop_rate=drop_rate)
        self.deconv1 = ConvBlock3d(dim_in=dim_hid[0]*2, dim_out=dim_hid[0], K=K, stride=1, drop_rate=drop_rate)
        self.deconv0 = ConvBlock3d(dim_in=dim_hid[0], dim_out=dim_out, K=K, stride=1, drop_rate=drop_rate)

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        # encode
        x = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x  = self.conv5(x4)

        # decode
        x = self.deconv5(x)
        x = torch.cat([x, x4], dim=1)
        x = self.deconv4(x)
        
        x = self.up(x)
        x = torch.cat([x, x3], dim=1)
        x = self.deconv3(x)
        
        x = self.up(x)
        x = torch.cat([x, x2], dim=1)
        x = self.deconv2(x)
        
        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = self.deconv1(x)
        
        x = self.deconv0(x)
        
        return x

