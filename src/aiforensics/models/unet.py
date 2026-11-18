"""Small U-Net for manipulated-region segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.down1 = DoubleConv(3, w)
        self.down2 = DoubleConv(w, w * 2)
        self.down3 = DoubleConv(w * 2, w * 4)
        self.down4 = DoubleConv(w * 4, w * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(w * 8, w * 16)
        self.up4 = nn.ConvTranspose2d(w * 16, w * 8, 2, stride=2)
        self.conv4 = DoubleConv(w * 16, w * 8)
        self.up3 = nn.ConvTranspose2d(w * 8, w * 4, 2, stride=2)
        self.conv3 = DoubleConv(w * 8, w * 4)
        self.up2 = nn.ConvTranspose2d(w * 4, w * 2, 2, stride=2)
        self.conv2 = DoubleConv(w * 4, w * 2)
        self.up1 = nn.ConvTranspose2d(w * 2, w, 2, stride=2)
        self.conv1 = DoubleConv(w * 2, w)
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        b = self.bottleneck(self.pool(d4))
        u4 = self.conv4(torch.cat([self.up4(b), d4], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1))
        return self.head(u1)


def build_unet(width: int = 32) -> UNet:
    return UNet(width=width)
