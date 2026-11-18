"""Torch model definitions (optional training path)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """Compact from-scratch CNN used as the CPU-friendly baseline."""

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(3, w),
            _block(w, w * 2),
            _block(w * 2, w * 4),
            _block(w * 4, w * 8),
            nn.Conv2d(w * 8, w * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w * 8),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(w * 8, 1)

    @property
    def target_layer(self) -> nn.Module:
        return self.features[-3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x)).squeeze(1)


class ConvNeXtDetector(nn.Module):
    """ConvNeXt-Tiny baseline from torchvision (used when installed)."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        from torchvision import models

        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        self.features = backbone.features
        self.norm = backbone.classifier[0]
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(768, 1)
        self.target_layer = self.features[-1][-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x).squeeze(1)


def build_detector(arch: str = "smallcnn", width: int = 32, pretrained: bool = False) -> nn.Module:
    if arch == "smallcnn":
        return SmallCNN(width=width)
    if arch == "convnext_tiny":
        return ConvNeXtDetector(pretrained=pretrained)
    raise ValueError(f"Unknown architecture '{arch}'")
