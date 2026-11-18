"""U-Net localizer wrapper (optional training route)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class UNetLocalizer:
    def __init__(self, model, torch, device: str, input_size: int):
        self.model = model
        self.torch = torch
        self.device = device
        self.input_size = input_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @classmethod
    def load_if_available(cls, path: Path) -> "UNetLocalizer | None":
        if not path.exists():
            return None
        try:
            import torch

            from aiforensics.models.unet import build_unet

            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = build_unet(width=int(checkpoint.get("width", 32)))
            model.load_state_dict(checkpoint["state_dict"])
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device).eval()
            return cls(model, torch, device, int(checkpoint.get("input_size", 512)))
        except Exception as exc:
            logger.warning("Could not load localizer from %s: %s", path, exc)
            return None

    def predict(self, rgb_uint8: np.ndarray) -> np.ndarray:
        torch = self.torch
        h, w = rgb_uint8.shape[:2]
        resized = cv2.resize(rgb_uint8, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        array = resized.astype(np.float32) / 255.0
        array = (array - self.mean) / self.std
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        mask = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2)
        return np.clip(mask, 0.0, 1.0)
