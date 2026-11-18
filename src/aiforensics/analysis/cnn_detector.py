"""CNN detector wrapper (optional training route)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class CNNDetector:
    def __init__(self, model, torch, device, input_size: int, mean, std, target_layer, arch: str):
        self.model = model
        self.torch = torch
        self.device = device
        self.input_size = input_size
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.target_layer = target_layer
        self.arch = arch

    @classmethod
    def load_if_available(cls, path: Path) -> "CNNDetector | None":
        if not path.exists():
            return None
        try:
            import torch

            from aiforensics.models.cnn import build_detector

            checkpoint = _load_checkpoint(torch, path)
            arch = checkpoint.get("arch", "smallcnn")
            model = build_detector(arch, width=int(checkpoint.get("width", 32)))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            return cls(
                model=model,
                torch=torch,
                device=device,
                input_size=int(checkpoint.get("input_size", 256)),
                mean=checkpoint.get("mean", [0.485, 0.456, 0.406]),
                std=checkpoint.get("std", [0.229, 0.224, 0.225]),
                target_layer=checkpoint.get("target_layer", None) or model.target_layer,
                arch=arch,
            )
        except Exception as exc:
            logger.warning("Could not load CNN checkpoint from %s: %s", path, exc)
            return None

    def _tensor(self, image: Image.Image):
        torch = self.torch
        resized = image.convert("RGB").resize(
            (self.input_size, self.input_size), Image.Resampling.BILINEAR
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image: Image.Image) -> float:
        torch = self.torch
        with torch.no_grad():
            logits = self.model(self._tensor(image))
        if logits.shape[-1] == 1:
            return float(torch.sigmoid(logits).item())
        return float(torch.softmax(logits, dim=-1)[0, 1].item())

    def gradcam(self, image: Image.Image) -> np.ndarray | None:
        try:
            from aiforensics.explain.gradcam import GradCAM

            tensor = self._tensor(image)
            cam = GradCAM(self.model, self.target_layer).generate(
                tensor, out_size=(image.height, image.width)
            )
            return cam
        except Exception as exc:
            logger.warning("Grad-CAM failed: %s", exc)
            return None


def _load_checkpoint(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)
