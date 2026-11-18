"""Lazy CLIP backbone loader.

The embedder is only initialised when a trained head needs it, so the service
starts instantly and works fully offline in heuristic mode.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from PIL import Image

from aiforensics.config import Settings

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loaded = False
        self.load_error: str | None = None
        self._model = None
        self._processor = None
        self._torch = None
        self._device = "cpu"
        self._lock = threading.Lock()

    def ensure_loaded(self) -> bool:
        if self.loaded:
            return True
        if self.load_error:
            return False
        with self._lock:
            if self.loaded:
                return True
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                if self.settings.clip_device == "auto":
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self._device = self.settings.clip_device

                logger.info(
                    "Loading CLIP backbone %s on %s",
                    self.settings.clip_model_name,
                    self._device,
                )
                model = CLIPModel.from_pretrained(self.settings.clip_model_name)
                model.to(self._device).eval()
                processor = CLIPProcessor.from_pretrained(self.settings.clip_model_name)
                self._torch = torch
                self._model = model
                self._processor = processor
                self.loaded = True
                return True
            except Exception as exc:  # offline, missing package, bad name...
                self.load_error = f"{type(exc).__name__}: {exc}"
                logger.warning("CLIP backbone unavailable: %s", self.load_error)
                return False

    def embed(self, image: Image.Image) -> np.ndarray | None:
        if not self.ensure_loaded():
            return None
        return self.embed_batch([image])

    def embed_batch(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray | None:
        if not self.ensure_loaded():
            return None
        torch = self._torch
        vectors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = images[start:start + batch_size]
                inputs = self._processor(images=batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self._device)
                features = self._model.get_image_features(pixel_values=pixel_values)
                features = torch.nn.functional.normalize(features, dim=-1)
                vectors.append(features.float().cpu().numpy())
        if not vectors:
            return None
        return np.concatenate(vectors, axis=0).astype(np.float32)
