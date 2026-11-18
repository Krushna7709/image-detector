"""Minimal Grad-CAM for CNN detector explainability."""

from __future__ import annotations

import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._forward),
            target_layer.register_full_backward_hook(self._backward),
        ]

    def _forward(self, module, inputs, output):
        self.activations = output.detach()

    def _backward(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()

    def generate(self, tensor: torch.Tensor, out_size: tuple[int, int] | None = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x = tensor.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            if logits.dim() > 1 and logits.shape[-1] > 1:
                score = logits[:, 1].sum()
            else:
                score = logits.sum()
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations")

        weights = self.gradients.mean(dim=(2, 3))[0]
        acts = self.activations[0]
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-9)
        cam_np = cam.detach().cpu().numpy().astype(np.float32)
        if out_size is not None:
            cam_np = cv2.resize(cam_np, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)
        return cam_np
