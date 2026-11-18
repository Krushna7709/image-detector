"""The ForensIQ analysis engine.

Orchestrates preprocessing, metadata forensics, classical evidence, optional
trained detectors, attribution and edit localization, then assembles one
explainable result.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from aiforensics.analysis.attribution import AttributionModel
from aiforensics.analysis.clip_embedder import CLIPEmbedder
from aiforensics.analysis.cnn_detector import CNNDetector
from aiforensics.analysis.detector import LinearDetector
from aiforensics.analysis.heuristic import HeuristicAnalyzer
from aiforensics.analysis.localization import (
    HeuristicLocalizer,
    regions_from_heat,
    summarize_heat,
)
from aiforensics.analysis.localizer_model import UNetLocalizer
from aiforensics.config import Settings, settings as default_settings
from aiforensics.core.fusion import (
    decide_label,
    fuse_probabilities,
    probability_to_confidence,
)
from aiforensics.explain import (
    encode_png,
    heatmap_overlay,
    regions_overlay,
    spectrum_data_url,
)
from aiforensics.metadata import extract_metadata
from aiforensics.preprocessing import distort_image, load_image
from aiforensics.schemas import (
    AnalysisResult,
    AttributionItem,
    EditRegion,
    EditReport,
    EngineHealth,
    ExplainReport,
    Verdict,
)

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "AI probability is probabilistic decision support, not proof. No detector is 100% reliable."
)
_PRIVACY = "The image is processed in memory and is not persisted by this service."
_MIN_INFORMATIVE_SIDE = 32
_MIN_INFORMATIVE_STD = 1.0


def _looks_uninformative(loaded) -> bool:
    if min(loaded.width, loaded.height) < _MIN_INFORMATIVE_SIDE:
        return True
    return float(loaded.analysis_rgb.std()) < _MIN_INFORMATIVE_STD


class ForensicsEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings
        self.heuristic = HeuristicAnalyzer()
        self.localizer = HeuristicLocalizer()
        self.embedder = CLIPEmbedder(self.settings)
        self.linear = LinearDetector.load_if_available(self.settings.detector_path)
        self.attribution = AttributionModel.load_if_available(self.settings.attribution_path)
        self.cnn = CNNDetector.load_if_available(self.settings.cnn_path)
        self.unet = UNetLocalizer.load_if_available(self.settings.localizer_path)
        self._started = time.time()
        self._cache: OrderedDict[str, AnalysisResult] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._signature = self._build_signature()

    # ------------------------------------------------------------------ info
    @property
    def detector_mode(self) -> str:
        if self.cnn is not None:
            return "cnn"
        if self.linear is not None:
            return "clip-linear"
        return "heuristic"

    def health(self) -> EngineHealth:
        return EngineHealth(
            status="ok",
            version=self.settings.version,
            detector=self.detector_mode,
            attribution=self.attribution is not None,
            localizer="unet" if self.unet is not None else "heuristic",
            clip_backbone=self.settings.clip_model_name,
            clip_loaded=self.embedder.loaded,
            uptime_seconds=int(time.time() - self._started),
        )

    def _build_signature(self) -> str:
        stamps = []
        for path in (
            self.settings.detector_path,
            self.settings.attribution_path,
            self.settings.cnn_path,
            self.settings.localizer_path,
        ):
            try:
                stamps.append(str(path.stat().st_mtime_ns))
            except OSError:
                stamps.append("-")
        return f"{self.settings.version}|{self.detector_mode}|" + "|".join(stamps)

    # --------------------------------------------------------------- analyse
    def analyze(self, data: bytes, filename: str, distortion: str = "none") -> AnalysisResult:
        started = time.perf_counter()
        data, distortion_label = distort_image(data, distortion, self.settings)
        loaded = load_image(data, filename, self.settings)
        cache_key = f"{loaded.sha256}:{distortion_label}:{self._signature}"
        safe_name = Path(filename or "upload").name[:120]

        cached = self._cache_get(cache_key)
        if cached is not None:
            if cached.filename != safe_name:
                return cached.model_copy(update={"filename": safe_name})
            return cached

        bundle = extract_metadata(data, loaded)
        meta = bundle.report

        heuristic = self.heuristic.analyze(
            loaded.analysis_rgb,
            generator_hint=bundle.generator_hint,
            has_camera_exif=bool(meta.camera),
            jpeg_quality=meta.jpeg_quality_estimate,
            synthetic_provenance=bundle.synthetic_provenance,
        )

        notes: list[str] = []
        if distortion_label != "none":
            notes.append(f"Analysed a simulated copy: {distortion_label.lower()}.")
        if (
            not bundle.generator_hint
            and not bundle.synthetic_provenance
            and _looks_uninformative(loaded)
        ):
            heuristic.probability = 0.5
            heuristic.reliability = min(heuristic.reliability, 0.05)
            notes.append(
                "Image is too small or too uniform for reliable classical forensic analysis; "
                "these signals are treated as inconclusive."
            )

        embedding: np.ndarray | None = None
        if self.linear is not None or self.attribution is not None:
            embedding = self.embedder.embed(loaded.pil)
            if embedding is None:
                notes.append(
                    "A trained head is configured but the CLIP backbone is unavailable; "
                    "falling back to classical evidence."
                )
                logger.info("CLIP embedding skipped: %s", self.embedder.load_error)

        model_probability: float | None = None
        engine_name = "heuristic"
        calibrated = False
        if self.cnn is not None:
            try:
                model_probability = self.cnn.predict(loaded.pil)
                engine_name = "cnn"
                notes.append(f"Trained {self.cnn.arch} detector applied (Grad-CAM available).")
            except Exception as exc:
                logger.warning("CNN inference failed: %s", exc)
                notes.append("CNN inference failed; falling back to classical evidence.")
        if model_probability is None and self.linear is not None and embedding is not None:
            try:
                model_probability = self.linear.predict(embedding)
                engine_name = "clip-linear"
                calibrated = self.linear.calibrated
                trained = ", ".join(self.linear.trained_on) if self.linear.trained_on else "unspecified generators"
                notes.append(f"CLIP embedding + linear head applied (trained on: {trained}).")
            except Exception as exc:
                logger.warning("Linear head inference failed: %s", exc)
                notes.append("Linear head inference failed; falling back to classical evidence.")

        probability = fuse_probabilities(
            model_probability, heuristic.probability, self.settings.model_fusion_weight
        )
        confidence = probability_to_confidence(probability)
        label = decide_label(
            probability, self.settings.uncertain_low, self.settings.uncertain_high
        )
        reliability = 0.8 if model_probability is not None else heuristic.reliability

        if model_probability is None:
            notes.append(
                "Classical forensic engine in use (no trained detector loaded). Scores are "
                "indicators, not calibrated probabilities; train a head to enable model decisions."
            )
        else:
            notes.append(
                "Final probability blends the trained head with classical evidence "
                f"({self.settings.model_fusion_weight:.0%} model weight)."
            )
        if label == "uncertain":
            notes.append(
                "Evidence is ambiguous (within the uncertain band); do not present this as a verdict."
            )
        notes.extend(heuristic.notes)

        attribution = self._attribute(bundle.generator_hint, label, embedding, notes)
        edit = self._localize(loaded, label, notes)
        explain = self._explain(loaded, notes)

        notes.append(_DISCLAIMER)
        notes.append(_PRIVACY)

        result = AnalysisResult(
            id=loaded.sha256[:16],
            filename=safe_name,
            sha256=loaded.sha256,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            engine=engine_name,
            distortion=distortion_label,
            verdict=Verdict(
                label=label,
                ai_probability=round(probability, 4),
                confidence=confidence,
                reliability=round(reliability, 4),
                threshold=self.settings.decision_threshold,
                uncertain_range=(self.settings.uncertain_low, self.settings.uncertain_high),
                engine=engine_name,
                calibrated=calibrated,
            ),
            attribution=attribution,
            edit=edit,
            metadata=meta,
            explain=explain,
            signals=heuristic.signals,
            notes=notes,
        )
        self._cache_put(cache_key, result)
        return result

    # ------------------------------------------------------------ sub-steps
    def _attribute(
        self,
        generator_hint: str | None,
        label: str,
        embedding: np.ndarray | None,
        notes: list[str],
    ) -> list[AttributionItem]:
        items: list[AttributionItem] = []
        if generator_hint:
            items.append(
                AttributionItem(
                    generator=generator_hint, confidence=0.92, source="metadata", known=True
                )
            )
            notes.append(
                f"Generator suggested by provenance metadata: {generator_hint} (metadata is spoofable)."
            )

        if label != "real" and self.attribution is not None and embedding is not None:
            try:
                predictions = self.attribution.predict(embedding)
                known = [(name, prob) for name, prob in predictions if prob >= self.attribution.threshold]
                existing = {item.generator for item in items}
                if known:
                    for name, prob in known[:3]:
                        if name not in existing:
                            items.append(
                                AttributionItem(
                                    generator=name, confidence=prob, source="model", known=True
                                )
                            )
                else:
                    top_name, top_prob = predictions[0] if predictions else ("unknown", 0.0)
                    items.append(
                        AttributionItem(
                            generator="unknown generator",
                            confidence=round(max(0.0, 1.0 - top_prob), 4),
                            source="model",
                            known=False,
                        )
                    )
                    notes.append(
                        f"Attribution confidence is below the open-set threshold "
                        f"({self.attribution.threshold:.0%}); reporting an unknown generator."
                    )
            except Exception as exc:
                logger.warning("Attribution inference failed: %s", exc)

        if not items:
            if label == "real":
                notes.append("Attribution skipped: the evidence favours a real photograph.")
                return []
            if self.attribution is None:
                notes.append(
                    "No attribution head is loaded. Generator names need provenance metadata "
                    "or a trained attribution model (see README)."
                )
            items.append(
                AttributionItem(
                    generator="unknown generator", confidence=0.0, source="none", known=False
                )
            )
        return items

    def _localize(self, loaded, label: str, notes: list[str]) -> EditReport:
        display = loaded.display_rgb
        dh, dw = display.shape[:2]
        heat = None
        regions_work = []
        edited = 0.0
        method = "heuristic"
        note = ""

        if self.unet is not None:
            try:
                heat = self.unet.predict(loaded.analysis_rgb)
                method = "unet"
                _, _, edited = summarize_heat(heat)
                regions_work = regions_from_heat(heat)
                note = (
                    "U-Net localizer output: probability that each pixel belongs to a "
                    "manipulated region, trained on edit masks."
                )
            except Exception as exc:
                logger.warning("U-Net localizer failed: %s", exc)
                heat = None

        if heat is None:
            localization = self.localizer.localize(loaded.analysis_rgb)
            heat = localization.heat
            method = localization.method
            note = localization.note
            edited = localization.edited_probability
            regions_work = localization.regions

        hw, hh = heat.shape[1], heat.shape[0]
        sx, sy = dw / hw, dh / hh
        regions = [
            EditRegion(
                x=round(region.x * sx),
                y=round(region.y * sy),
                width=max(1, round(region.width * sx)),
                height=max(1, round(region.height * sy)),
                score=region.score,
            )
            for region in regions_work
        ]

        heat_display = cv2.resize(heat, (dw, dh), interpolation=cv2.INTER_LINEAR)
        overlay = heatmap_overlay(display, heat_display)
        boxes = regions_overlay((dh, dw), regions)

        if label == "ai_generated":
            note += (
                " For a fully synthetic image this map reflects internal inconsistencies, "
                "not a known edit mask."
            )
        elif method == "heuristic":
            notes.append("Edit localization is heuristic and requires human review.")

        return EditReport(
            edited_probability=round(edited, 4),
            regions=regions,
            heatmap=encode_png(overlay),
            boxes=encode_png(boxes),
            method=method,
            note=note,
        )

    def _explain(self, loaded, notes: list[str]) -> ExplainReport:
        display = loaded.display_rgb
        dh, dw = display.shape[:2]
        gradcam_url = None
        explain_notes: list[str] = []
        if self.cnn is not None:
            cam = self.cnn.gradcam(loaded.pil)
            if cam is not None:
                cam_display = cv2.resize(cam, (dw, dh), interpolation=cv2.INTER_LINEAR)
                gradcam_url = encode_png(heatmap_overlay(display, cam_display, alpha=0.45))
                explain_notes.append("Grad-CAM shows image regions that drove the CNN decision.")
                notes.append("Grad-CAM attention map included.")
        return ExplainReport(
            spectrum=spectrum_data_url(loaded.analysis_rgb),
            preview=encode_png(display),
            gradcam=gradcam_url,
            method="Grad-CAM + spectrum + classical evidence" if gradcam_url else "Spectrum + classical evidence",
            notes=explain_notes,
        )

    # ---------------------------------------------------------------- cache
    def _cache_get(self, key: str) -> AnalysisResult | None:
        with self._cache_lock:
            result = self._cache.get(key)
            if result is not None:
                self._cache.move_to_end(key)
                return result.model_copy(deep=True)
        return None

    def _cache_put(self, key: str, result: AnalysisResult) -> None:
        with self._cache_lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self.settings.cache_size:
                self._cache.popitem(last=False)
