"""Pydantic models shared by the API, CLI and UI."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

VerdictLabel = Literal["real", "ai_generated", "uncertain"]


class Signal(BaseModel):
    """One piece of forensic evidence feeding the decision."""

    name: str
    label: str
    score: float = Field(ge=0, le=1, description="0 = real-like, 1 = AI-like")
    weight: float = Field(ge=0, le=1)
    reliability: float = Field(ge=0, le=1)
    detail: str


class Verdict(BaseModel):
    label: VerdictLabel
    ai_probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    reliability: float = Field(ge=0, le=1)
    threshold: float
    uncertain_range: tuple[float, float]
    engine: str
    calibrated: bool


class AttributionItem(BaseModel):
    generator: str
    confidence: float = Field(ge=0, le=1)
    source: Literal["metadata", "model", "none"]
    known: bool


class EditRegion(BaseModel):
    x: int
    y: int
    width: int
    height: int
    score: float = Field(ge=0, le=1)


class EditReport(BaseModel):
    edited_probability: float = Field(ge=0, le=1)
    regions: list[EditRegion]
    heatmap: str = Field(description="Data URL with the smooth heatmap overlay")
    boxes: str = Field(description="Data URL with transparent bounding boxes")
    method: Literal["heuristic", "unet"]
    note: str


class MetadataReport(BaseModel):
    format: str
    width: int
    height: int
    file_size_bytes: int
    has_exif: bool
    camera: str | None = None
    software: str | None = None
    captured_at: str | None = None
    jpeg_quality_estimate: int | None = None
    has_c2pa: bool = False
    c2pa_agent: str | None = None
    c2pa_source_type: str | None = None
    synthetic_provenance: bool = False
    has_xmp: bool = False
    png_text_keys: list[str] = Field(default_factory=list)
    generator_hint: str | None = None
    watermark_marker: str | None = None
    notes: list[str] = Field(default_factory=list)


class ExplainReport(BaseModel):
    spectrum: str = Field(description="Data URL of the frequency spectrum")
    preview: str = Field(description="Data URL of the analysed image")
    gradcam: str | None = None
    method: str
    notes: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    id: str
    filename: str
    sha256: str
    created_at: str
    elapsed_ms: int
    engine: str
    distortion: str
    verdict: Verdict
    attribution: list[AttributionItem]
    edit: EditReport
    metadata: MetadataReport
    explain: ExplainReport
    signals: list[Signal]
    notes: list[str]


class BatchResult(BaseModel):
    results: list[AnalysisResult]
    errors: list[dict[str, str]] = Field(default_factory=list)
    elapsed_ms: int


class EngineHealth(BaseModel):
    status: Literal["ok"]
    version: str
    detector: str
    attribution: bool
    localizer: str
    clip_backbone: str
    clip_loaded: bool
    uptime_seconds: int


class ExampleImage(BaseModel):
    name: str
    label: str
    kind: Literal["ai", "real"]
    url: str
