"""Metadata and provenance extraction.

These signals are intentionally conservative: metadata is trivially stripped
or spoofed, so it never decides a verdict on its own and every observation is
reported with its limitations.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ExifTags

from aiforensics.preprocessing import LoadedImage
from aiforensics.schemas import MetadataReport

# Annex K luminance quantisation table, used to estimate JPEG quality.
_STD_LUMA = np.array(
    [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ],
    dtype=np.float64,
)

GENERATOR_PATTERNS: dict[str, tuple[str, ...]] = {
    "Midjourney": ("midjourney",),
    "DALL-E": ("dall-e", "dall·e", "dall e"),
    "OpenAI ImageGen": ("openai imagegen", "azure openai imagegen", "openai image generation"),
    "Stable Diffusion": (
        "stable diffusion",
        "stablediffusion",
        "dreamstudio",
        "automatic1111",
        "comfyui",
        "invoke ai",
    ),
    "Adobe Firefly": ("adobe firefly", "firefly image"),
    "FLUX": ("flux.1", "flux 1", "black forest labs", "blackforestlabs"),
    "Leonardo AI": ("leonardo.ai", "leonardo ai"),
    "Ideogram": ("ideogram",),
    "Google Imagen": ("imagen 3", "imagen-3", "google imagen"),
    "Recraft": ("recraft",),
    "Playground AI": ("playground ai", "playground.ai"),
}

HINT_KEYS = ("software", "comment", "description", "parameters", "prompt", "author", "creator")
WATERMARK_MARKERS = ("synthid", "digimarc")


@dataclass
class MetadataBundle:
    report: MetadataReport
    generator_hint: str | None
    synthetic_provenance: bool = False


def _estimate_jpeg_quality(im: Image.Image) -> int | None:
    try:
        tables = im.quantization
    except Exception:
        return None
    if not tables:
        return None
    actual = np.sort(np.array(tables.get(0, tables[sorted(tables)[0]]), dtype=np.float64))
    if actual.size != 64:
        return None

    best_q, best_err = None, float("inf")
    for q in range(10, 101):
        scale = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
        candidate = np.clip(np.floor((_STD_LUMA * scale + 50.0) / 100.0), 1, 255)
        err = float(np.abs(np.sort(candidate) - actual).mean())
        if err < best_err:
            best_q, best_err = q, err
    return best_q


def _exif_fields(pil: Image.Image) -> dict[str, str]:
    fields: dict[str, str] = {}
    try:
        exif = pil.getexif()
    except Exception:
        return fields
    if not exif:
        return fields

    for tag_id, value in exif.items():
        name = ExifTags.TAGS.get(tag_id, str(tag_id))
        fields[name] = str(value)
    try:
        sub = exif.get_ifd(0x8769)  # ExifIFD
        for tag_id, value in sub.items():
            name = ExifTags.TAGS.get(tag_id, str(tag_id))
            fields.setdefault(name, str(value))
    except Exception:
        pass
    return fields


def _png_text(pil: Image.Image) -> dict[str, str]:
    text = getattr(pil, "text", None)
    if isinstance(text, dict):
        return {str(k): str(v) for k, v in text.items()}
    info = pil.info or {}
    return {k: str(v) for k, v in info.items() if isinstance(v, str) and k in HINT_KEYS}


def _find_generator(haystack: str) -> str | None:
    lowered = haystack.casefold()
    for generator, patterns in GENERATOR_PATTERNS.items():
        if any(p in lowered for p in patterns):
            return generator
    return None


def _normalize_agent(agent: str) -> str:
    lowered = agent.casefold()
    if "azure openai" in lowered or "openai imagegen" in lowered:
        return "OpenAI ImageGen"
    return agent.strip()


def _c2pa_details(data: bytes) -> tuple[str | None, str | None]:
    """Best-effort read of the C2PA claim fields (no signature verification)."""

    text = data[:1_500_000].decode("latin-1", errors="ignore")
    agent = None
    match = re.search(r"softwareAgent.{0,60}?name[\x00-\x20]?([\x20-\x7e]{3,80})", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        # A CBOR text-length byte can leak in front of the value.
        if len(candidate) > 1 and candidate[0].islower() and candidate[1].isupper():
            candidate = candidate[1:]
        agent = candidate.strip("\x00 ")

    source_type = None
    if "compositeWithTrainedAlgorithmicMedia" in text:
        source_type = "compositeWithTrainedAlgorithmicMedia"
    elif "trainedAlgorithmicMedia" in text:
        source_type = "trainedAlgorithmicMedia"
    return agent, source_type


def extract_metadata(data: bytes, loaded: LoadedImage) -> MetadataBundle:
    notes: list[str] = []
    pil = Image.open(io.BytesIO(data))
    exif = _exif_fields(pil)

    make = exif.get("Make", "").strip()
    model = exif.get("Model", "").strip()
    software = exif.get("Software", "").strip() or None
    captured = (
        exif.get("DateTimeOriginal")
        or exif.get("DateTimeDigitized")
        or exif.get("DateTime")
    )
    camera = " ".join(part for part in (make, model) if part) or None
    has_exif = bool(exif)

    png_text = _png_text(pil)
    png_text_keys = sorted(png_text.keys())[:24]

    raw_lower = data[:400_000].lower()
    has_c2pa = (b"c2pa" in raw_lower) or (b"jumbf" in raw_lower)
    has_xmp = (b"<x:xmpmeta" in raw_lower) or (b"xmpmeta" in raw_lower)
    c2pa_agent, c2pa_source_type = _c2pa_details(data) if has_c2pa else (None, None)
    synthetic_provenance = c2pa_source_type is not None

    watermark = next((m for m in WATERMARK_MARKERS if m.encode() in raw_lower), None)

    searchable = " ".join(
        [software or "", camera or "", c2pa_agent or ""] + [f"{k}: {v}" for k, v in png_text.items()]
    )
    try:
        searchable += " " + data[:200_000].decode("latin-1", errors="ignore")
    except Exception:
        pass
    generator_hint = _find_generator(searchable)
    if synthetic_provenance and c2pa_agent:
        generator_hint = _find_generator(c2pa_agent) or _normalize_agent(c2pa_agent)

    jpeg_quality = _estimate_jpeg_quality(pil) if loaded.format == "JPEG" else None

    if has_c2pa:
        notes.append(
            "C2PA / Content Credentials markers were found. The signature chain is not "
            "cryptographically verified by this tool."
        )
    if c2pa_agent:
        notes.append(f"C2PA software agent: {c2pa_agent}.")
    if synthetic_provenance:
        notes.append(
            f"C2PA digitalSourceType is '{c2pa_source_type}': the manifest claims synthetic origin."
        )
    if not has_exif:
        notes.append(
            "No EXIF metadata. This is common for AI generators and social-media downloads alike, "
            "so it is weak evidence."
        )
    if software:
        notes.append(f"Editing/authoring software reported by EXIF: {software}.")
    if generator_hint:
        notes.append(
            f"Metadata mentions '{generator_hint}'. Metadata can be spoofed; treat this as a lead, not proof."
        )
    if watermark:
        notes.append(
            f"A '{watermark}' watermark marker was found in the file bytes. Marker presence is not validated."
        )
    if jpeg_quality is not None and jpeg_quality <= 60:
        notes.append(
            f"Estimated JPEG quality is ~{jpeg_quality}; recompression erases fine high-frequency evidence."
        )
    if has_xmp:
        notes.append("XMP packet present (may contain editing history or generator settings).")

    report = MetadataReport(
        format=loaded.format,
        width=loaded.width,
        height=loaded.height,
        file_size_bytes=loaded.file_size,
        has_exif=has_exif,
        camera=camera,
        software=software,
        captured_at=captured,
        jpeg_quality_estimate=jpeg_quality,
        has_c2pa=has_c2pa,
        c2pa_agent=c2pa_agent,
        c2pa_source_type=c2pa_source_type,
        synthetic_provenance=synthetic_provenance,
        has_xmp=has_xmp,
        png_text_keys=png_text_keys,
        generator_hint=generator_hint,
        watermark_marker=watermark,
        notes=notes,
    )
    return MetadataBundle(
        report=report,
        generator_hint=generator_hint,
        synthetic_provenance=synthetic_provenance,
    )
