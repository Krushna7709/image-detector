"""FastAPI application serving the JSON API and the web UI."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from aiforensics.config import settings
from aiforensics.core.engine import ForensicsEngine
from aiforensics.preprocessing import ImageValidationError
from aiforensics.schemas import (
    AnalysisResult,
    BatchResult,
    EngineHealth,
    ExampleImage,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
EXAMPLES_DIR = BASE_DIR / "examples"

EXAMPLES: list[dict[str, str]] = [
    {
        "name": "ai_mountain",
        "file": "ai_mountain.png",
        "label": "Synthetic landscape · C2PA",
        "kind": "ai",
    },
    {
        "name": "ai_portrait",
        "file": "ai_portrait.png",
        "label": "Synthetic portrait · C2PA",
        "kind": "ai",
    },
    {
        "name": "ai_product",
        "file": "ai_product.png",
        "label": "Synthetic product shot · C2PA",
        "kind": "ai",
    },
    {
        "name": "real_mountain_lake",
        "file": "real_mountain_lake.jpg",
        "label": "Camera photo · mountain lake",
        "kind": "real",
    },
    {
        "name": "real_cat_portrait",
        "file": "real_cat_portrait.jpg",
        "label": "Camera photo · cat portrait",
        "kind": "real",
    },
    {
        "name": "real_street_market",
        "file": "real_street_market.jpg",
        "label": "Camera photo · street market",
        "kind": "real",
    },
]

EXAMPLES_BY_NAME = {entry["name"]: entry for entry in EXAMPLES}
MAX_BATCH_FILES = 8

_ANALYSIS_SEMAPHORE = asyncio.Semaphore(2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = ForensicsEngine(settings)
    logger.info(
        "ForensIQ started | detector=%s attribution=%s localizer=%s",
        app.state.engine.detector_mode,
        app.state.engine.attribution is not None,
        "unet" if app.state.engine.unet else "heuristic",
    )
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=settings.description,
    lifespan=lifespan,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(ImageValidationError)
async def image_validation_handler(request: Request, exc: ImageValidationError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.code, "message": str(exc)},
    )


def _engine(request: Request) -> ForensicsEngine:
    return request.app.state.engine


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise ImageValidationError(
            f"File is larger than the {settings.max_upload_bytes // (1024 * 1024)} MB limit.",
            status_code=413,
            code="file_too_large",
        )
    return data


def _check_distortion(distortion: str) -> str:
    if distortion not in settings.distortions:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown distortion '{distortion}'. Choose from: {', '.join(settings.distortions)}",
        )
    return distortion


@app.get("/api/health", response_model=EngineHealth, tags=["system"])
async def health(request: Request) -> EngineHealth:
    return _engine(request).health()


@app.get("/api/examples", response_model=list[ExampleImage], tags=["system"])
async def list_examples(request: Request) -> list[ExampleImage]:
    base = str(request.base_url).rstrip("/")
    return [
        ExampleImage(
            name=entry["name"],
            label=entry["label"],
            kind=entry["kind"],
            url=f"{base}/examples/{entry['file']}",
        )
        for entry in EXAMPLES
    ]


@app.post("/api/analyze", response_model=AnalysisResult, tags=["analysis"])
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    distortion: str = Form("none"),
) -> AnalysisResult:
    distortion = _check_distortion(distortion)
    data = await _read_upload(file)
    engine = _engine(request)
    async with _ANALYSIS_SEMAPHORE:
        return await run_in_threadpool(
            engine.analyze, data, file.filename or "upload", distortion
        )


@app.post("/api/analyze/example/{name}", response_model=AnalysisResult, tags=["analysis"])
async def analyze_example(
    request: Request,
    name: str,
    distortion: str = Form("none"),
) -> AnalysisResult:
    distortion = _check_distortion(distortion)
    entry = EXAMPLES_BY_NAME.get(name)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown example image")
    path = EXAMPLES_DIR / entry["file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Example image is missing")
    data = await run_in_threadpool(path.read_bytes)
    engine = _engine(request)
    async with _ANALYSIS_SEMAPHORE:
        return await run_in_threadpool(engine.analyze, data, path.name, distortion)


@app.post("/api/analyze/batch", response_model=BatchResult, tags=["analysis"])
async def analyze_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    distortion: str = Form("none"),
) -> BatchResult:
    distortion = _check_distortion(distortion)
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400, detail=f"Batch limit is {MAX_BATCH_FILES} images per request"
        )
    started = time.perf_counter()
    engine = _engine(request)
    results: list[AnalysisResult] = []
    errors: list[dict[str, str]] = []

    for upload in files:
        try:
            data = await _read_upload(upload)
            async with _ANALYSIS_SEMAPHORE:
                result = await run_in_threadpool(
                    engine.analyze, data, upload.filename or "upload", distortion
                )
            results.append(result)
        except ImageValidationError as exc:
            errors.append(
                {"filename": upload.filename or "upload", "error": str(exc), "code": exc.code}
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.exception("Batch item failed")
            errors.append(
                {"filename": upload.filename or "upload", "error": str(exc), "code": "internal_error"}
            )

    return BatchResult(
        results=results,
        errors=errors,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


@app.get("/api/examples/{name}", include_in_schema=False)
async def example_file(name: str) -> FileResponse:
    entry = EXAMPLES_BY_NAME.get(name)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown example image")
    path = EXAMPLES_DIR / entry["file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Example image is missing")
    return FileResponse(path)


app.mount("/examples", StaticFiles(directory=EXAMPLES_DIR), name="examples")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
