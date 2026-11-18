"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
from pathlib import Path

from aiforensics.config import settings
from aiforensics.preprocessing import ImageValidationError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _write_json(path: str, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _print_result(result, json_path: str | None) -> None:
    v = result.verdict
    label = {
        "ai_generated": "AI-GENERATED",
        "real": "REAL PHOTOGRAPH",
        "uncertain": "UNCERTAIN",
    }[v.label]
    attribution = ", ".join(
        f"{item.generator} ({item.confidence:.0%}, {item.source})" for item in result.attribution
    )
    print(f"File:        {result.filename}")
    print(f"Verdict:     {label}  |  P(AI) = {v.ai_probability:.3f}  confidence {v.confidence:.0%}")
    print(f"Engine:      {result.engine}  (reliability {v.reliability:.0%})")
    print(f"Attribution: {attribution}")
    print(
        f"Edits:       {result.edit.edited_probability:.0%}  "
        f"({len(result.edit.regions)} region(s), {result.edit.method})"
    )
    meta = result.metadata
    print(
        f"Metadata:    {meta.format} {meta.width}x{meta.height}  "
        f"EXIF={meta.has_exif}  C2PA={meta.has_c2pa}  "
        f"generator_hint={meta.generator_hint}"
    )
    print(f"Latency:     {result.elapsed_ms} ms")
    if json_path:
        _write_json(json_path, result.model_dump_json(indent=2))
        print(f"Report:      {json_path}")


def cmd_analyze(args: argparse.Namespace) -> int:
    from aiforensics.core.engine import ForensicsEngine

    path = Path(args.image)
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2
    engine = ForensicsEngine(settings)
    try:
        result = engine.analyze(path.read_bytes(), path.name, args.distortion)
        _print_result(result, args.json)
    except ImageValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from aiforensics.core.engine import ForensicsEngine

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"error: folder not found: {folder}", file=sys.stderr)
        return 2
    if args.limit < 0:
        print("error: --limit must be zero or greater", file=sys.stderr)
        return 2
    files = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("error: no images found", file=sys.stderr)
        return 2
    engine = ForensicsEngine(settings)
    payload = {"settings": {"distortion": args.distortion}, "results": [], "errors": []}
    for path in files:
        try:
            result = engine.analyze(path.read_bytes(), path.name, args.distortion)
        except Exception as exc:
            payload["errors"].append({"file": str(path), "error": str(exc)})
            print(f"{path.name:40s} ERROR {exc}")
            continue
        payload["results"].append(result.model_dump())
        generator = result.attribution[0].generator if result.attribution else "none"
        print(
            f"{path.name:40s} {result.verdict.label:13s} "
            f"p={result.verdict.ai_probability:.3f} "
            f"edit={result.edit.edited_probability:.2f} "
            f"{generator}"
        )
    if args.json:
        try:
            _write_json(args.json, json.dumps(payload, indent=2))
        except OSError as exc:
            print(f"error: cannot write report: {exc}", file=sys.stderr)
            return 2
        print(f"\nWrote {args.json}")
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    from aiforensics.core.engine import ForensicsEngine

    engine = ForensicsEngine(settings)
    print(json.dumps(engine.health().model_dump(), indent=2))
    print(f"model_dir: {settings.model_dir}")
    return 0


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
        return True
    except OSError:
        return False


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    port = args.port if args.port is not None else 8000
    if not 1 <= port <= 65535:
        print("error: --port must be between 1 and 65535", file=sys.stderr)
        return 2
    if not _port_available(args.host, port):
        if args.port is not None:
            print(
                f"error: {args.host}:{port} is already in use. "
                "Stop the other server or choose another port with --port.",
                file=sys.stderr,
            )
            return 2
        for candidate in range(8001, 8011):
            if _port_available(args.host, candidate):
                print(f"Port 8000 is busy; using {candidate} instead.")
                port = candidate
                break
        else:
            print(
                "error: ports 8000-8010 are all in use. Choose one with --port.",
                file=sys.stderr,
            )
            return 2

    print(f"Image Detector {settings.version} -> http://{args.host}:{port}")
    uvicorn.run(
        "aiforensics.api.app:app",
        host=args.host,
        port=port,
        reload=args.reload,
        log_level="info",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiforensics",
        description="Detect, localize and attribute AI-generated images.",
    )
    parser.add_argument("--version", action="version", version=f"Image Detector {settings.version}")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_parser = sub.add_parser("analyze", help="Analyze one image")
    analyze_parser.add_argument("image", help="Path to an image file")
    analyze_parser.add_argument(
        "--distortion", default="none", choices=settings.distortions, help="Simulated distortion"
    )
    analyze_parser.add_argument("--json", help="Write the full result JSON here")
    analyze_parser.add_argument("-v", "--verbose", action="store_true")
    analyze_parser.set_defaults(func=cmd_analyze)

    batch_parser = sub.add_parser("batch", help="Analyze a folder of images")
    batch_parser.add_argument("folder", help="Folder with images")
    batch_parser.add_argument("--distortion", default="none", choices=settings.distortions)
    batch_parser.add_argument("--json", help="Write a combined JSON report")
    batch_parser.add_argument("--limit", type=int, default=0, help="Max images to analyse")
    batch_parser.add_argument("-v", "--verbose", action="store_true")
    batch_parser.set_defaults(func=cmd_batch)

    serve_parser = sub.add_parser("serve", help="Run the API and web UI")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to use (default: 8000, or the next free port if 8000 is busy)",
    )
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.set_defaults(func=cmd_serve)

    info_parser = sub.add_parser("info", help="Show engine capabilities")
    info_parser.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
