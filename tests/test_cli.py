from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from aiforensics.__main__ import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = PROJECT_ROOT / "src" / "aiforensics" / "api" / "examples"


def test_analyze_command(tmp_path, image_factory, capsys):
    path = tmp_path / "photo.png"
    path.write_bytes(image_factory())
    report = tmp_path / "report.json"

    assert main(["analyze", str(path), "--json", str(report)]) == 0
    output = capsys.readouterr().out
    assert "Verdict:" in output
    assert "Metadata:    PNG" in output

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["filename"] == "photo.png"
    assert payload["verdict"]["label"] in {"real", "uncertain", "ai_generated"}


def test_analyze_missing_file(capsys):
    assert main(["analyze", "does-not-exist.jpg"]) == 2
    assert "file not found" in capsys.readouterr().err


def test_analyze_corrupt_file_reports_error(tmp_path, capsys):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    assert main(["analyze", str(path)]) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_analyze_rejects_directory(tmp_path, capsys):
    assert main(["analyze", str(tmp_path)]) == 2
    assert "file not found" in capsys.readouterr().err


def test_analyze_creates_json_parent_directories(tmp_path, image_factory):
    path = tmp_path / "photo.png"
    path.write_bytes(image_factory())
    report = tmp_path / "reports" / "nested" / "report.json"
    assert main(["analyze", str(path), "--json", str(report)]) == 0
    assert report.is_file()


def test_batch_command(tmp_path, image_factory, capsys):
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "one.png").write_bytes(image_factory(seed=1))
    (folder / "two.png").write_bytes(image_factory(seed=2))
    (folder / "notes.txt").write_text("ignore me", encoding="utf-8")
    report = tmp_path / "batch.json"

    assert main(["batch", str(folder), "--json", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert len(payload["results"]) == 2
    assert payload["errors"] == []


def test_batch_handles_real_verdict_without_attribution(tmp_path, capsys):
    folder = tmp_path / "images"
    folder.mkdir()
    shutil.copy(EXAMPLES / "real_mountain_lake.jpg", folder / "real.jpg")
    report = tmp_path / "batch.json"

    assert main(["batch", str(folder), "--json", str(report)]) == 0
    output = capsys.readouterr().out
    assert "real" in output
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["errors"] == []
    assert payload["results"][0]["attribution"] == []


def test_batch_empty_folder(tmp_path, capsys):
    folder = tmp_path / "empty"
    folder.mkdir()
    assert main(["batch", str(folder)]) == 2
    assert "no images" in capsys.readouterr().err


def test_batch_rejects_negative_limit(tmp_path, image_factory, capsys):
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "one.png").write_bytes(image_factory())
    assert main(["batch", str(folder), "--limit", "-1"]) == 2
    assert "--limit" in capsys.readouterr().err


def test_serve_rejects_invalid_port(capsys):
    assert main(["serve", "--port", "0"]) == 2
    assert "between 1 and 65535" in capsys.readouterr().err


def test_info_command(capsys):
    assert main(["info"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output.split("\nmodel_dir:", 1)[0])
    assert payload["status"] == "ok"
    assert payload["detector"] in {"heuristic", "clip-linear", "cnn"}


def test_module_entry_point(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "aiforensics", "--version"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert "ForensIQ" in result.stdout
