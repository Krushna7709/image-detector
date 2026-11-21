"""Write SRT/VTT captions from the narration and mux them into both cuts.

The visible captions are drawn into the frame by make_video.py and make_reel.py;
this embeds the soft subtitle track browsers can expose and writes the sidecars.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from voiceover import LINES, ffmpeg

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
SRT = HERE / "promo.srt"
VTT = HERE / "promo.vtt"


def clip_length(path: Path) -> float:
    info = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", info.stderr)
    if not match:
        raise SystemExit(f"cannot read duration of {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def stamp(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def cues() -> list[tuple[float, float, str]]:
    starts = [float(start) for start, _ in LINES]
    found = []
    for index, (start, text) in enumerate(LINES, start=1):
        start = float(start)
        end = start + clip_length(HERE / "voice" / f"line{index}.mp3") + 0.35
        if index < len(LINES):
            end = min(end, starts[index] - 0.05)
        found.append((start, end, text))
    return found


def write_subs(items: list[tuple[float, float, str]]) -> None:
    srt = []
    for number, (start, end, text) in enumerate(items, start=1):
        srt.append(f"{number}\n{stamp(start, ',')} --> {stamp(end, ',')}\n{text}\n")
    SRT.write_text("\n".join(srt), encoding="utf-8")

    vtt = ["WEBVTT\n"]
    for number, (start, end, text) in enumerate(items, start=1):
        vtt.append(f"{number}\n{stamp(start, '.')} --> {stamp(end, '.')}\n{text}\n")
    VTT.write_text("\n".join(vtt), encoding="utf-8")


def has_subtitle_track(video: Path) -> bool:
    info = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(video)], capture_output=True, text=True)
    return "Subtitle:" in info.stderr


def mux(video: Path) -> None:
    if has_subtitle_track(video):
        return
    out = video.with_name(video.stem + ".subbed.mp4")
    subprocess.run(
        [
            ffmpeg(), "-y", "-loglevel", "error",
            "-i", str(video), "-i", str(SRT),
            "-map", "0:v:0", "-map", "0:a:0", "-map", "1:0",
            "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
            "-disposition:s:0", "0",
            "-metadata:s:s:0", "language=eng",
            "-metadata:s:s:0", "title=English",
            "-movflags", "+faststart",
            str(out),
        ],
        check=True,
    )
    out.replace(video)


def main() -> None:
    items = cues()
    write_subs(items)
    for start, end, text in items:
        print(f"{stamp(start, ',')} --> {stamp(end, ',')}  {text}")
    for name in ("promo.mp4", "reel.mp4"):
        video = DOCS / name
        mux(video)
        info = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(video)], capture_output=True, text=True)
        streams = [line.strip() for line in info.stderr.splitlines() if "Stream #" in line]
        print(name, "->", "; ".join(streams))
    print(f"subtitles: {SRT.name}, {VTT.name}")


if __name__ == "__main__":
    main()
