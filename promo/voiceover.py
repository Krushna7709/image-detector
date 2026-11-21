"""Generate narration lines and mux them onto the promo video."""

from __future__ import annotations

import asyncio
import subprocess
import wave
from pathlib import Path

import edge_tts

HERE = Path(__file__).resolve().parent
VIDEO = HERE.parent / "docs" / "promo.mp4"
WORK = HERE / "voice"
VOICE = "en-US-AndrewNeural"
RATE = "+6%"

# (start seconds, line) - each line lands inside its scene window.
LINES = [
    (0.4, "One of these pictures has never existed. Can you tell which?"),
    (5.0, "Start with a sample. Or drop in your own."),
    (8.5, "Nine forensic checks run in seconds. Here: ninety percent AI-generated."),
    (14.9, "Noise fields. Pixel grids. Spectral decay. Provenance. Every signal, explained."),
    (21.0, "And you can see exactly where the evidence lives."),
    (25.9, "Real photos stay real. Weak evidence stays uncertain."),
    (30.1, "Image Detector. Detect, localize, attribute. Open source on GitHub."),
]


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def duration(path: Path) -> float:
    wav = path.with_suffix(".wav")
    subprocess.run(
        [ffmpeg(), "-y", "-loglevel", "error", "-i", str(path), "-ar", "48000", "-ac", "1", str(wav)],
        check=True,
    )
    with wave.open(str(wav)) as handle:
        return handle.getnframes() / handle.getframerate()


async def synth(index: int, text: str) -> Path:
    out = WORK / f"line{index}.mp3"
    for attempt in range(4):
        try:
            await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(out))
            return out
        except edge_tts.exceptions.NoAudioReceived:
            if attempt == 3:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))
    return out


def main() -> None:
    WORK.mkdir(exist_ok=True)
    clips = []
    for index, (start, text) in enumerate(LINES, start=1):
        clip = asyncio.run(synth(index, text))
        length = duration(clip)
        clips.append((start, clip, length))
        print(f"line {index}: start {start:5.1f}s  length {length:4.1f}s  end {start + length:5.1f}s")

    video_length = 35.8
    overrun = [i for i, (start, _, length) in enumerate(clips, 1) if start + length > video_length]
    if overrun:
        raise SystemExit(f"lines run past the video: {overrun}")

    args = [ffmpeg(), "-y", "-loglevel", "error", "-i", str(VIDEO)]
    for _, clip, _ in clips:
        args += ["-i", str(clip)]
    filters = []
    labels = []
    for index, (start, _, _) in enumerate(clips, start=1):
        delay = int(round(start * 1000))
        filters.append(f"[{index}:a]aresample=48000,adelay={delay}|{delay}[a{index}]")
        labels.append(f"[a{index}]")
    chain = (
        "".join(labels)
        + f"amix=inputs={len(clips)}:duration=longest:normalize=0,"
        + "loudnorm=I=-15:TP=-1.5:LRA=11,alimiter=limit=0.95,aresample=48000[aout]"
    )
    filters.append(chain)
    out = VIDEO.with_name("promo-voiced.mp4")
    args += [
        "-filter_complex", ";".join(filters),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(args, check=True)
    out.replace(VIDEO)
    info = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(VIDEO)], capture_output=True, text=True)
    for line in info.stderr.splitlines():
        if "Duration" in line or "Audio:" in line or "Video:" in line:
            print(line.strip())
    print(f"voiced video: {VIDEO}")


if __name__ == "__main__":
    main()
