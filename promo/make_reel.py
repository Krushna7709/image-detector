"""Compose the vertical 9:16 reel from phone-viewport screenshots, with promo narration."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from make_subs import cues

HERE = Path(__file__).resolve().parent
SHOTS = HERE / "shots"
SILENT = HERE / "reel_silent.mp4"
AUDIO_SOURCE = HERE.parent / "docs" / "promo.mp4"
OUT = HERE.parent / "docs" / "reel.mp4"
W, H = 1080, 1920
FPS = 30
XFADE = 0.7

CAPTIONS = cues()

INK = (12, 14, 17)
WHITE = (238, 240, 243)
MUTED = (160, 168, 178)
AMBER = (246, 197, 83)

FONT_DIR = Path("C:/Windows/Fonts")


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for candidate in (name, "segoeui.ttf", "arial.ttf"):
        path = FONT_DIR / candidate
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


BOLD = "segoeuib.ttf"
REG = "segoeui.ttf"
MONO = "consola.ttf"


def smooth(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def load_shot(name: str) -> Image.Image:
    return Image.open(SHOTS / name).convert("RGB")


def crop_at(img: Image.Image, cx: float, cy: float, width: float) -> Image.Image:
    height = width * H / W
    x0 = max(0.0, min(img.width - width, cx - width / 2))
    y0 = max(0.0, min(img.height - height, cy - height / 2))
    return img.crop((round(x0), round(y0), round(x0 + width), round(y0 + height))).resize(
        (W, H), Image.Resampling.LANCZOS
    )


def build_scrim() -> Image.Image:
    alpha = np.zeros((H, W, 4), dtype=np.uint8)
    alpha[:, :, :3] = INK
    start, ramp_end = H - 790, H - 700
    alpha[start:ramp_end, :, 3] = np.linspace(0, 255, ramp_end - start, dtype=np.uint8)[:, None]
    alpha[ramp_end:, :, 3] = 255
    return Image.fromarray(alpha, "RGBA")


def build_top() -> Image.Image:
    alpha = np.zeros((H, W, 4), dtype=np.uint8)
    alpha[:, :, :3] = INK
    alpha[:120, :, 3] = np.linspace(190, 0, 120, dtype=np.uint8)[:, None]
    return Image.fromarray(alpha, "RGBA")


SCRIM = build_scrim()
TOP = build_top()


def caption_layer(title: str, sub: str) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x, y = 72, H - 660
    draw.rectangle((x, y, x + 72, y + 6), fill=AMBER + (255,))
    draw.text((x, y + 34), title, font=font(BOLD, 66), fill=WHITE + (255,))
    draw.text((x, y + 124), sub, font=font(REG, 36), fill=MUTED + (255,))
    return layer


CAP_FONT = font(REG, 38)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: float) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if not line or draw.textlength(trial, font=fnt) <= max_width:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def caption_strip(now: float) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    text = next((text for start, end, text in CAPTIONS if start <= now < end), "")
    if not text:
        return layer
    draw = ImageDraw.Draw(layer)
    lines = wrap(draw, text, CAP_FONT, W - 140)
    line_height = 50
    y = H - 340 - len(lines) * line_height
    for line in lines:
        width = draw.textlength(line, font=CAP_FONT)
        draw.text(((W - width) / 2, y), line, font=CAP_FONT, fill=WHITE + (255,))
        y += line_height
    return layer


def closing_frame() -> Image.Image:
    base = Image.new("RGB", (W, H), INK)
    glow = Image.new("L", (W, H), 0)
    gd = ImageDraw.Draw(glow)
    gd.ellipse((-900, -600, 900, 1200), fill=60)
    gd.ellipse((300, 900, 2100, 2700), fill=40)
    glow = glow.filter(ImageFilter.GaussianBlur(300))
    base = Image.composite(Image.new("RGB", (W, H), (20, 23, 27)), base, glow)
    draw = ImageDraw.Draw(base)
    draw.rectangle((72, 620, 72 + 96, 626), fill=AMBER)
    draw.text((72, 664), "Image", font=font(BOLD, 132), fill=WHITE)
    draw.text((72, 812), "Detector", font=font(BOLD, 132), fill=WHITE)
    draw.text((76, 994), "Detect, localize, attribute.", font=font(REG, 54), fill=(205, 210, 216))
    draw.text(
        (76, 1082),
        "Nine forensic signals, fused into\none explainable decision.",
        font=font(REG, 38),
        fill=MUTED,
        spacing=14,
    )
    draw.text(
        (76, 1210),
        "github.com/Krushna7709/\nimage-detector",
        font=font(MONO, 36),
        fill=AMBER,
        spacing=12,
    )
    draw.text(
        (76, 1330),
        "Open source · Python · FastAPI\nRuns locally, nothing stored",
        font=font(REG, 30),
        fill=(130, 138, 148),
        spacing=12,
    )
    return base


SCENES = [
    dict(img="r01-hero.png", start=(645, 1100, 1240), end=(645, 1400, 1240), dur=5.5,
         title="Can you tell which is fake?",
         sub="Nine forensic checks run in seconds."),
    dict(img="r02-samples.png", start=(645, 1300, 1290), end=(645, 1420, 1080), dur=4.0,
         title="Three synthetic. Three real.",
         sub="Tap a sample, or upload your own."),
    dict(img="r03-verdict.png", start=(645, 1750, 1150), end=(645, 1720, 1100), dur=7.0,
         title="90% AI-generated",
         sub="Confidence, reliability and uncertainty, every time."),
    dict(img="r04-signals.png", start=(645, 1200, 1180), end=(645, 1800, 1120), dur=6.0,
         title="Nothing hidden in a black box",
         sub="Noise fields, pixel grids, spectral decay, provenance."),
    dict(img="r06-viewer.png", start=(645, 1150, 1180), end=(645, 1500, 1120), dur=6.5,
         title="See where the evidence lives",
         sub="Localized heatmaps, suspicious regions, FFT spectrum."),
    dict(img="r08-real.png", start=(645, 1750, 1200), end=(645, 1730, 1120), dur=5.0,
         title="Real photos stay real",
         sub="Weak evidence is reported as uncertain."),
    dict(closing=True, dur=6.0),
]


def scene_frame(scene: dict, t: float) -> Image.Image:
    if scene.get("closing"):
        return closing_frame().convert("RGBA")
    img = load_shot(scene["img"])
    e = smooth(t)
    cx0, cy0, w0 = scene["start"]
    cx1, cy1, w1 = scene["end"]
    frame = crop_at(img, cx0 + (cx1 - cx0) * e, cy0 + (cy1 - cy0) * e, w0 + (w1 - w0) * e)
    frame = frame.convert("RGBA")
    frame = Image.alpha_composite(frame, TOP)
    frame = Image.alpha_composite(frame, SCRIM)
    return Image.alpha_composite(frame, caption_layer(scene["title"], scene["sub"]))


def ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> None:
    encoder = subprocess.Popen(
        [
            ffmpeg(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
            "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv",
            str(SILENT),
        ],
        stdin=subprocess.PIPE,
    )

    starts = []
    clock = 0.0
    for scene in SCENES:
        starts.append(clock)
        clock += scene["dur"] - XFADE
    total = clock + XFADE
    frame_count = int(total * FPS)

    for index in range(frame_count):
        now = index / FPS
        for i, scene in enumerate(SCENES):
            begin = starts[i]
            finish = begin + scene["dur"]
            if begin <= now < finish or (i == len(SCENES) - 1 and now >= begin):
                frame = scene_frame(scene, (now - begin) / scene["dur"])
                if i + 1 < len(SCENES) and now > finish - XFADE:
                    nxt = scene_frame(SCENES[i + 1], (now - (finish - XFADE)) / SCENES[i + 1]["dur"])
                    frame = Image.blend(frame, nxt, smooth((now - (finish - XFADE)) / XFADE))
                break
        frame = Image.alpha_composite(frame, caption_strip(now))
        rgb = np.array(frame.convert("RGB"))[:, :, ::-1]
        encoder.stdin.write(rgb.tobytes())
        if index % 120 == 0:
            print(f"frame {index}/{frame_count}", flush=True)

    encoder.stdin.close()
    encoder.wait()

    subprocess.run(
        [
            ffmpeg(), "-y", "-loglevel", "error",
            "-i", str(SILENT), "-i", str(AUDIO_SOURCE),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "copy",
            "-movflags", "+faststart",
            str(OUT),
        ],
        check=True,
    )
    SILENT.unlink()
    info = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(OUT)], capture_output=True, text=True)
    for line in info.stderr.splitlines():
        if "Duration" in line or "Audio:" in line or "Video:" in line:
            print(line.strip())
    print(f"reel: {OUT}")


if __name__ == "__main__":
    main()
