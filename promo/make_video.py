"""Compose the Image Detector promo video from real UI screenshots."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from make_subs import cues


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

from make_subs import cues

HERE = Path(__file__).resolve().parent
SHOTS = HERE / "shots"
OUT = HERE.parent / "docs" / "promo.mp4"
W, H = 1920, 1080
FPS = 30
XFADE = 0.7
STRIP = 104
DH = H - STRIP

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
SEMI = "seguisb.ttf"
REG = "segoeui.ttf"
MONO = "consola.ttf"


def smooth(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def load_shot(name: str) -> Image.Image:
    return Image.open(SHOTS / name).convert("RGB")


def crop_at(img: Image.Image, cx: float, cy: float, width: float) -> Image.Image:
    height = width * DH / W
    x0 = max(0.0, min(img.width - width, cx - width / 2))
    y0 = max(0.0, min(img.height - height, cy - height / 2))
    return img.crop((round(x0), round(y0), round(x0 + width), round(y0 + height))).resize(
        (W, DH), Image.Resampling.LANCZOS
    )


def build_scrim() -> Image.Image:
    alpha = np.zeros((DH, W, 4), dtype=np.uint8)
    alpha[:, :, :3] = INK
    start, ramp_end = DH - 330, DH - 260
    alpha[start:ramp_end, :, 3] = np.linspace(0, 255, ramp_end - start, dtype=np.uint8)[:, None]
    alpha[ramp_end:, :, 3] = 255
    return Image.fromarray(alpha, "RGBA")


SCRIM = build_scrim()


def caption_layer(title: str, sub: str) -> Image.Image:
    layer = Image.new("RGBA", (W, DH), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x, y = 108, DH - 236
    draw.rectangle((x, y, x + 64, y + 5), fill=AMBER + (255,))
    draw.text((x, y + 26), title, font=font(BOLD, 54), fill=WHITE + (255,))
    draw.text((x, y + 102), sub, font=font(REG, 31), fill=MUTED + (255,))
    return layer


CAP_FONT = font(SEMI, 36)


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
    lines = wrap(draw, text, CAP_FONT, W - 420)
    line_height = int(36 * 1.3)
    y = DH + (STRIP - len(lines) * line_height) // 2
    for line in lines:
        width = draw.textlength(line, font=CAP_FONT)
        draw.text(((W - width) / 2, y), line, font=CAP_FONT, fill=WHITE + (255,))
        y += line_height
    return layer


def closing_frame() -> Image.Image:
    base = Image.new("RGB", (W, H), INK)
    draw = ImageDraw.Draw(base)
    glow = Image.new("L", (W, H), 0)
    gd = ImageDraw.Draw(glow)
    gd.ellipse((-700, -900, 1300, 1100), fill=60)
    gd.ellipse((800, 400, 2800, 2400), fill=40)
    glow = glow.filter(ImageFilter.GaussianBlur(260))
    base = Image.composite(Image.new("RGB", (W, H), (20, 23, 27)), base, glow)
    draw = ImageDraw.Draw(base)
    draw.rectangle((108, 300, 108 + 96, 305), fill=AMBER)
    draw.text((108, 336), "Image Detector", font=font(BOLD, 108), fill=WHITE)
    draw.text(
        (112, 498),
        "Detect, localize, attribute.",
        font=font(REG, 46),
        fill=(205, 210, 216),
    )
    draw.text(
        (112, 596),
        "Nine forensic signals, fused into one explainable decision.",
        font=font(REG, 31),
        fill=MUTED,
    )
    draw.text(
        (112, 760),
        "github.com/Krushna7709/image-detector",
        font=font(MONO, 30),
        fill=AMBER,
    )
    draw.text(
        (112, 812),
        "Open source  ·  Python  ·  FastAPI  ·  runs locally, nothing stored",
        font=font(REG, 26),
        fill=(130, 138, 148),
    )
    return base


SCENES = [
    dict(img="01-hero.png", start=(1600, 900, 3120), end=(1580, 860, 3020), dur=5.5,
         title="Every photo tells on itself.",
         sub="Upload an image. Get an evidence-backed AI estimate, not a black box."),
    dict(img="06-samples.png", start=(1600, 1050, 3040), end=(1420, 1230, 2450), dur=4.0,
         title="Start from a sample.",
         sub="Six bundled images: three synthetic, three real camera captures."),
    dict(img="09-readout3x.png", start=(950, 1350, 1920), end=(940, 1370, 1800), dur=7.0,
         title="AI-generated · P(AI) 90%",
         sub="Confidence, reliability, and uncertainty travel with every verdict."),
    dict(img="09-readout3x.png", start=(2970, 620, 1900), end=(2970, 1900, 1900), dur=6.0,
         title="Nine signals, each explained.",
         sub="Noise fields, pixel grids, spectral decay, provenance and more."),
    dict(img="07-viewer.png", start=(1250, 850, 2500), end=(1250, 1180, 2400), dur=6.5,
         title="See where the evidence lives.",
         sub="Edit heatmaps, suspicious regions and an FFT frequency view."),
    dict(img="05-real-verdict.png", start=(820, 1110, 1600), end=(810, 1130, 1520), dur=5.0,
         title="And real photos stay real.",
         sub="Weak evidence is reported as uncertain, never as a confident guess."),
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
    frame = Image.alpha_composite(frame, SCRIM)
    frame = Image.alpha_composite(frame, caption_layer(scene["title"], scene["sub"]))
    canvas = Image.new("RGBA", (W, H), INK + (255,))
    canvas.paste(frame, (0, 0))
    return canvas


def main() -> None:
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        writer = subprocess.Popen(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                "-i", "-", "-an",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                str(OUT),
            ],
            stdin=subprocess.PIPE,
        )
        codec = "libx264"
    else:
        writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        if not writer.isOpened():
            raise SystemExit("no video writer available")
        codec = "mp4v"

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
                local = (now - begin) / scene["dur"]
                frame = scene_frame(scene, local)
                if i + 1 < len(SCENES) and now > finish - XFADE:
                    nxt = scene_frame(SCENES[i + 1], (now - (finish - XFADE)) / SCENES[i + 1]["dur"])
                    frame = Image.blend(frame, nxt, smooth((now - (finish - XFADE)) / XFADE))
                break
        frame = Image.alpha_composite(frame, caption_strip(now))
        rgb = np.array(frame.convert("RGB"))[:, :, ::-1]
        if ffmpeg:
            writer.stdin.write(rgb.tobytes())
        else:
            writer.write(rgb)
        if index % 90 == 0:
            print(f"frame {index}/{frame_count}", flush=True)

    if ffmpeg:
        writer.stdin.close()
        writer.wait()
    else:
        writer.release()
    cap = cv2.VideoCapture(str(OUT))
    ok, probe = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    print(f"wrote {OUT} codec={codec} frames={int(frames)} fps={fps:g} readable={ok}")
    if ok:
        Image.fromarray(probe[:, :, ::-1]).save(HERE / "probe.png")


if __name__ == "__main__":
    main()
