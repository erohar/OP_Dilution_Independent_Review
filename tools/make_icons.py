from __future__ import annotations

import io
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "dilution", "assets")
SVG = os.path.join(ASSETS, "logo.svg")
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]
SPLASH_SIZE = (480, 270)

SPLASH_BG, SPLASH_ACCENT = "#1E1E1E", "#DCA91F"
SPLASH_TEXT, SPLASH_MUTED = "#ECEAE7", "#A8A4A0"

FONTS = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
BOLD_FONTS = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf",
              "LiberationSans-Bold.ttf"]

log = logging.getLogger("make_icons")

def render(size: int):
    import cairosvg
    from PIL import Image
    png = cairosvg.svg2png(url=SVG, output_width=size, output_height=size)
    return Image.open(io.BytesIO(png)).convert("RGBA")

def _font(names: list[str], size: int):
    from PIL import ImageFont
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)

def make_splash(path: str) -> str:
    from PIL import Image, ImageDraw
    w, h = SPLASH_SIZE
    img = Image.new("RGB", (w, h), SPLASH_BG)
    logo = render(128)
    img.paste(logo, (36, (h - 128) // 2), logo)
    d = ImageDraw.Draw(img)
    x = 36 + 128 + 28
    d.text((x, 92), "Open Pit", font=_font(BOLD_FONTS, 30), fill=SPLASH_TEXT)
    d.text((x, 128), "SMU Dilution", font=_font(BOLD_FONTS, 30), fill=SPLASH_TEXT)
    d.text((x, 176), "Starting\u2026", font=_font(FONTS, 15), fill=SPLASH_MUTED)
    d.rectangle([0, h - 6, w, h], fill=SPLASH_ACCENT)
    img.save(path, optimize=True)
    return path

def make_icons() -> list[str]:
    png_path = os.path.join(ASSETS, "logo.png")
    ico_path = os.path.join(ASSETS, "logo.ico")
    big = render(256)
    big.save(png_path, optimize=True)

    frames = [render(s) for s in ICO_SIZES]
    frames[-1].save(ico_path, format="ICO", sizes=[(s, s) for s in ICO_SIZES],
                    append_images=frames[:-1])
    splash = make_splash(os.path.join(ASSETS, "splash.png"))
    return [png_path, ico_path, splash]

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        for path in make_icons():
            log.info("Wrote %s", path)
    except ImportError as e:
        log.error("Needs cairosvg and pillow: pip install cairosvg pillow (%s)", e)
        return 1
    except OSError as e:
        log.error("Could not make the icons: %s", e)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
