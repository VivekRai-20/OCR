"""Builds synthetic test pages in tests/samples (printed text, table, formula,
rotated and slanted copies)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocrapp.imaging import rotate_free  # noqa: E402

OUT = Path(__file__).parent / "samples"
OUT.mkdir(exist_ok=True)


def font(size, bold=False):
    for f in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def formula_img(tex: str, fs: int = 26) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=fs)
    import io

    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    return Image.open(buf).convert("RGB")


def page() -> Image.Image:
    im = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(im)
    d.text((100, 80), "Quarterly Laboratory Report", font=font(46, True), fill="black")
    body = ("The reaction rate was measured at three temperatures. Results show that the\n"
            "rate constant increases with temperature, as predicted by the Arrhenius law.\n"
            "All measurements were repeated twice and averaged before analysis.")
    d.multiline_text((100, 190), body, font=font(26), fill="black", spacing=12)
    # table
    rows = [["Sample", "Temp (K)", "Rate (mol/s)", "Error"],
            ["A-1", "298", "0.0123", "2.1%"],
            ["A-2", "310", "0.0187", "1.8%"],
            ["A-3", "325", "0.0291", "2.4%"]]
    x0, y0, cw, rh = 100, 380, 250, 56
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            d.rectangle([x0 + j * cw, y0 + i * rh, x0 + (j + 1) * cw, y0 + (i + 1) * rh], outline="black", width=2)
            d.text((x0 + j * cw + 14, y0 + i * rh + 14), c, font=font(24, i == 0), fill="black")
    d.text((100, 650), "The rate constant follows the Arrhenius equation:", font=font(26), fill="black")
    f = formula_img(r"k = A \exp\left(-\frac{E_a}{RT}\right)", 30)
    im.paste(f, (300, 720))
    d.text((100, 900), "Conclusion: the activation energy is about 48 kJ per mole.", font=font(26), fill="black")
    return im


def main() -> None:
    p = page()
    p.save(OUT / "printed_page.png")
    bgr = cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT / "rotated_90.png"), cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE))
    cv2.imwrite(str(OUT / "upside_down.png"), cv2.rotate(bgr, cv2.ROTATE_180))
    cv2.imwrite(str(OUT / "slanted_12deg.png"), rotate_free(bgr, 12))
    cv2.imwrite(str(OUT / "slanted_minus7deg.png"), rotate_free(bgr, -7))
    formula_img(r"\int_0^1 x^2\,dx = \frac{1}{3}", 34).save(OUT / "formula.png")
    print("samples in", OUT)


if __name__ == "__main__":
    main()
