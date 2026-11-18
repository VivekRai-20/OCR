"""Fills the experiment data sheet with handwriting-style entries (random
handwriting fonts, some cells left blank, notes written beside the table) and
writes a ground-truth JSON of what each cell holds.

    python tests/make_table_samples.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocrapp import grid_tables, imaging  # noqa: E402

DIR = Path(__file__).parent / "tables"
FONTS = ["Inkfree.ttf", "segoepr.ttf", "LHANDW.TTF", "BRADHITC.TTF", "mvboli.ttf"]


def hand_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(random.choice(FONTS), size)


def main() -> None:
    random.seed(7)
    base = imaging.imread(DIR / "sheet_p1.png")
    t = grid_tables.find_tables(base)[0]
    pil = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    truth: dict[str, str] = {}
    ink = (20, 30, 120)  # blue pen
    # data rows start below the header (rows 7..27 in this sheet)
    for cell in t.cells:
        x1, y1, x2, y2 = cell.box
        if not (7 <= cell.r <= 26) or cell.rowspan != 1:
            continue
        if random.random() < 0.35:
            continue  # blank cell
        if cell.c == 13:
            val = random.choice(["ok", "stable", "leak checked", "purged", "drop"])
        elif cell.c == 0:
            val = str(cell.r - 6)
        else:
            val = f"{random.uniform(0.1, 999):.{random.choice([0, 1, 2])}f}"
        f = hand_font(int((y2 - y1) * random.uniform(0.6, 0.8)))
        d.text((x1 + random.randint(4, 20), y1 + random.randint(-2, 4)), val, font=f, fill=ink)
        truth[f"{cell.r},{cell.c}"] = val
    # header fields filled by hand
    for (r, c), val in {(0, 2): "UC-07", (0, 4): "Run 3", (1, 13): "MM-12", (2, 13): "5 A", (3, 2): "99.9%"}.items():
        cell = next(x for x in t.cells if x.r == r and x.c == c)
        x1, y1, x2, y2 = cell.box
        d.text((x1 + 8, y1 - 2), val, font=hand_font(int((y2 - y1) * 0.75)), fill=ink)
        truth[f"{r},{c}"] = val
    # a note written beside the table (outside it)
    d.text((t.bbox[0] + 40, t.bbox[3] + 40), "Note: cell flooded after 6 hrs, restarted", font=hand_font(46), fill=ink)
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(DIR / "sheet_filled.png"), img)
    # a phone-photo-like copy: slanted 3 degrees, slightly blurred
    sl = imaging.rotate_free(cv2.GaussianBlur(img, (3, 3), 0), 3)
    cv2.imwrite(str(DIR / "sheet_filled_slanted.png"), sl)
    (DIR / "sheet_filled_truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(len(truth), "cells filled; images in", DIR)


if __name__ == "__main__":
    main()
