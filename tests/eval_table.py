"""Scores table reading on the filled data sheet against its ground truth.

    python tests/eval_table.py [image]   (default: tests/tables/sheet_filled.png)
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocrapp import imaging  # noqa: E402
from ocrapp.engine import OCREngine  # noqa: E402
from ocrapp.model import table_grid  # noqa: E402

DIR = Path(__file__).parent / "tables"


def norm(s: str) -> str:
    return re.sub(r"[\s·]", "", s.replace(",", ".")).lower()


def main() -> None:
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DIR / "sheet_filled.png"
    truth = json.loads((DIR / "sheet_filled_truth.json").read_text(encoding="utf-8"))
    e = OCREngine()
    t = time.time()
    page = e.process_image(imaging.imread(img_path))
    secs = time.time() - t
    tables = [b for b in page.blocks if b.kind == "table"]
    print(f"{img_path.name}: {secs:.0f}s, {len(tables)} table block(s), notes={page.notes}")
    if not tables:
        return
    b = max(tables, key=lambda b: len(b.cells))
    # cell (r, c) -> text, using the span layout
    pos: dict[tuple[int, int], str] = {}
    occupied = set()
    for r, row in enumerate(b.cells):
        c = 0
        for cell in row:
            while (r, c) in occupied:
                c += 1
            pos[(r, c)] = cell["text"]
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    occupied.add((r + dr, c + dc))
            c += cell["colspan"]
    grid = table_grid(b)
    print(f"grid {len(grid)} x {max(len(r) for r in grid)}")
    right = wrong = 0
    misses = []
    for key, val in truth.items():
        r, c = map(int, key.split(","))
        got = pos.get((r, c), "<no cell>")
        if norm(got) == norm(val):
            right += 1
        else:
            wrong += 1
            misses.append(f"  ({r},{c}) want {val!r} got {got!r}")
    # data-area cells that should be empty
    empty_ok = empty_bad = 0
    for (r, c), txt in pos.items():
        if 7 <= r <= 26 and f"{r},{c}" not in truth:
            if txt.strip():
                empty_bad += 1
            else:
                empty_ok += 1
    headers = [pos.get(k, "") for k in ((5, 0), (5, 7), (5, 13), (6, 0), (6, 2), (28, 0), (28, 12))]
    print(f"handwritten cells exact: {right}/{right + wrong} = {right / max(1, right + wrong):.0%}")
    print(f"empty cells kept empty: {empty_ok}/{empty_ok + empty_bad}")
    print("printed headers:", headers)
    other = [x.text for x in page.blocks if x.kind != "table"]
    print("text outside the table:", other)
    print("\n".join(misses[:25]))


if __name__ == "__main__":
    main()
