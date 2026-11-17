"""Per-stage timing for the first N pages of a document.

    .venv\\Scripts\\python tests\\profile_pages.py file.pdf [N]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocrapp.engine import OCREngine  # noqa: E402
from ocrapp import imaging  # noqa: E402

STAGES = ("detect_rotation", "detect_skew", "_read_lines", "_handwriting", "_layout", "_table",
          "_figure_formula", "_formula_lines")


def main() -> None:
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    e = OCREngine()
    pages = imaging.load_pages(path, e.s.pdf_dpi)[:n]
    fr = None
    for i, img in enumerate(pages):
        timings: dict[str, float] = {}
        for name in STAGES:
            f = getattr(e, name)

            def wrap(*a, _f=f, _n=name, **k):
                t = time.time()
                try:
                    return _f(*a, **k)
                finally:
                    timings[_n] = timings.get(_n, 0) + time.time() - t

            setattr(e, name, wrap)
        if e._formula is None:
            e.formula_model()
        fr = e._formula
        orig_read = type(fr).read

        def fread(self, *a, **k):
            t = time.time()
            try:
                return orig_read(self, *a, **k)
            finally:
                timings["formula_read"] = timings.get("formula_read", 0) + time.time() - t
                timings["formula_calls"] = timings.get("formula_calls", 0) + 1

        type(fr).read = fread
        t = time.time()
        p = e.process_image(img, i)
        total = time.time() - t
        type(fr).read = orig_read
        for name in STAGES:
            delattr(e, name)
        lines = [l for b in p.blocks for l in b.lines]
        print(f"page {i + 1}: {total:.1f}s  " + ", ".join(f"{k}={v:.1f}" for k, v in timings.items()), flush=True)
        print(f"   {len(p.blocks)} blocks {sorted({b.kind for b in p.blocks})}, {len(lines)} lines, "
              f"{sum(l.source == 'hand' for l in lines)} handwriting", flush=True)
        for b in p.blocks[:40]:
            print(f"   [{b.kind}] {' '.join(b.text.split())[:110]}", flush=True)


if __name__ == "__main__":
    main()
