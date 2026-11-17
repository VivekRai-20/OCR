"""Command-line / batch OCR (same engine as the GUI).

    python cli.py input.pdf -o output/ -f docx xlsx pdf txt
    python cli.py scans_folder -o output/ --mode handwritten
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocrapp.engine import OCREngine  # noqa: E402
from ocrapp.exporters import EXPORTERS, export  # noqa: E402
from ocrapp.imaging import SUPPORTED  # noqa: E402
from ocrapp.settings import OCRSettings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline OCR")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="output")
    ap.add_argument("-f", "--formats", nargs="+", default=["txt", "json"], choices=sorted(EXPORTERS))
    ap.add_argument("--mode", choices=["auto", "printed", "handwritten", "formula"])
    ap.add_argument("--no-layout", action="store_true")
    ap.add_argument("--no-orient", action="store_true")
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--enhance", action="store_true")
    a = ap.parse_args()

    s = OCRSettings.load()
    if a.mode:
        s.mode = a.mode
    if a.no_layout:
        s.layout = False
    if a.no_orient:
        s.auto_orient = False
    if a.no_deskew:
        s.deskew = False
    if a.enhance:
        s.enhance = True
    eng = OCREngine(s)
    files: list[Path] = []
    for i in a.inputs:
        p = Path(i)
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED) if p.is_dir() else [p]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        doc = eng.process_file(str(f), progress=lambda m, x: print(f"\r{f.name}: {m:<50}", end="", flush=True))
        print()
        for fmt in a.formats:
            dest = out / f"{f.stem}.{EXPORTERS[fmt][1]}"
            export(doc, fmt, dest)
            print("  ->", dest)


if __name__ == "__main__":
    main()
