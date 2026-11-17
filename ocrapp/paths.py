"""Project-relative locations. Everything lives inside the project folder."""
from __future__ import annotations

import sys
from pathlib import Path

# In the packaged OfflineOCR.exe the data folders (models, user, output, ...)
# sit next to the exe; otherwise they are in the project folder.
ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PADDLE = MODELS / "paddle"
FORMULA = MODELS / "formula"
HANDWRITING = MODELS / "handwriting"
DATASETS = ROOT / "datasets"
HW_DATASET = DATASETS / "handwriting"
USER = ROOT / "user"            # settings, corrections dictionary, logs
OUTPUT = ROOT / "output"

for _d in (DATASETS, HW_DATASET, USER, OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)


def paddle_dict(rel: str) -> str:
    """A dictionary file shipped inside the paddleocr package."""
    import paddleocr

    return str(Path(paddleocr.__file__).parent / rel)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))
