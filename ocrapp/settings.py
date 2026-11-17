"""User settings (JSON in user/settings.json) and the post-OCR corrections dictionary."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields

from .paths import USER

SETTINGS_FILE = USER / "settings.json"
CORRECTIONS_FILE = USER / "corrections.json"


@dataclass
class OCRSettings:
    mode: str = "auto"            # auto | printed | handwritten | formula
    reader: str = "auto"          # AI page reader (GLM-OCR): auto (handwritten pages) | ai (always) | classic (never)
    vlm_device: str = "auto"      # auto (GPU if available) | cpu
    vlm_max_side: int = 1100      # page size given to the AI reader (1100 was best on the test notes)
    layout: bool = True           # layout analysis (titles, columns, figures)
    tables: bool = True           # table structure recognition
    formulas: bool = True         # LaTeX for equation regions
    auto_orient: bool = True      # detect 0/90/180/270 page rotation
    deskew: bool = True           # straighten slanted pages
    enhance: bool = False         # contrast / denoise for poor scans
    pdf_dpi: int = 200
    det_model: str = "det_v4"     # det_v4 (PP-OCRv4) | det (PP-OCRv3 English)
    drop_score: float = 0.3
    hw_threshold: float = 0.85    # auto mode: re-read lines below this confidence with TrOCR
    hw_model: str = ""            # folder in models/handwriting; "" = active.txt
    hw_beams: int = 3
    det_box_thresh: float = 0.5
    det_unclip: float = 1.6
    max_side: int = 2400          # downscale huge pages for speed
    threads: int = 4
    apply_corrections: bool = True
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "OCRSettings":
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def save(self) -> None:
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def load_corrections() -> dict[str, str]:
    try:
        return dict(json.loads(CORRECTIONS_FILE.read_text(encoding="utf-8")))
    except Exception:
        return {}


def save_corrections(d: dict[str, str]) -> None:
    CORRECTIONS_FILE.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def apply_corrections(text: str, corr: dict[str, str]) -> str:
    """Whole-word replacements (case-sensitive) learned from the user."""
    for wrong, right in corr.items():
        if not wrong:
            continue
        text = re.sub(r"(?<!\w)" + re.escape(wrong) + r"(?!\w)", right, text)
    return text
