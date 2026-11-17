"""
Page-level OCR with a small vision-language model (GLM-OCR, 0.9B, MIT licence),
stored in models/vlm/glm-ocr. It reads a whole page at once, which is far more
accurate than line-by-line recognition on handwritten notes with equations
(word accuracy on the Chemical Kinetics notes: 91% vs 63%), and it writes
formulas as LaTeX and tables as Markdown/HTML.

The model gives text but no positions, so its lines are aligned to the text
lines PaddleOCR detected on the same page (monotonic alignment by text
similarity). That gives every block a box for the page view and the
searchable-PDF text layer.

Runs on an NVIDIA GPU when the CUDA build of torch is installed (fp16, ~2.5 GB
of video memory), otherwise on the CPU (slow: minutes per page).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .model import Block, Line
from .paths import MODELS
from .tables import parse_table_html

VLM_DIR = MODELS / "vlm"
DEFAULT = "glm-ocr"
PROMPTS = {"text": "Text Recognition:", "formula": "Formula Recognition:", "table": "Table Recognition:"}


def available(name: str = DEFAULT) -> bool:
    d = VLM_DIR / name
    return (d / "config.json").exists() and any(d.glob("*.safetensors"))


class PageReader:
    def __init__(self, name: str = DEFAULT, device: str = "auto", max_side: int = 1100) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        path = str(VLM_DIR / name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_side = max_side
        dtype = torch.float16 if device == "cuda" else torch.bfloat16
        self.path = path
        self.proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = None
        if device == "cuda":
            try:  # straight onto the GPU: no second copy in the PC's RAM
                self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, device_map="cuda",
                                                                         local_files_only=True)
            except Exception:  # not enough video memory (or no accelerate): use the CPU
                self.model = None
                torch.cuda.empty_cache()
                self.device = device = "cpu"
        if self.model is None:
            self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True)
        self.model.eval()

    def _to_cpu(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText

        self.model = None
        torch.cuda.empty_cache()
        self.device = "cpu"
        self.model = AutoModelForImageTextToText.from_pretrained(self.path, dtype=torch.bfloat16,
                                                                 local_files_only=True).eval()

    def read(self, img_bgr: np.ndarray, task: str = "text", max_new_tokens: int = 2048) -> str:
        """Read a page. If the GPU runs out of memory (4 GB cards are tight, and other
        programs use video memory too), retry with a smaller image, then on the CPU."""
        import torch

        side = self.max_side
        for _ in range(3):
            try:
                return self._read(img_bgr, task, max_new_tokens, side)
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if self.device != "cuda" or "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                side = int(side * 0.8)
        self._to_cpu()
        return self._read(img_bgr, task, max_new_tokens, self.max_side)

    def _read(self, img_bgr: np.ndarray, task: str, max_new_tokens: int, max_side: int) -> str:
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((max_side, max_side))
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": PROMPTS[task]}]}]
        inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        del inputs, out
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return _strip_loop(text.strip())


def _strip_loop(text: str) -> str:
    """Cut a trailing run where the decoder repeats the same line over and over."""
    lines = text.split("\n")
    out: list[str] = []
    for l in lines:
        if len(out) >= 3 and l.strip() and all(x.strip() == l.strip() for x in out[-3:]):
            continue
        out.append(l)
    return "\n".join(out)


# ------------------------------------------------------------ markdown -> units
def _units(md: str) -> list[dict]:
    """Split the model's Markdown into units: one per text line, formula or table."""
    units: list[dict] = []
    md = md.replace("\r", "")
    # display maths: $$...$$ or \[...\] possibly over several lines
    parts = re.split(r"(\$\$.*?\$\$|\\\[.*?\\\]|<table.*?</table>)", md, flags=re.S)
    para = 0
    for part in parts:
        if not part:
            continue
        if part.startswith("$$") or part.startswith("\\["):
            latex = part[2:-2].strip()
            units.append({"kind": "formula", "text": latex, "para": para})
            para += 1
            continue
        if part.lower().startswith("<table"):
            units.append({"kind": "table", "html": part, "para": para})
            para += 1
            continue
        table: list[str] = []
        for raw in part.split("\n"):
            line = raw.strip()
            if line.startswith("|") and line.endswith("|"):
                table.append(line)
                continue
            if table:
                units.append({"kind": "table", "md": table, "para": para})
                para += 1
                table = []
            if not line:
                para += 1
                continue
            kind = "text"
            if line.startswith("#"):
                kind, line = "title", line.lstrip("#").strip()
            else:
                core = re.sub(r"^[*\-•]\s*", "", line)
                letters = re.sub(r"[^A-Za-z]", "", core)
                if len(letters) >= 4 and letters.isupper() and len(core) < 60:
                    kind, line = "title", core
            units.append({"kind": kind, "text": line, "para": para})
            if kind == "title":
                para += 1
        if table:
            units.append({"kind": "table", "md": table, "para": para})
            para += 1
    return units


def _md_table_cells(rows: list[str]) -> list[list[dict]]:
    out = []
    for r in rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        out.append([{"text": c, "rowspan": 1, "colspan": 1} for c in cells])
    return out


def _plain(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ------------------------------------------------------------ alignment
def _align(units: list[dict], rows: list[Line]) -> list[Optional[list[int]]]:
    """Monotonic alignment of units to detected rows (each unit gets 0..3 consecutive
    rows), maximising text similarity. Returns the row indices for each unit."""
    n, m = len(units), len(rows)
    ut = [_plain(u.get("text", "") or " ".join(u.get("md", [])) or re.sub("<[^>]+>", " ", u.get("html", "")))
          for u in units]
    rt = [_plain(r.text) for r in rows]
    neg = -1e9
    best = np.full((n + 1, m + 1), neg)
    back: dict[tuple[int, int], tuple[int, int]] = {}
    best[0, :] = 0  # rows before the first unit may be skipped (stray marks)
    for i in range(1, n + 1):
        for j in range(0, m + 1):
            # unit i takes no row
            cand = [(best[i - 1, j] - 0.05, (i - 1, j))]
            # skip row j (unread mark / diagram text)
            if j > 0:
                cand.append((best[i, j - 1] - 0.02, (i, j - 1)))
            for k in (1, 2, 3):
                if j - k >= 0:
                    joined = "".join(rt[j - k:j])
                    sim = SequenceMatcher(None, ut[i - 1], joined, autojunk=False).ratio() if ut[i - 1] or joined else 0
                    cand.append((best[i - 1, j - k] + sim - 0.08 * (k - 1), (i - 1, j - k)))
            v, prev = max(cand, key=lambda x: x[0])
            best[i, j] = v
            back[(i, j)] = prev
    # walk back
    res: list[Optional[list[int]]] = [None] * n
    i, j = n, int(np.argmax(best[n]))
    while i > 0:
        pi, pj = back[(i, j)]
        if pi == i - 1 and pj < j:
            res[i - 1] = list(range(pj, j))
        i, j = pi, pj
    return res


def to_blocks(md: str, rows: list[Line], page_w: int, page_h: int) -> list[Block]:
    """Turn the model's Markdown into Blocks with boxes taken from aligned PaddleOCR rows."""
    rows = sorted(rows, key=lambda l: (l.bbox[1] + l.bbox[3]) / 2)
    units = _units(md)
    if not units:
        return []
    match = _align(units, rows) if rows else [None] * len(units)
    boxes: list[Optional[list[int]]] = []
    for mt in match:
        if mt:
            bs = [rows[k].bbox for k in mt]
            boxes.append([min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)])
        else:
            boxes.append(None)
    # rows nobody claimed that sit between two units of the same formula/paragraph:
    # give them to the unit above (fraction denominators, sub/superscripts)
    claimed = {k for mt in match if mt for k in mt}
    for k, r in enumerate(rows):
        if k in claimed:
            continue
        above = [i for i, mt in enumerate(match) if mt and max(mt) < k]
        if above:
            i = above[-1]
            b = boxes[i]
            if r.bbox[1] - b[3] < 1.5 * max(1, b[3] - b[1]) and units[i]["kind"] in ("formula", "text"):
                boxes[i] = [min(b[0], r.bbox[0]), min(b[1], r.bbox[1]), max(b[2], r.bbox[2]), max(b[3], r.bbox[3])]
    # units without rows: place them between their neighbours
    for i, b in enumerate(boxes):
        if b is None:
            prev = next((boxes[k] for k in range(i - 1, -1, -1) if boxes[k]), None)
            nxt = next((boxes[k] for k in range(i + 1, len(boxes)) if boxes[k]), None)
            y1 = prev[3] if prev else 0
            y2 = nxt[1] if nxt else page_h
            if y2 <= y1:
                y2 = y1 + 20
            boxes[i] = [int(page_w * 0.05), int(y1), int(page_w * 0.95), int(min(page_h, y2))]

    blocks: list[Block] = []
    for u, b in zip(units, boxes):
        poly = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        if u["kind"] == "formula":
            blocks.append(Block("formula", list(b), lines=[Line(u["text"], 0.95, poly, source="vlm")], latex=u["text"]))
        elif u["kind"] == "table":
            cells = parse_table_html(u["html"]) if "html" in u else _md_table_cells(u["md"])
            blocks.append(Block("table", list(b), cells=cells, html=u.get("html", "")))
        else:
            line = Line(u["text"], 0.95, poly, source="vlm")
            last = blocks[-1] if blocks else None
            if (u["kind"] == "text" and last is not None and last.kind == "text"
                    and getattr(last, "_para", None) == u["para"]):
                last.lines.append(line)
                last.bbox = [min(last.bbox[0], b[0]), min(last.bbox[1], b[1]), max(last.bbox[2], b[2]), max(last.bbox[3], b[3])]
            else:
                nb = Block(u["kind"], list(b), lines=[line])
                nb._para = u["para"]  # type: ignore[attr-defined]
                blocks.append(nb)
    return blocks
