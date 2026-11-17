"""Document model produced by the OCR engine and consumed by the GUI and exporters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

BLOCK_KINDS = ("title", "text", "list", "table", "formula", "figure", "caption", "header", "footer")


@dataclass
class Line:
    text: str
    conf: float
    poly: list  # 4 points [[x, y], ...] in page-image coordinates
    source: str = "print"  # print | hand | formula
    alt_text: str = ""     # the other recogniser's reading, if any
    alt_conf: float = 0.0

    @property
    def bbox(self) -> list[int]:
        xs = [p[0] for p in self.poly]
        ys = [p[1] for p in self.poly]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


@dataclass
class Block:
    kind: str
    bbox: list  # [x1, y1, x2, y2]
    lines: list = field(default_factory=list)
    html: str = ""          # tables
    cells: list = field(default_factory=list)  # tables: rows of {"text", "rowspan", "colspan"} grid
    latex: str = ""         # formulas
    edited_text: Optional[str] = None  # user edits from the GUI win over OCR text

    @property
    def text(self) -> str:
        if self.edited_text is not None:
            return self.edited_text
        if self.kind == "formula" and self.latex:
            return self.latex
        if self.kind == "table" and self.cells:
            return "\n".join("\t".join(c for c in row) for row in table_grid(self))
        return "\n".join(l.text for l in self.lines)

    @property
    def conf(self) -> float:
        if not self.lines:
            return 1.0
        return float(np.mean([l.conf for l in self.lines]))


@dataclass
class Page:
    index: int
    image: np.ndarray  # processed (rotated/deskewed) BGR image the coordinates refer to
    blocks: list = field(default_factory=list)
    rotation: int = 0       # 0/90/180/270 applied
    skew: float = 0.0       # degrees applied
    seconds: float = 0.0
    notes: list = field(default_factory=list)  # e.g. "handwritten page detected"
    markdown: str = ""      # raw output of the AI page reader, when it was used

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())


@dataclass
class Document:
    source: str
    pages: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n\f".join(p.text for p in self.pages)


def table_grid(block: Block) -> list[list[str]]:
    """Expand a table's cells (with spans) to a rectangular grid of strings."""
    if block.edited_text is not None:
        return [r.split("\t") for r in block.edited_text.splitlines()]
    grid: dict[tuple[int, int], str] = {}
    r = 0
    max_c = 0
    for row in block.cells:
        c = 0
        for cell in row:
            while (r, c) in grid:
                c += 1
            for dr in range(cell.get("rowspan", 1)):
                for dc in range(cell.get("colspan", 1)):
                    grid[(r + dr, c + dc)] = cell["text"] if (dr, dc) == (0, 0) else ""
            c += cell.get("colspan", 1)
            max_c = max(max_c, c)
        r += 1
    rows = max((k[0] for k in grid), default=-1) + 1
    return [[grid.get((i, j), "") for j in range(max_c)] for i in range(rows)]
