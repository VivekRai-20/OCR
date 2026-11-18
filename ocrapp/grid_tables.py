"""
Ruled-table reader: finds tables from their printed lines and rebuilds the grid.

Works for any layout of ruled table (forms, data sheets, logs):
  * horizontal / vertical rules are extracted with morphology,
  * every rule position becomes a grid line (rows / columns),
  * neighbouring grid cells with no rule between them are merged
    (row / column spans, e.g. "Continuous run" over 7 columns),
  * every cell is kept, including empty ones, so the exported Excel / Word table
    has exactly the same shape as the paper form.

Text inside a cell is read separately by the engine (printed text with
PaddleOCR, handwritten entries with the handwriting model).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class GridCell:
    r: int
    c: int
    rowspan: int
    colspan: int
    box: list  # x1, y1, x2, y2 (page pixels, inside the rules)
    text: str = ""
    conf: float = 1.0
    lines: list = field(default_factory=list)


@dataclass
class GridTable:
    bbox: list
    rows: list  # y positions of horizontal rules
    cols: list  # x positions of vertical rules
    cells: list  # GridCell, row-major


def _binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)


def _line_masks(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bw = _binarize(gray)
    h, w = bw.shape
    # a rule is a long thin run of ink; handwriting and text are not
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 45), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 45)))
    horiz = cv2.dilate(cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk), np.ones((3, 3), np.uint8))
    vert = cv2.dilate(cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk), np.ones((3, 3), np.uint8))
    return horiz, vert


def _positions(profile: np.ndarray, min_frac: float, min_gap: int) -> list[int]:
    """Centres of runs where a projection profile is high (one per rule)."""
    if profile.max() <= 0:
        return []
    on = profile >= min_frac * profile.max()
    out, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start + i - 1) // 2)
            start = None
    if start is not None:
        out.append((start + len(on) - 1) // 2)
    merged: list[int] = []
    for p in out:
        if merged and p - merged[-1] < min_gap:
            merged[-1] = (merged[-1] + p) // 2
        else:
            merged.append(p)
    return merged


def find_tables(img: np.ndarray, min_rows: int = 2, min_cols: int = 2) -> list[GridTable]:
    horiz, vert = _line_masks(img)
    grid = cv2.bitwise_or(horiz, vert)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    H, W = grid.shape
    tables = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < W * 0.08 or h < H * 0.03:
            continue
        sub_h = horiz[y:y + h, x:x + w] & (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        # a rule counts if it is at least ~1 cell long: use a low fraction so
        # short rules (sub-columns under a merged header) are still found
        rows = _positions((sub_h > 0).sum(axis=1).astype(float), 0.06, 6)
        if len(rows) < min_rows + 1:
            continue
        # vertical rules can be as short as one row, too short for a morphology
        # kernel that ignores text; find them row band by row band instead: a rule
        # crosses the whole band, letters and handwriting do not
        gray = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img[y:y + h, x:x + w]
        bw = _binarize(gray)
        sub_v = np.zeros_like(bw)
        for r0, r1 in zip(rows[:-1], rows[1:]):
            a, b = r0 + 4, r1 - 4
            if b - a < 4:
                continue
            band = bw[a:b] > 0
            cover = band.mean(axis=0)
            cover = np.maximum(cover, np.maximum(np.roll(cover, 1), np.roll(cover, -1)))  # 1-px slant
            sub_v[a:b, cover >= 0.9] = 255
        cols = _positions((sub_v > 0).sum(axis=0).astype(float), 0.02, 12)
        # the dark edge of a scanned / photographed page is not a table rule
        edge = max(4, int(0.02 * W))
        cols = [c for c in cols if edge <= c + x <= W - edge]
        if len(cols) < min_cols + 1:
            continue
        if not _is_real_table(sub_h, sub_v, rows, cols):
            continue  # e.g. ruled notebook paper: lines run past any "frame"
        t = _build(sub_h, sub_v, rows, cols)
        for c in t.cells:  # page coordinates
            c.box = [c.box[0] + x, c.box[1] + y, c.box[2] + x, c.box[3] + y]
        t.rows = [r + y for r in rows]
        t.cols = [c + x for c in cols]
        t.bbox = [int(x), int(y), int(x + w), int(y + h)]
        tables.append(t)
    tables.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
    return tables


def _is_real_table(horiz: np.ndarray, vert: np.ndarray, rows: list[int], cols: list[int]) -> bool:
    """A ruled table is closed by a left and a right border that run down most of
    it, and its horizontal rules stop at those borders. Ruled notebook paper has
    full-width lines with at most a margin line, and handwriting strokes that
    happen to cross one line gap are short."""
    h, w = horiz.shape
    bands = list(zip(rows[:-1], rows[1:]))

    def coverage(x: int) -> float:
        hit = sum(1 for a, b in bands if b - a > 8 and vert[a + 4:b - 4, max(0, x - 3):x + 4].any(axis=1).mean() > 0.8)
        return hit / max(1, len(bands))

    framed = [x for x in cols if coverage(x) >= 0.5]
    if len(framed) < 2:
        return False
    left, right = framed[0], framed[-1]
    if right - left < 0.6 * w:
        return False
    # horizontal rules must not run on well past the borders
    margin = max(15, int(0.02 * w))
    over = 0
    for y in rows:
        seg = horiz[max(0, y - 3):y + 4]
        outside = seg[:, :max(0, left - margin)].any(axis=0).sum() + seg[:, right + margin:].any(axis=0).sum()
        if outside > 0.015 * w:
            over += 1
    return over <= 0.25 * len(rows)


def _has_rule(mask: np.ndarray, horizontal: bool, at: int, a: int, b: int, cover: float = 0.5) -> bool:
    """Is there a rule at row/col `at` between a and b (covering `cover` of it)?"""
    if b <= a:
        return False
    band = 4
    if horizontal:
        seg = mask[max(0, at - band):at + band + 1, a:b]
        return seg.size > 0 and (seg.max(axis=0) > 0).mean() >= cover
    seg = mask[a:b, max(0, at - band):at + band + 1]
    return seg.size > 0 and (seg.max(axis=1) > 0).mean() >= cover


def _build(horiz: np.ndarray, vert: np.ndarray, rows: list[int], cols: list[int]) -> GridTable:
    nr, nc = len(rows) - 1, len(cols) - 1
    # walls: right wall of (r, c) and bottom wall of (r, c)
    right = np.ones((nr, nc), bool)
    bottom = np.ones((nr, nc), bool)
    for r in range(nr):
        for c in range(nc - 1):
            right[r, c] = _has_rule(vert, False, cols[c + 1], rows[r] + 3, rows[r + 1] - 3)
    for r in range(nr - 1):
        for c in range(nc):
            bottom[r, c] = _has_rule(horiz, True, rows[r + 1], cols[c] + 3, cols[c + 1] - 3)
    # flood-fill cells that are not separated by a rule -> merged cells
    owner = -np.ones((nr, nc), int)
    cells: list[GridCell] = []
    for r in range(nr):
        for c in range(nc):
            if owner[r, c] >= 0:
                continue
            # grow right while no wall, then down while the whole row band has no wall
            c2 = c
            while c2 + 1 < nc and not right[r, c2] and owner[r, c2 + 1] < 0:
                c2 += 1
            r2 = r
            while r2 + 1 < nr and all(not bottom[r2, k] for k in range(c, c2 + 1)) \
                    and all(owner[r2 + 1, k] < 0 for k in range(c, c2 + 1)):
                r2 += 1
            idx = len(cells)
            owner[r:r2 + 1, c:c2 + 1] = idx
            pad = 3
            cells.append(GridCell(r, c, r2 - r + 1, c2 - c + 1,
                                  [cols[c] + pad, rows[r] + pad, cols[c2 + 1] - pad, rows[r2 + 1] - pad]))
    return GridTable(bbox=[0, 0, 0, 0], rows=rows, cols=cols, cells=cells)


def cell_has_ink(img: np.ndarray, box, min_frac: float = 0.004) -> bool:
    """True when a cell holds writing (rules near the border are ignored)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    if x2 - x1 < 6 or y2 - y1 < 6:
        return False
    m = 3
    crop = img[y1 + m:y2 - m, x1 + m:x2 - m]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    bw = _binarize(gray)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # specks
    return (bw > 0).mean() >= min_frac and (bw > 0).sum() >= 25


def to_block_cells(t: GridTable) -> list[list[dict]]:
    """Rows of {"text", "rowspan", "colspan"} in HTML order (cells listed in the
    row where they start), the format used by the exporters."""
    rows: list[list[dict]] = [[] for _ in range(len(t.rows) - 1)]
    for c in sorted(t.cells, key=lambda c: (c.r, c.c)):
        rows[c.r].append({"text": c.text, "rowspan": c.rowspan, "colspan": c.colspan})
    return rows
