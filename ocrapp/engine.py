"""
The OCR engine. Fully offline: every model is loaded from models/.

Per page:
 1. optional enhancement (denoise + contrast)
 2. page orientation (0/90/180/270) chosen by OCR confidence
 3. deskew: median angle of detected text lines, any angle up to +-45 deg
 4. PaddleOCR PP-OCRv4 detection + recognition of every text line; slanted or
    vertical lines are cut out with a perspective warp and the 180-degree line
    classifier fixes upside-down lines
 5. layout analysis (CDLA picodet: title, text, table, figure, equation, ...)
 6. tables -> SLANet structure (HTML with row/col spans)
 7. equations -> LaTeX-OCR (ONNX)
 8. handwriting -> TrOCR re-reads lines (all lines in "handwritten" mode, and
    low-confidence lines in "auto" mode, keeping the more confident reading)
"""
from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from typing import Callable, Optional

import cv2
import numpy as np

from . import compat

compat.prepare()

from . import imaging, vlm  # noqa: E402
from .model import Block, Document, Line, Page  # noqa: E402
from .paths import PADDLE, paddle_dict  # noqa: E402
from .settings import OCRSettings, apply_corrections, load_corrections  # noqa: E402

LAYOUT_MAP = {
    "text": "text", "title": "title", "figure": "figure", "figure_caption": "caption",
    "table": "table", "table_caption": "caption", "header": "header", "footer": "footer",
    "reference": "text", "equation": "formula", "list": "list",
}

Progress = Callable[[str, float], None]


def _no_download(model_storage_directory, url):  # replaces paddleocr's downloader
    import os

    if not os.path.exists(os.path.join(model_storage_directory, "inference.pdiparams")):
        raise FileNotFoundError(
            f"Model missing in {model_storage_directory}. This app is offline; "
            "run tools/fetch_assets.py on a PC with internet and copy models/ over.")


class OCREngine:
    def __init__(self, settings: Optional[OCRSettings] = None) -> None:
        self.s = settings or OCRSettings()
        self._ocr = None
        self._ocr_key = None
        self._struct = None
        self._struct_key = None
        self._hw = None
        self._hw_key = None
        self._formula = None
        self._vlm = None
        self._vlm_key = None
        self.cancelled = False

    # ------------------------------------------------------------------ models
    def _paddle_common(self) -> dict:
        return dict(
            lang="en", show_log=False, use_gpu=False, ocr_version="PP-OCRv4",
            det_model_dir=str(PADDLE / self.s.det_model), rec_model_dir=str(PADDLE / "rec"),
            cls_model_dir=str(PADDLE / "cls"), use_angle_cls=True,
            rec_char_dict_path=paddle_dict("ppocr/utils/en_dict.txt"),
            det_db_box_thresh=self.s.det_box_thresh, det_db_unclip_ratio=self.s.det_unclip,
            det_limit_side_len=1600, det_limit_type="max", drop_score=self.s.drop_score,
            cpu_threads=self.s.threads, enable_mkldnn=False, rec_batch_num=6,
        )

    def ocr_model(self):
        key = (self.s.det_model, self.s.det_box_thresh, self.s.det_unclip, self.s.drop_score)
        if self._ocr is None or self._ocr_key != key:
            import paddleocr.paddleocr as pp

            pp.maybe_download = _no_download
            self._ocr = pp.PaddleOCR(**self._paddle_common())
            self._ocr_key = key
        return self._ocr

    def struct_model(self):
        key = (self.s.det_model,)
        if self._struct is None or self._struct_key != key:
            import paddleocr.paddleocr as pp

            pp.maybe_download = _no_download
            kw = self._paddle_common()
            kw.update(
                structure_version="PP-StructureV2", layout=True, table=True, ocr=False,
                table_model_dir=str(PADDLE / "table"), layout_model_dir=str(PADDLE / "layout"),
                layout_dict_path=paddle_dict("ppocr/utils/dict/layout_dict/layout_cdla_dict.txt"),
                table_char_dict_path=paddle_dict("ppocr/utils/dict/table_structure_dict.txt"),
                layout_score_threshold=0.5, image_orientation=False, recovery=False,
            )
            self._struct = pp.PPStructure(**kw)
            self._struct_key = key
        return self._struct

    def hw_model(self):
        from .handwriting import HandwritingReader

        key = (self.s.hw_model, self.s.hw_beams)
        if self._hw is None or self._hw_key != key:
            try:
                import torch

                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = "cpu"
            self._hw = HandwritingReader(self.s.hw_model, beams=self.s.hw_beams, device=dev)
            self._hw_key = key
        return self._hw

    def formula_model(self):
        if self._formula is None:
            from .formula import FormulaReader

            self._formula = FormulaReader()
        return self._formula

    def unload_handwriting(self) -> None:
        self._hw = None

    # --------------------------------------------------------------- helpers
    def _read_lines(self, img: np.ndarray, cls: bool = True) -> list[Line]:
        res = self.ocr_model().ocr(img, det=True, rec=True, cls=cls)
        lines = []
        for item in (res[0] or []) if res else []:
            poly, (text, conf) = item
            lines.append(Line(text=text, conf=float(conf), poly=[[float(x), float(y)] for x, y in poly]))
        if cls:
            self._recheck(img, [l for l in lines if l.conf < 0.85])
        return lines

    def _recheck(self, img: np.ndarray, lines: list[Line]) -> None:
        """The 180-degree line classifier sometimes flips an upright line (or misses an
        upside-down one). Re-read doubtful lines both ways and keep the better reading."""
        if not lines:
            return
        crops = []
        for l in lines:
            c = imaging.crop_poly(img, l.poly)
            crops += [c, np.ascontiguousarray(np.rot90(c, 2))]
        res = self.ocr_model().ocr(crops, det=False, rec=True, cls=False)
        flat = res[0] if res and len(res) == 1 and isinstance(res[0], list) else res
        for i, l in enumerate(lines):
            for text, conf in (flat[2 * i], flat[2 * i + 1]):
                if conf > l.conf + 0.02:
                    l.text, l.conf = text, float(conf)

    def _score(self, lines: list[Line]) -> float:
        return sum(l.conf * len(l.text) for l in lines if l.conf > 0.5)

    def detect_rotation(self, img: np.ndarray) -> int:
        """Counter-clockwise rotation (0/90/180/270) that makes the page upright."""
        ocr = self.ocr_model()
        small = imaging.limit_size(img, 1280)

        def boxes_of(im):
            det = ocr.ocr(im, det=True, rec=False, cls=False)
            return [np.asarray(b) for b in (det[0] if det and det[0] is not None else [])]

        boxes = boxes_of(small)
        if not boxes:
            return 0
        # text lines that run vertically -> the page is turned by 90 or 270 degrees
        tall = sum(1 for p in boxes if np.linalg.norm(p[0] - p[3]) > 1.3 * np.linalg.norm(p[0] - p[1]))
        base = 90 if tall > len(boxes) / 2 else 0
        if base:
            small = imaging.rotate90(small, base)
            boxes = boxes_of(small)
        # upright vs upside down: vote of the 0/180 line classifier over the longest lines
        boxes = sorted(boxes, key=lambda p: -np.linalg.norm(p[0] - p[1]))[:40]
        crops = [imaging.crop_poly(small, p) for p in boxes]
        if not crops:
            return base
        try:
            _, cls_res, _ = ocr.text_classifier(crops)
        except Exception:
            scores = {a: self._score(self._read_lines(imaging.rotate90(small, a), cls=False)) for a in (0, 180)}
            return (base + max(scores, key=scores.get)) % 360
        flip = sum(s for lab, s in cls_res if str(lab) == "180")
        keep = sum(s for lab, s in cls_res if str(lab) == "0")
        return (base + 180) % 360 if flip > keep else base

    def detect_skew(self, img: np.ndarray) -> float:
        small = imaging.limit_size(img, 1280)
        det = self.ocr_model().ocr(small, det=True, rec=False, cls=False)
        boxes = det[0] if det and det[0] is not None else []
        angles, weights = [], []
        for b in boxes:
            a, length = imaging.box_angle(b)
            h = np.linalg.norm(np.asarray(b[0]) - np.asarray(b[3]))
            if length > 2.5 * max(h, 1) and abs(a) < 45:
                angles.append(a)
                weights.append(length)
        if len(angles) < 2:
            return 0.0
        order = np.argsort(angles)
        a, w = np.asarray(angles)[order], np.asarray(weights)[order]
        return float(a[np.searchsorted(np.cumsum(w), w.sum() / 2)])  # weighted median

    # ------------------------------------------------------------ pipeline
    def process_file(self, path: str, progress: Optional[Progress] = None, pages: Optional[list[int]] = None) -> Document:
        self.cancelled = False
        total = imaging.page_count(path)
        doc = Document(source=str(path))
        idx = pages if pages is not None else list(range(total))
        for n, i in enumerate(idx):
            if self.cancelled:
                break

            def sub(msg: str, frac: float, n=n) -> None:
                if progress:
                    progress(f"Page {i + 1}/{total}: {msg}", (n + frac) / len(idx))

            doc.pages.append(self.process_image(imaging.load_page(path, i, self.s.pdf_dpi), index=i, progress=sub))
        if progress:
            progress("Done", 1.0)
        return doc

    def process_image(self, img: np.ndarray, index: int = 0, progress: Optional[Progress] = None) -> Page:
        t0 = time.time()
        s = self.s
        say = progress or (lambda m, f: None)
        img = imaging.limit_size(img, s.max_side)
        if s.enhance:
            say("enhancing", 0.02)
            img = imaging.enhance(img)
        if self._vlm is None and s.reader != "classic" and s.mode != "printed" and vlm.available():
            try:  # load the AI reader before Paddle's models (see vlm_model)
                import torch

                if s.reader == "ai" or (torch.cuda.is_available() and s.vlm_device != "cpu"):
                    say("loading the AI page reader", 0.03)
                    self.vlm_model()
            except Exception as e:
                page_note = f"AI reader unavailable: {e}"
                print(page_note)
        rot, skew = 0, 0.0
        if s.auto_orient and s.mode != "formula":
            say("detecting orientation", 0.05)
            rot = self.detect_rotation(img)
            img = imaging.rotate90(img, rot)
        if s.deskew and s.mode != "formula":
            say("deskewing", 0.1)
            skew = self.detect_skew(img)
            if abs(skew) > 0.3:
                img = imaging.rotate_free(img, -skew)
        page = Page(index=index, image=img, rotation=rot, skew=skew)

        if s.mode == "formula":
            say("reading formula", 0.3)
            h, w = img.shape[:2]
            latex = ""
            if s.reader != "classic" and vlm.available():
                try:
                    latex = self.vlm_model().read(img, task="formula", max_new_tokens=600).strip().strip("$").strip()
                except Exception as e:
                    page.notes.append(f"AI reader failed: {e}")
            if not latex:
                latex = self.formula_model().read(img, max_tokens=400)
            page.blocks = [Block("formula", [0, 0, w, h], latex=latex)]
            page.seconds = time.time() - t0
            return page

        say("reading text", 0.2)
        lines = self._read_lines(img)
        all_lines = list(lines)
        # ruled tables (forms, data sheets): exact grid from the printed rules,
        # every cell read on its own; the rest of the page is read as usual
        grid_blocks: list[Block] = []
        if s.tables:
            say("ruled tables", 0.25)
            grid_blocks, lines = self._grid_tables(img, lines, page.notes, say)
        if self._use_vlm(all_lines):
            say(f"AI page reader ({self.vlm_device()}) - reading the whole page", 0.35)
            try:
                h, w = img.shape[:2]
                rest = img
                if grid_blocks:  # the tables are done; let the AI reader see only the rest
                    rest = img.copy()
                    for b in grid_blocks:
                        x1, y1, x2, y2 = b.bbox
                        rest[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4] = 255
                blocks = []
                if not grid_blocks or len(lines) >= 1:
                    md = self.vlm_model().read(rest)
                    page.markdown = md
                    blocks = vlm.to_blocks(md, _join_words(list(lines)), w, h)
                page.blocks = _reading_order(blocks + grid_blocks, w)
                page.notes.append(f"read by GLM-OCR on {self.vlm_device()}")
                self._apply_corrections(page)
                page.seconds = time.time() - t0
                return page
            except Exception as e:  # fall back to the classic pipeline
                page.notes.append(f"AI reader failed ({e}); used classic OCR")
        if s.mode in ("auto", "handwritten"):
            say("handwriting", 0.45)
            self._handwriting(img, lines, force=s.mode == "handwritten", notes=page.notes)

        regions = []
        if s.layout or s.tables or s.formulas:
            say("layout analysis", 0.65)
            regions = self._layout(img)
            if grid_blocks:  # the ruled tables are already read
                regions = [r for r in regions
                           if not any(_overlap(r["bbox"], b.bbox) > 0.4 for b in grid_blocks)]
        page.blocks = self._assemble(img, lines, regions, say)
        if grid_blocks:
            page.blocks = _reading_order(page.blocks + grid_blocks, img.shape[1])
        self._apply_corrections(page)
        page.seconds = time.time() - t0
        return page

    # ------------------------------------------------------------ ruled tables
    def _grid_tables(self, img: np.ndarray, lines: list[Line], notes: list, say) -> tuple[list[Block], list[Line]]:
        """Ruled tables -> Blocks with the exact grid (merged and empty cells kept).
        Returns the table blocks and the page lines that are outside every table."""
        from . import grid_tables

        try:
            tables = grid_tables.find_tables(img)
        except Exception as e:
            notes.append(f"ruled-table detection failed: {e}")
            return [], lines
        if not tables:
            return [], lines
        outside = [l for l in lines if not any(_center_in(l.bbox, t.bbox) for t in tables)]
        blocks = []
        hw: list[tuple] = []  # (cell, line) that should be checked by the handwriting model
        n_cells = sum(len(t.cells) for t in tables)
        done = 0
        for t in tables:
            for cell in t.cells:
                done += 1
                if done % 20 == 0:
                    say(f"reading table cells {done}/{n_cells}", 0.25 + 0.1 * done / n_cells)
                if not grid_tables.cell_has_ink(img, cell.box):
                    continue  # empty cell: stays empty
                cell.lines = self._read_cell(img, cell.box)
                if not cell.lines:
                    hw.append((cell, None))  # ink, but no printed text found: likely handwriting
                    continue
                for l in cell.lines:
                    if self.s.mode == "handwritten" or l.conf < self.s.hw_threshold:
                        hw.append((cell, l))
        if hw and self.s.mode != "printed":
            say(f"handwriting in {len(hw)} table cells", 0.4)
            self._read_cells_handwriting(img, hw, notes)
        for t in tables:
            all_cell_lines = []
            for cell in t.cells:
                cell.lines = _merge_rows(cell.lines) if cell.lines else []
                cell.text = "\n".join(l.text for l in cell.lines).strip()
            _fix_numeric_columns(t)
            for cell in t.cells:
                all_cell_lines += cell.lines
            b = Block("table", list(t.bbox), lines=all_cell_lines, cells=grid_tables.to_block_cells(t))
            blocks.append(b)
        filled = sum(1 for t in tables for c in t.cells if c.text)
        notes.append(f"{len(tables)} ruled table(s): {n_cells} cells, {filled} with text, "
                     f"{n_cells - filled} empty")
        return blocks, outside

    def _read_cell(self, img: np.ndarray, box) -> list[Line]:
        """Printed-text OCR of one table cell (the crop stops text running into the
        next cell). Lines come back in page coordinates."""
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = _erase_rules(img[y1:y2, x1:x2].copy())
        if crop.size == 0:
            return []
        pad = 10
        padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        res = self.ocr_model().ocr(padded, det=True, rec=True, cls=False)
        out = []
        for item in (res[0] or []) if res else []:
            poly, (text, conf) = item
            if not text.strip():
                continue
            out.append(Line(text=text, conf=float(conf),
                            poly=[[float(px) - pad + x1, float(py) - pad + y1] for px, py in poly]))
        if not out:  # very small or faint text: recognise the whole cell as one line
            r = self.ocr_model().ocr(padded, det=False, rec=True, cls=False)
            if r and r[0] and r[0][0][0].strip() and r[0][0][1] > 0.5:
                text, conf = r[0][0]
                out.append(Line(text=text, conf=float(conf), poly=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
        return out

    def _read_cells_handwriting(self, img: np.ndarray, todo: list[tuple], notes: list) -> None:
        try:
            reader = self.hw_model()
        except Exception as e:
            notes.append(f"handwriting model unavailable: {e}")
            return
        crops = []
        for cell, line in todo:
            if line is None:
                crops.append(_trim_ink(imaging.crop_box(img, cell.box, pad=0)))
            else:
                crops.append(imaging.crop_poly(img, line.poly))
        results = reader.read(crops)
        for (cell, line), (text, conf) in zip(todo, results):
            text = re.sub(r"\s+([.,;:!?)])", lambda m: m.group(1), text).strip()
            if not text:
                continue
            if line is None:
                if conf >= 0.35:  # below that TrOCR is guessing at specks or strokes
                    x1, y1, x2, y2 = cell.box
                    cell.lines = [Line(text=text, conf=conf, poly=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                       source="hand")]
            elif self.s.mode == "handwritten" or conf > line.conf + 0.03:
                line.alt_text, line.alt_conf = line.text, line.conf
                line.text, line.conf, line.source = text, conf, "hand"
            else:
                line.alt_text, line.alt_conf = text, conf

    def _apply_corrections(self, page: Page) -> None:
        if not self.s.apply_corrections:
            return
        corr = load_corrections()
        if corr:
            for b in page.blocks:
                for l in b.lines:
                    l.text = apply_corrections(l.text, corr)

    def _use_vlm(self, lines: list[Line]) -> bool:
        """AI page reader: always ('ai'), never ('classic'), or in 'auto' for handwritten
        pages, where it is far more accurate than the line-by-line pipeline."""
        s = self.s
        if s.reader == "classic" or not vlm.available():
            return False
        if s.reader == "ai" or s.mode == "handwritten":
            return True
        return s.mode == "auto" and looks_handwritten(lines)

    def vlm_model(self):
        key = (self.s.vlm_device, self.s.vlm_max_side)
        if self._vlm is None or self._vlm_key != key:
            self._vlm = None
            if self._ocr is not None or self._struct is not None:
                # Loading the model's weights while PaddlePaddle predictors are alive
                # crashes the process (access violation in the weight copy); drop them
                # first, they are rebuilt on demand.
                import gc

                self._ocr = self._struct = None
                self._ocr_key = self._struct_key = None
                gc.collect()
            self._vlm = vlm.PageReader(device=self.s.vlm_device, max_side=self.s.vlm_max_side)
            self._vlm_key = key
        return self._vlm

    def vlm_device(self) -> str:
        if self._vlm is not None:
            return "GPU" if self._vlm.device == "cuda" else "CPU"
        try:
            import torch

            return "GPU" if self.s.vlm_device != "cpu" and torch.cuda.is_available() else "CPU"
        except Exception:
            return "CPU"

    def _handwriting(self, img: np.ndarray, lines: list[Line], force: bool, notes: Optional[list] = None) -> None:
        """Re-read lines with TrOCR.

        handwritten mode: every line.
        auto mode: first a sample of the longest lines is read by both recognisers. On a
        printed page they agree; on a handwritten page PaddleOCR (trained on print) produces
        confident non-words and the two readings differ, so the whole page is treated as
        handwriting. Otherwise only low-confidence lines are re-read, and TrOCR's reading is
        kept when it is more confident."""
        notes = notes if notes is not None else []
        if not lines:
            return
        try:
            reader = self.hw_model()
        except Exception as e:
            notes.append(f"handwriting model unavailable: {e}")
            return
        done: dict[int, tuple[str, float]] = {}

        def read(ls: list[Line]) -> None:
            todo = [l for l in ls if id(l) not in done]
            if todo:
                res = reader.read([imaging.crop_poly(img, l.poly) for l in todo])
                for l, (t, c) in zip(todo, res):
                    done[id(l)] = (re.sub(r"\s+([.,;:!?)])", lambda m: m.group(1), t), c)

        def adopt(l: Line) -> None:
            t, c = done[id(l)]
            if t:
                l.alt_text, l.alt_conf = l.text, l.conf
                l.text, l.conf, l.source = t, c, "hand"

        # PaddleOCR is trained on print: on handwriting its average confidence drops
        # (about 0.84 on a handwritten notebook page against 0.96-0.98 on printed pages)
        page_hand = force or looks_handwritten(lines)
        if page_hand:
            if not force:
                notes.append("handwritten page detected")
            # The detector splits handwriting into words; TrOCR reads whole lines best
            # (it was trained on lines and uses the context), so words are joined back
            # into lines along their slant before reading.
            lines[:] = _join_words(lines)
            read(lines)
            for l in lines:
                t, c = done[id(l)]
                # TrOCR invents fluent text when it cannot read a line; then its own
                # confidence is low and PaddleOCR's reading is usually closer
                unsure = c < 0.6 and l.conf > c + 0.15
                if t and (force or not unsure):
                    adopt(l)
                elif t:
                    l.alt_text, l.alt_conf = t, c
            return
        low = [l for l in lines if l.conf < self.s.hw_threshold]
        read(low)
        for l in low:
            t, c = done[id(l)]
            if not t:
                continue
            if c > l.conf + 0.03:
                adopt(l)
            else:
                l.alt_text, l.alt_conf = t, c

    def _layout(self, img: np.ndarray) -> list[dict]:
        st = self.struct_model()
        res, _ = st.layout_predictor(img)
        out = []
        for r in res:
            kind = LAYOUT_MAP.get(r["label"].lower(), "text")
            if kind == "table" and not self.s.tables:
                kind = "text"
            if kind == "formula" and not self.s.formulas:
                kind = "text"
            if not self.s.layout and kind not in ("table", "formula"):
                continue
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            out.append({"kind": kind, "bbox": [x1, y1, x2, y2], "score": float(r["score"])})
        # drop regions mostly inside a larger region of the same kind
        out.sort(key=lambda r: -(r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]))
        kept: list[dict] = []
        for r in out:
            if any(_overlap(r["bbox"], k["bbox"]) > 0.8 for k in kept):
                continue
            kept.append(r)
        return kept

    def _formula_lines(self, lines: list[Line], regions: list[dict]) -> list[dict]:
        """Equations that the layout model missed: maths-looking text lines (an '=' or maths
        symbols with few real words, or unusually tall short lines such as fractions)."""
        if not self.s.formulas or not lines:
            return []
        taken = [r["bbox"] for r in regions if r["kind"] in ("table", "formula")]
        heights = sorted(l.bbox[3] - l.bbox[1] for l in lines)
        med = heights[len(heights) // 2] or 1
        out = []
        for l in lines:
            if any(_center_in(l.bbox, t) for t in taken):
                continue
            t = l.text
            words = re.findall(r"[A-Za-z]{4,}", t)
            symbols = sum(t.count(c) for c in "=+^_<>∑∫√±≤≥≈×÷∂∞")
            h = l.bbox[3] - l.bbox[1]
            mathy = ("=" in t and len(words) <= 2 and len(t) < 60) or (symbols >= 2 and len(words) <= 1)
            tall = h > 1.8 * med and len(t) < 40 and len(words) <= 2 and l.conf < 0.97
            if mathy or tall:
                box = _absorb_fragments(list(l.bbox), l, lines, med)
                if not any(_overlap(box, o["bbox"]) > 0.5 for o in out):
                    out.append({"kind": "formula", "bbox": box, "score": 0.5, "auto": True})
        return out[:20]  # bound the time spent on pages full of equations

    def _figure_formula(self, img: np.ndarray, b: Block) -> None:
        """A 'figure' that holds only maths-like fragments (e.g. a cropped equation
        image) is read as a formula."""
        if not self.s.formulas or not b.lines:
            return
        text = " ".join(l.text for l in b.lines)
        words = [w for w in re.findall(r"[A-Za-z]{4,}", text) if w.lower() not in _MATH_WORDS]
        if words or len(text) > 80:
            return
        try:
            latex = self.formula_model().read(imaging.crop_box(img, b.bbox), hint=text)
        except Exception:
            return
        if re.search(r"\\[a-zA-Z]+|[\^_]", latex or ""):
            b.kind, b.latex = "formula", latex

    def _assemble(self, img: np.ndarray, lines: list[Line], regions: list[dict], say) -> list[Block]:
        blocks: list[Block] = []
        used = set()
        regions = regions + self._formula_lines(lines, regions)
        # tables and formulas first: they claim their lines
        for r in sorted(regions, key=lambda r: {"table": 0, "formula": 1}.get(r["kind"], 2)):
            inside = [i for i, l in enumerate(lines) if i not in used and _center_in(l.bbox, r["bbox"])]
            b = Block(r["kind"], r["bbox"], lines=[lines[i] for i in inside])
            words = _real_words(" ".join(l.text for l in b.lines))
            if r["kind"] in ("figure", "formula") and len(words) >= (4 if r["kind"] == "formula" else 15):
                continue  # mislabelled prose (common on ruled notebook pages): becomes paragraphs
            if r["kind"] == "table":
                say("table structure", 0.75)
                self._table(img, b)
            elif r["kind"] == "formula":
                say("formula", 0.85)
                hint = " ".join(l.text for l in b.lines)
                try:
                    b.latex = self.formula_model().read(imaging.crop_box(img, r["bbox"]), hint=hint,
                                                        quick=bool(r.get("auto")))
                except Exception:
                    b.latex = ""
                if not _plausible_latex(b.latex, hint):
                    continue  # unreadable as maths: keep it as ordinary text
            elif r["kind"] == "figure":
                self._figure_formula(img, b)
            if not b.lines and r["kind"] not in ("table", "formula", "figure"):
                continue
            used.update(inside)
            blocks.append(b)
        # lines outside every region -> paragraphs
        rest = [l for i, l in enumerate(lines) if i not in used]
        blocks.extend(_group_paragraphs(rest))
        for b in blocks:
            if b.kind not in ("table", "formula"):
                b.lines = _merge_rows(b.lines)
        return _reading_order(blocks, img.shape[1])

    def _table(self, img: np.ndarray, b: Block) -> None:
        from .tables import parse_table_html

        roi = imaging.crop_box(img, b.bbox, pad=0)
        try:
            res, _ = self.struct_model().table_system(roi, False)
            b.html = res.get("html", "") if isinstance(res, dict) else ""
        except Exception:
            b.html = ""
        b.cells = parse_table_html(b.html) if b.html else []
        if not b.cells and b.lines:  # fallback: rows of lines
            b.cells = [[{"text": l.text, "rowspan": 1, "colspan": 1} for l in row] for row in _rows(b.lines)]


_MATH_WORDS = {"sin", "cos", "tan", "exp", "log", "lim", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
                "max", "min", "sup", "inf", "det", "mod"}


def _erase_rules(crop: np.ndarray) -> np.ndarray:
    """Paint out bits of table rules left inside a cell crop (they read as '1', '|')."""
    if crop.size == 0:
        return crop
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    h, w = bw.shape
    hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(w * 0.6)), 1)))
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(h * 0.85)))))
    # only rules hugging the cell border; long strokes inside the cell are writing
    edge = np.zeros_like(bw)
    m = max(4, min(h, w) // 8)
    edge[:m, :] = edge[-m:, :] = 255
    edge[:, :m] = edge[:, -m:] = 255
    rules = cv2.dilate(cv2.bitwise_and(cv2.bitwise_or(hl, vl), edge), np.ones((3, 3), np.uint8))
    crop[rules > 0] = 255
    return crop


_NUM_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1", "i": "1",
                          "F": "7", "T": "7", "S": "5", "s": "5", "B": "8", "q": "9", "g": "9", "Z": "2",
                          "z": "2", "G": "6", "b": "6", "A": "4", "·": ".", ",": "."})
_NUMBER = re.compile(r"^[-+]?\d+(\.\d+)?%?$")


def _fix_numeric_columns(t) -> None:
    """In a column that is mostly numbers, fix the usual handwriting confusions
    (7 read as F, 0 as O, decimal point as '-' or a space...). A cell is changed
    only when the corrected text is a valid number."""
    by_col: dict[int, list] = {}
    for c in t.cells:
        if c.colspan == 1 and c.text:
            by_col.setdefault(c.c, []).append(c)
    for cells in by_col.values():
        if len(cells) < 4:
            continue
        numeric = sum(1 for c in cells if _NUMBER.match(c.text.replace(" ", "")))
        if numeric / len(cells) < 0.5:
            continue
        for c in cells:
            if _NUMBER.match(c.text):
                continue
            cand = c.text.strip().translate(_NUM_FIX)
            cand = re.sub(r"(?<=\d)\s*[-–—·:]\s*(?=\d)", ".", cand)  # 501-20 -> 501.20
            cand = re.sub(r"(?<=\d)\s+(?=\d{2}$)", ".", cand)       # 79 98 -> 79.98
            cand = cand.replace(" ", "")
            if _NUMBER.match(cand):
                c.text = cand
                for l in c.lines:
                    l.alt_text = l.alt_text or l.text
                if c.lines:
                    c.lines = [c.lines[0].__class__(text=cand, conf=min(l.conf for l in c.lines),
                                                    poly=c.lines[0].poly, source=c.lines[0].source,
                                                    alt_text=" ".join(l.text for l in c.lines))]


def _trim_ink(crop: np.ndarray, pad: int = 6) -> np.ndarray:
    """Crop a cell image to its writing (drops leftover rule pixels at the edges)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = 4
    bw[:m, :] = 0
    bw[-m:, :] = 0
    bw[:, :m] = 0
    bw[:, -m:] = 0
    ys, xs = np.where(bw > 0)
    if len(xs) == 0:
        return crop
    y1, y2 = max(0, ys.min() - pad), min(crop.shape[0], ys.max() + pad)
    x1, x2 = max(0, xs.min() - pad), min(crop.shape[1], xs.max() + pad)
    return crop[y1:y2, x1:x2]


def looks_handwritten(lines: list[Line]) -> bool:
    """PaddleOCR is trained on print: on handwriting its average confidence drops
    (about 0.84 on a handwritten notebook page against 0.96-0.98 on printed pages)."""
    if len(lines) < 5:
        return False
    weights = [max(1, len(l.text)) for l in lines]
    return float(np.average([l.conf for l in lines], weights=weights)) < 0.9


def _order_quad(pts: np.ndarray) -> list:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float64)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return [pts[np.argmin(s)].tolist(), pts[np.argmin(d)].tolist(), pts[np.argmax(s)].tolist(), pts[np.argmax(d)].tolist()]


def _join_words(lines: list[Line]) -> list[Line]:
    """Join word boxes on the same text line (with no big gap between them) into one
    line whose box is the rotated rectangle around all of them, so slanted lines stay
    tight."""
    import cv2

    if not lines:
        return lines
    heights = sorted(l.bbox[3] - l.bbox[1] for l in lines)
    med = max(1, heights[len(heights) // 2])

    def cy(l: Line) -> float:
        return (l.bbox[1] + l.bbox[3]) / 2

    # Walk words left to right and chain each onto the line whose last word sits at
    # the same height just before it. Chaining word to word (instead of comparing with
    # the whole row) follows a line that drifts up or down across the page and never
    # merges neighbouring lines on ruled paper.
    rows: list[list[Line]] = []
    for w in sorted(lines, key=lambda l: l.bbox[0]):
        h = w.bbox[3] - w.bbox[1]
        best, best_d = None, None
        for r in rows:
            last = r[-1]
            lh = last.bbox[3] - last.bbox[1]
            gap = w.bbox[0] - last.bbox[2]
            if gap > 2.5 * med or gap < -1.5 * med:  # slanted words may overlap a little
                continue
            d = abs(cy(w) - cy(last))
            if d < 0.45 * max(1, min(h, lh, 1.5 * med)) and (best_d is None or d < best_d):
                best, best_d = r, d
        if best is None:
            rows.append([w])
        else:
            best.append(w)
    return [_merge_line(r, cv2) for r in rows]


def _merge_line(seg: list[Line], cv2) -> Line:
    if len(seg) == 1:
        return seg[0]
    pts = np.asarray([p for l in seg for p in l.poly], dtype=np.float32)
    quad = _order_quad(cv2.boxPoints(cv2.minAreaRect(pts)))
    return Line(text=" ".join(l.text for l in seg), conf=float(np.mean([l.conf for l in seg])), poly=quad)


def _real_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z]{3,}", text) if w.lower() not in _MATH_WORDS]


def _absorb_fragments(box: list, seed: Line, lines: list[Line], med: float) -> list:
    """Grow a formula box over nearby short fragments (fraction parts, limits, exponents)."""
    changed = True
    while changed:
        changed = False
        for o in lines:
            ob = o.bbox
            if o is seed or len(o.text) > 10 or _overlap(ob, box) > 0.9:
                continue
            vov = min(box[3], ob[3]) - max(box[1], ob[1])
            gap_x = max(ob[0] - box[2], box[0] - ob[2], 0)
            hov = min(box[2], ob[2]) - max(box[0], ob[0])
            near_v = box[1] - 0.8 * med < ob[3] and ob[1] < box[3] + 0.8 * med
            if (vov > 0 and gap_x < 3.5 * med) or (hov > 0 and near_v):
                box[:] = _union([box, ob])
                changed = True
    return box


def _plausible_latex(latex: str, hint: str) -> bool:
    """LaTeX that is maths and shares most of its letters/digits with what the text OCR
    saw in the same region (the formula model invents long garbage on non-formulas)."""
    from .formula import _plain

    if not _looks_like_latex(latex) or len(latex) > 6 * len(hint) + 60:
        return False
    h = _plain(hint)
    if len(h) < 4:
        return True
    return SequenceMatcher(None, _plain(latex), h).ratio() >= 0.4


def _looks_like_latex(s: str) -> bool:
    return bool(s) and bool(re.search(r"\\[a-zA-Z]+|[\^_{}]|=", s))


# ---------------------------------------------------------------- geometry
def _area(b) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _overlap(a, b) -> float:
    """Fraction of a covered by b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(_area(a), 1)


def _center_in(box, region) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] - 3 <= cx <= region[2] + 3 and region[1] - 3 <= cy <= region[3] + 3


def _rows(lines: list[Line]) -> list[list[Line]]:
    """Group lines whose vertical extents overlap into rows (left to right)."""
    rows: list[list[Line]] = []
    for l in sorted(lines, key=lambda l: (l.bbox[1] + l.bbox[3]) / 2):
        b = l.bbox
        h = b[3] - b[1]
        for row in rows:
            rb = [min(x.bbox[1] for x in row), max(x.bbox[3] for x in row)]
            ov = min(b[3], rb[1]) - max(b[1], rb[0])
            if ov > 0.5 * min(h, rb[1] - rb[0]):
                row.append(l)
                break
        else:
            rows.append([l])
    return [sorted(r, key=lambda l: l.bbox[0]) for r in rows]


def _merge_rows(lines: list[Line]) -> list[Line]:
    out = []
    for row in _rows(lines):
        if len(row) == 1:
            out.append(row[0])
            continue
        xs = [p[0] for l in row for p in l.poly]
        ys = [p[1] for l in row for p in l.poly]
        out.append(Line(
            text=" ".join(l.text for l in row), conf=float(np.mean([l.conf for l in row])),
            poly=[[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]],
            source="hand" if any(l.source == "hand" for l in row) else "print"))
    return out


def _group_paragraphs(lines: list[Line]) -> list[Block]:
    blocks: list[list[Line]] = []
    for row in _rows(lines):
        for l in row:
            b = l.bbox
            h = b[3] - b[1]
            best = None
            for blk in blocks:
                bb = _union([x.bbox for x in blk])
                gap = b[1] - bb[3]
                hov = min(b[2], bb[2]) - max(b[0], bb[0])
                if -0.5 * h < gap < 1.0 * h and hov > 0:
                    best = blk
                    break
            if best is None:
                blocks.append([l])
            else:
                best.append(l)
    return [Block("text", _union([l.bbox for l in blk]), lines=blk) for blk in blocks]


def _union(boxes) -> list[int]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _reading_order(blocks: list[Block], width: int) -> list[Block]:
    """Top-to-bottom; blocks in the left half of a two-column band come first."""
    full, left, right = [], [], []
    for b in blocks:
        x1, _, x2, _ = b.bbox
        if x2 < width * 0.55 and (x2 - x1) < width * 0.5:
            left.append(b)
        elif x1 > width * 0.45 and (x2 - x1) < width * 0.5:
            right.append(b)
        else:
            full.append(b)
    if not left or not right:
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    out: list[Block] = []
    cols = sorted(left + right, key=lambda b: b.bbox[1])
    full.sort(key=lambda b: b.bbox[1])
    # emit columns between consecutive full-width blocks
    lids = {id(b) for b in left}
    bounds = [-1] + [f.bbox[1] for f in full] + [10 ** 9]
    for i in range(len(bounds) - 1):
        seg = [b for b in cols if bounds[i] <= b.bbox[1] < bounds[i + 1]]
        out.extend(sorted([b for b in seg if id(b) in lids], key=lambda b: b.bbox[1]))
        out.extend(sorted([b for b in seg if id(b) not in lids], key=lambda b: b.bbox[1]))
        if i < len(full):
            out.append(full[i])
    return out
