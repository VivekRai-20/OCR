"""Math formula image -> LaTeX, using the ONNX export of LaTeX-OCR (RapidLaTeXOCR).
Model files live in models/formula; nothing is downloaded.

LaTeX-OCR is very sensitive to the size of the symbols, so each formula is read
at several scales and the reading that best agrees with the plain-text OCR of
the same region (and with the other readings) is kept.
"""
from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

import cv2
import numpy as np

from .paths import FORMULA

_ODD = re.compile(r"\\(underline|begin\{array\}|mathfrak|cal|boldsymbol|bigoplus|bigwedge|longrightarrow|textstyle|mathcal)"
                  r"|\\(in|Theta|in\{\})\\times")  # "exp" misread as \in\times / \Theta\times


def available() -> bool:
    return all((FORMULA / f).exists() for f in ("image_resizer.onnx", "encoder.onnx", "decoder.onnx", "tokenizer.json"))


def _plain(s: str) -> str:
    """Letters/digits/=+- only, for comparing a LaTeX reading with plain OCR text.
    Function names (\\exp, \\sin, ...) are kept because the plain OCR sees them."""
    s = re.sub(r"\\(exp|sin|cos|tan|log|ln|lim|max|min|det)(?![a-zA-Z])", r" \1 ", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    return re.sub(r"[^A-Za-z0-9=+\-]", "", s).lower()


def _collapse_repeats(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(.{6,}?)(?:-?\1)+", lambda m: m.group(1), s)
    return s


def _trim(img: np.ndarray, pad: int = 6) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(ink > 0)
    if len(xs) == 0:
        return img
    return img[max(0, ys.min() - pad):ys.max() + pad + 1, max(0, xs.min() - pad):xs.max() + pad + 1]


def _ink_height(img: np.ndarray) -> float:
    """Median height of ink blobs ~ the symbol size."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    hs = [s[3] for s in stats[1:] if s[3] > 3 and s[4] > 6]
    return float(np.median(hs)) if hs else float(img.shape[0])


class FormulaReader:
    def __init__(self) -> None:
        from rapid_latex_ocr import LaTeXOCR

        self._m = LaTeXOCR(
            image_resizer_path=str(FORMULA / "image_resizer.onnx"),
            encoder_path=str(FORMULA / "encoder.onnx"),
            decoder_path=str(FORMULA / "decoder.onnx"),
            tokenizer_json=str(FORMULA / "tokenizer.json"),
        )

    def _once(self, img: np.ndarray) -> str:
        _, enc = cv2.imencode(".png", img)
        latex, _ = self._m(enc.tobytes())
        return latex.strip()

    def read(self, img_bgr: np.ndarray, hint: str = "", max_tokens: int = 0, quick: bool = False) -> str:
        # The ONNX decoder has no cache, so cost grows with the square of the length:
        # cap the length by what the plain-text OCR saw in the region.
        n = max_tokens or min(400, max(80, 8 * len(hint) + 40))
        self._m.encoder_decoder.max_seq_len = n
        return self._read(img_bgr, hint, quick)

    def _read(self, img_bgr: np.ndarray, hint: str, quick: bool) -> str:
        img = img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        img = _trim(img)
        if img.shape[0] < 8 or img.shape[1] < 8:
            return ""
        sym = _ink_height(img)
        # rendered training formulas have symbols roughly 8-20 px tall
        targets = (12.0, 16.0) if quick else (9.0, 12.0, 15.0, 19.0)
        scales = sorted({round(t / sym, 3) for t in targets} | ({1.0} if not quick else set()))
        scales = [s for s in scales if 0.15 <= s <= 3.0] or [1.0]
        cands = []
        for s in scales:
            im = img if s == 1.0 else cv2.resize(img, None, fx=s, fy=s,
                                                  interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            try:
                cands.append(self._once(im))
            except Exception:
                continue
        # the decoder sometimes loops, emitting the same group again and again
        cands = [_collapse_repeats(c) for c in cands if c]
        if not cands:
            return ""
        votes = Counter(_plain(c) for c in cands)
        h = _plain(hint)

        def score(c: str) -> float:
            p = _plain(c)
            agree = (votes[p] - 1) / max(1, len(cands) - 1)
            sim = SequenceMatcher(None, p, h).ratio() if h else 0.0
            odd = len(_ODD.findall(c))
            rep = 1 if re.search(r"(.{8,})\1", c) else 0  # decoder loops
            return sim + 0.6 * agree - 0.15 * odd - 0.5 * rep - 0.002 * len(c)

        return max(cands, key=score)
