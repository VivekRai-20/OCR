"""Handwritten text-line recognition with a local TrOCR model (never downloads).

models/handwriting/
    active.txt                    folder name of the model in use
    trocr-small-handwritten/      base model
    <name>/                       fine-tuned models made in the Fine-tune tab
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .paths import HANDWRITING


def is_model_dir(folder: Path) -> bool:
    return folder.is_dir() and (folder / "config.json").exists() and any(
        (folder / n).exists() for n in ("model.safetensors", "pytorch_model.bin"))


def list_models() -> list[str]:
    if not HANDWRITING.is_dir():
        return []
    return sorted(f.name for f in HANDWRITING.iterdir() if is_model_dir(f))


def active_model() -> str:
    f = HANDWRITING / "active.txt"
    name = f.read_text(encoding="utf-8").strip() if f.exists() else ""
    models = list_models()
    if name in models:
        return name
    return models[0] if models else ""


def set_active_model(name: str) -> None:
    (HANDWRITING / "active.txt").write_text(name, encoding="utf-8")


def load_trocr(model_dir: Path, device: str = "cpu"):
    """Processor + model from a local folder. The tokenizer class is taken from
    tokenizer_config.json (the auto loader cannot build trocr-small's
    XLM-R tokenizer from the sentencepiece file)."""
    import transformers
    from transformers import AutoImageProcessor, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    model_dir = Path(model_dir)
    image_processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    tokenizer = None
    cfg = model_dir / "tokenizer_config.json"
    if cfg.exists():
        cls_name = (json.loads(cfg.read_text(encoding="utf-8")).get("tokenizer_class") or "").strip()
        tcls = getattr(transformers, cls_name, None) or getattr(transformers, cls_name.replace("Fast", ""), None)
        if tcls is not None:
            try:
                tokenizer = tcls.from_pretrained(str(model_dir), local_files_only=True)
            except Exception:
                tokenizer = None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(str(model_dir), local_files_only=True)

    def first(*vals):
        return next((v for v in vals if v is not None), None)

    gen, conf = model.generation_config, model.config
    start = first(getattr(gen, "decoder_start_token_id", None), getattr(conf, "decoder_start_token_id", None),
                  getattr(getattr(conf, "decoder", None), "decoder_start_token_id", None),
                  tokenizer.cls_token_id, tokenizer.bos_token_id)
    pad = first(getattr(gen, "pad_token_id", None), getattr(conf, "pad_token_id", None), tokenizer.pad_token_id)
    eos = first(getattr(gen, "eos_token_id", None), getattr(conf, "eos_token_id", None),
                tokenizer.sep_token_id, tokenizer.eos_token_id)
    gen.decoder_start_token_id, gen.pad_token_id, gen.eos_token_id = start, pad, eos
    for k, v in (("decoder_start_token_id", start), ("pad_token_id", pad), ("eos_token_id", eos)):
        try:
            setattr(conf, k, v)
        except Exception:
            pass
    model.to(device).eval()
    return processor, model


def prepare_line_image(img: np.ndarray) -> np.ndarray:
    """Grayscale, remove ruled lines, and trim the crop to the main text band.
    Same step is used for training and for reading. Returns RGB."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = g.shape[:2]
    if h < 8 or w < 8:
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    _, ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 4), 1)))
    rules = cv2.dilate(rules, np.ones((3, 1), np.uint8))
    text = cv2.subtract(ink, rules)
    bg = int(np.median(g[ink == 0])) if (ink == 0).any() else 255
    g[rules > 0] = bg
    rows = text.sum(axis=1).astype(float)
    if rows.sum() == 0:
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    k = max(3, h // 12) | 1
    smooth = np.convolve(rows, np.ones(k) / k, mode="same")
    inside = smooth > 0.12 * smooth.max()
    bands, s = [], None
    for i, f in enumerate(inside):
        if f and s is None:
            s = i
        elif not f and s is not None:
            bands.append((s, i))
            s = None
    if s is not None:
        bands.append((s, h))

    def weight(b):
        c = (b[0] + b[1]) / 2.0
        return rows[b[0]:b[1]].sum() * (1.0 - 0.5 * abs(c - h / 2.0) / (h / 2.0))

    top, bottom = max(bands, key=weight)
    pad = int(0.25 * (bottom - top)) + 2
    top, bottom = max(0, top - pad), min(h, bottom + pad)
    cols = np.where(text[top:bottom].sum(axis=0) > 0)[0]
    left, right = (max(0, cols[0] - pad), min(w, cols[-1] + pad + 1)) if cols.size else (0, w)
    return cv2.cvtColor(g[top:bottom, left:right], cv2.COLOR_GRAY2RGB)


class HandwritingReader:
    def __init__(self, model_name: str = "", beams: int = 4, device: str = "cpu") -> None:
        name = model_name or active_model()
        if not name:
            raise FileNotFoundError(f"No handwriting model found in {HANDWRITING}")
        self.name = name
        self.beams = beams
        self.device = device
        self.processor, self.model = load_trocr(HANDWRITING / name, device)

    def read(self, crops: list[np.ndarray], batch: int = 8) -> list[tuple[str, float]]:
        import torch
        from PIL import Image

        prepared = [prepare_line_image(c) for c in crops]
        # batch lines of similar length together and stop generation at a length that
        # fits the line (a few tokens per character-height of width): much faster on CPU
        order = sorted(range(len(prepared)), key=lambda i: prepared[i].shape[1] / max(1, prepared[i].shape[0]))
        results: dict[int, tuple[str, float]] = {}
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            chunk = [Image.fromarray(prepared[j]) for j in idx]
            aspect = max(prepared[j].shape[1] / max(1, prepared[j].shape[0]) for j in idx)
            max_new = int(min(96, 12 + 3.5 * aspect))
            pv = self.processor(images=chunk, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                g = self.model.generate(pv, num_beams=self.beams, max_new_tokens=max_new,
                                        no_repeat_ngram_size=3, repetition_penalty=1.2,
                                        return_dict_in_generate=True, output_scores=True)
            texts = self.processor.batch_decode(g.sequences, skip_special_tokens=True)
            if getattr(g, "sequences_scores", None) is not None:
                confs = [float(math.exp(s)) for s in g.sequences_scores.cpu().tolist()]
            else:
                ts = self.model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True)
                confs = [float(torch.exp(r[torch.isfinite(r)].mean())) if torch.isfinite(r).any() else 0.0 for r in ts]
            for j, t, c in zip(idx, texts, confs):
                results[j] = (t.strip(), min(1.0, c))
        return [results[j] for j in range(len(prepared))]


def model_meta(name: str) -> dict:
    f = HANDWRITING / name / "model_meta.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}
