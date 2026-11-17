"""
Fine-tuning the handwriting recogniser (TrOCR) on the user's own line images.

Dataset: datasets/handwriting/images/*.png + datasets/handwriting/labels.tsv
         (one "filename<TAB>text" per line). Lines can be added from any OCR'd
         document after correcting them in the editor, or imported from a folder.
Output:  models/handwriting/<name>/ (Hugging Face format + model_meta.json)
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import imaging
from .handwriting import HANDWRITING, is_model_dir, load_trocr, prepare_line_image
from .paths import HW_DATASET

IMAGES = HW_DATASET / "images"
LABELS = HW_DATASET / "labels.tsv"
IMAGES.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- dataset
@dataclass
class Sample:
    file: str
    text: str

    @property
    def path(self) -> Path:
        return IMAGES / self.file


def load_samples() -> list[Sample]:
    if not LABELS.exists():
        return []
    out = []
    with open(LABELS, encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 2 and (IMAGES / row[0]).exists():
                out.append(Sample(row[0], "\t".join(row[1:])))
    return out


def save_samples(samples: list[Sample]) -> None:
    with open(LABELS, "w", encoding="utf-8", newline="") as f:
        for s in samples:
            f.write(f"{s.file}\t{s.text.replace(chr(9), ' ').replace(chr(10), ' ')}\n")


def add_sample(img: np.ndarray, text: str, samples: Optional[list[Sample]] = None) -> Sample:
    name = f"{time.strftime('%Y%m%d')}_{uuid.uuid4().hex[:10]}.png"
    cv2.imencode(".png", img)[1].tofile(str(IMAGES / name))
    s = Sample(name, text.strip())
    if samples is not None:
        samples.append(s)
    return s


def add_from_document(doc, only_handwritten: bool = False) -> int:
    """Every recognised text line of a document (with its current, possibly corrected, text)."""
    samples = load_samples()
    n = 0
    for p in doc.pages:
        for b in p.blocks:
            if b.kind in ("table", "formula", "figure"):
                continue
            for l in b.lines:
                if not l.text.strip() or (only_handwritten and l.source != "hand"):
                    continue
                add_sample(imaging.crop_poly(p.image, l.poly), l.text, samples)
                n += 1
    save_samples(samples)
    return n


def import_folder(folder: str | Path) -> int:
    """Import line images with labels. Accepted layouts:
    * labels.tsv / labels.txt / gt.txt  with "image<TAB>text" lines, or
    * image files with a .txt file of the same name holding the text."""
    folder = Path(folder)
    samples = load_samples()
    n = 0
    for lf in ("labels.tsv", "labels.txt", "gt.txt", "gt.tsv"):
        f = folder / lf
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if "\t" not in line:
                    continue
                name, text = line.split("\t", 1)
                ip = folder / name
                if ip.exists():
                    add_sample(imaging.imread(ip), text, samples)
                    n += 1
            save_samples(samples)
            return n
    for ip in sorted(folder.iterdir()):
        if ip.suffix.lower() in imaging.IMAGE_EXT:
            tp = ip.with_suffix(".txt")
            if not tp.exists():
                tp = ip.with_suffix(".gt.txt")
            if tp.exists():
                add_sample(imaging.imread(ip), tp.read_text(encoding="utf-8").strip(), samples)
                n += 1
    save_samples(samples)
    return n


def delete_samples(files: set[str]) -> None:
    keep = []
    for s in load_samples():
        if s.file in files:
            try:
                s.path.unlink()
            except OSError:
                pass
        else:
            keep.append(s)
    save_samples(keep)


# ----------------------------------------------------------------- metrics
def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(preds: list[str], refs: list[str]) -> float:
    errs = sum(levenshtein(p, r) for p, r in zip(preds, refs))
    return errs / max(1, sum(len(r) for r in refs))


def wer(preds: list[str], refs: list[str]) -> float:
    errs = sum(levenshtein_tokens(p.split(), r.split()) for p, r in zip(preds, refs))
    return errs / max(1, sum(len(r.split()) for r in refs))


def levenshtein_tokens(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ----------------------------------------------------------------- training
def _augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    if rng.random() < 0.7:  # small rotation / shear (slanted handwriting)
        ang = rng.uniform(-4, 4)
        shear = rng.uniform(-0.25, 0.25)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, rng.uniform(0.9, 1.08))
        m[0, 1] += shear
        m[0, 2] -= shear * h / 2
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng.random() < 0.3:  # stroke width
        k = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)
    if rng.random() < 0.3:
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


@dataclass
class TrainConfig:
    base: str = "trocr-small-handwritten"
    name: str = "my-handwriting"
    epochs: int = 8
    batch_size: int = 4
    lr: float = 3e-5
    val_fraction: float = 0.15
    augment: bool = True
    freeze_encoder: bool = False
    seed: int = 42


def _labels(tok, texts: list[str], max_len: int = 96):
    """Token ids to predict. TrOCR generates right after the decoder start token, so the
    <s> the tokenizer puts first is not a target (keeping it gives a loss ~6x too high)."""
    import torch

    rows = []
    for t in texts:
        ids = tok(t, add_special_tokens=True).input_ids
        if len(ids) > 1 and ids[0] in (tok.bos_token_id, tok.cls_token_id):
            ids = ids[1:]
        rows.append(ids[:max_len])
    out = torch.full((len(rows), max(len(r) for r in rows)), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
    return out


def _predict(processor, model, imgs: list[np.ndarray], device: str = "cpu", bs: int = 8) -> list[str]:
    import torch
    from PIL import Image

    out = []
    for i in range(0, len(imgs), bs):
        pil = [Image.fromarray(x) for x in imgs[i:i + bs]]
        pv = processor(images=pil, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            g = model.generate(pv, num_beams=1, max_new_tokens=96)
        out += [t.strip() for t in processor.batch_decode(g, skip_special_tokens=True)]
    return out


def train(cfg: TrainConfig, log: Callable[[str], None], progress: Callable[[float], None],
          should_stop: Callable[[], bool] = lambda: False) -> dict:
    import torch

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    samples = load_samples()
    if len(samples) < 4:
        raise ValueError("Need at least 4 labelled line images (have %d)." % len(samples))
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * cfg.val_fraction))
    val, tr = samples[:n_val], samples[n_val:]
    log(f"{len(tr)} training lines, {len(val)} validation lines")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = HANDWRITING / cfg.base
    log(f"Loading base model {cfg.base} on {device} ...")
    processor, model = load_trocr(base_dir, device)
    tok = processor.tokenizer
    if cfg.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    def prep(s: Sample) -> np.ndarray:
        return prepare_line_image(imaging.imread(s.path))

    val_imgs = [prep(s) for s in val]
    val_refs = [s.text for s in val]
    tr_imgs = [prep(s) for s in tr]
    base_preds = _predict(processor, model, val_imgs, device)
    base_cer = cer(base_preds, val_refs)
    log(f"Base model CER on validation: {base_cer:.2%}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.01)
    steps = cfg.epochs * ((len(tr) + cfg.batch_size - 1) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.05, 1 - s / steps))
    out_dir = HANDWRITING / cfg.name
    tmp_dir = HANDWRITING / (cfg.name + ".tmp")
    best = base_cer
    best_epoch = 0
    step = 0
    from PIL import Image

    for ep in range(1, cfg.epochs + 1):
        model.train()
        order = list(range(len(tr)))
        rng.shuffle(order)
        losses = []
        for i in range(0, len(order), cfg.batch_size):
            if should_stop():
                log("Stopped by user.")
                break
            idx = order[i:i + cfg.batch_size]
            imgs = [_augment(tr_imgs[j], rng) if cfg.augment else tr_imgs[j] for j in idx]
            pv = processor(images=[Image.fromarray(x) for x in imgs], return_tensors="pt").pixel_values.to(device)
            lab = _labels(tok, [tr[j].text for j in idx])
            loss = model(pixel_values=pv, labels=lab.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(float(loss))
            step += 1
            progress(step / steps)
        if should_stop():
            break
        model.eval()
        preds = _predict(processor, model, val_imgs, device)
        c = cer(preds, val_refs)
        log(f"Epoch {ep}/{cfg.epochs}: loss {np.mean(losses):.3f}, validation CER {c:.2%}")
        if c < best or best_epoch == 0 and c <= best:
            best, best_epoch = c, ep
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            model.save_pretrained(str(tmp_dir))
            processor.save_pretrained(str(tmp_dir))
    result = {"base_cer": base_cer, "best_cer": best, "best_epoch": best_epoch}
    if best_epoch and tmp_dir.exists():
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_dir.rename(out_dir)
        meta = {"version": cfg.name, "base_model": cfg.base, "kind": "fine-tuned",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "training_samples": len(tr),
                "validation_samples": len(val), "metrics": {"val_cer": best, "base_val_cer": base_cer},
                "epochs": cfg.epochs, "best_epoch": best_epoch, "lr": cfg.lr}
        (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log(f"Saved {out_dir.name}: CER {base_cer:.2%} -> {best:.2%}")
        result["model"] = cfg.name
    else:
        log("The fine-tuned model did not beat the base model on validation; nothing saved.")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return result


def evaluate(model_name: str, log: Callable[[str], None] = print) -> dict:
    """CER/WER of a model on the whole dataset."""
    samples = load_samples()
    processor, model = load_trocr(HANDWRITING / model_name)
    imgs = [prepare_line_image(imaging.imread(s.path)) for s in samples]
    preds = _predict(processor, model, imgs)
    refs = [s.text for s in samples]
    r = {"model": model_name, "samples": len(samples), "cer": cer(preds, refs), "wer": wer(preds, refs)}
    log(f"{model_name}: CER {r['cer']:.2%}, WER {r['wer']:.2%} on {len(samples)} lines")
    return r


def delete_model(name: str) -> None:
    d = HANDWRITING / name
    if is_model_dir(d) and name != "trocr-small-handwritten":
        shutil.rmtree(d)
