"""Try a page-level OCR vision-language model on rendered pages and score it.

    python tests/vlm_test.py <model_dir> "<prompt>" <max_side> <pages...>
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ["HF_HUB_OFFLINE"] = "1"

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

from score import score  # noqa: E402


def main() -> None:
    mdir, prompt, max_side = sys.argv[1], sys.argv[2], int(sys.argv[3])
    pages = [int(p) for p in sys.argv[4:]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.bfloat16
    t = time.time()
    proc = AutoProcessor.from_pretrained(mdir)
    model = AutoModelForImageTextToText.from_pretrained(mdir, dtype=dtype).to(dev).eval()
    print(f"load {time.time() - t:.0f}s on {dev}", flush=True)
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(r"C:/CDTRS-main/OCR_new/input/ChemicalKinetics.pdf")
    name = Path(mdir).name
    for p in pages:
        img = pdf[p - 1].render(scale=200 / 72).to_pil().convert("RGB")
        img.thumbnail((max_side, max_side))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                          return_tensors="pt").to(dev)
        t = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1500, do_sample=False)
        txt = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        Path(ROOT / "output" / f"{name}_p{p}.txt").write_text(txt, encoding="utf-8")
        gt = (ROOT / "tests" / "gt" / f"ck_p{p}.txt")
        s = score(txt, gt.read_text(encoding="utf-8")) if gt.exists() else {}
        print(f"page {p}: {time.time() - t:.0f}s {s}", flush=True)
        print(txt[:1500], flush=True)


if __name__ == "__main__":
    main()
