"""
tools/fetch_assets.py
---------------------
ONE-TIME, ONLINE step for the developer. Fetches every model weight and wheel
the app needs into this folder, so that the whole project can be copied to an
offline PC and installed there with install.bat (no internet).

    python tools/fetch_assets.py            # models + wheels
    python tools/fetch_assets.py --models   # models only
    python tools/fetch_assets.py --wheels   # wheels only

Models that already exist are skipped. The app itself never downloads anything.
"""
from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

PADDLE = {
    # folder name            : url of the inference tar
    "det": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
    "det_v4": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
    "rec": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
    "cls": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
    "table": "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar",
    "layout": "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar",
}
LATEX_URL = "https://github.com/RapidAI/RapidLaTeXOCR/releases/download/v0.0.0"
LATEX_FILES = ["image_resizer.onnx", "encoder.onnx", "decoder.onnx", "tokenizer.json"]
TROCR_REPO = "microsoft/trocr-small-handwritten"


def _get(url: str) -> bytes:
    print("  GET", url, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "offline-ocr-fetch"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def fetch_paddle() -> None:
    for name, url in PADDLE.items():
        dest = MODELS / "paddle" / name
        if any(dest.glob("*.pdiparams")):
            print(f"[paddle/{name}] present")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(_get(url))) as tar:
            for m in tar.getmembers():
                if m.isfile():
                    # PaddleOCR only treats a folder as "installed" when it holds
                    # inference.pdiparams; model.* files would trigger a download.
                    fname = Path(m.name).name.replace("model.pd", "inference.pd")
                    (dest / fname).write_bytes(tar.extractfile(m).read())
        print(f"[paddle/{name}] ok")


def fetch_latex() -> None:
    dest = MODELS / "formula"
    dest.mkdir(parents=True, exist_ok=True)
    for f in LATEX_FILES:
        if (dest / f).exists():
            continue
        (dest / f).write_bytes(_get(f"{LATEX_URL}/{f}"))
    print("[formula] ok")


def fetch_trocr() -> None:
    dest = MODELS / "handwriting" / "trocr-small-handwritten"
    if (dest / "config.json").exists():
        print("[handwriting] present")
    else:
        local = Path(r"C:\CDTRS-main\OCR_new\models\handwriting\trocr-small-handwritten")
        if local.exists():
            shutil.copytree(local, dest, ignore=shutil.ignore_patterns(".cache"))
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(TROCR_REPO, local_dir=str(dest))
        print("[handwriting] ok")
    active = MODELS / "handwriting" / "active.txt"
    if not active.exists():
        active.write_text("trocr-small-handwritten", encoding="utf-8")


def fetch_vlm() -> None:
    dest = MODELS / "vlm" / "glm-ocr"
    if (dest / "model.safetensors").exists():
        print("[vlm/glm-ocr] present")
        return
    from huggingface_hub import snapshot_download

    snapshot_download("zai-org/GLM-OCR", local_dir=str(dest), allow_patterns=["*.json", "*.jinja", "*.safetensors"])
    print("[vlm/glm-ocr] ok")


def fetch_wheels() -> None:
    wheels = ROOT / "wheels"
    wheels.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", "-r", str(ROOT / "requirements.txt"),
           "-d", str(wheels), "--only-binary=:all:",
           "--extra-index-url", "https://download.pytorch.org/whl/cpu"]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    # CUDA build of torch for PCs with an NVIDIA GPU (install.bat picks it when nvidia-smi exists)
    subprocess.check_call([sys.executable, "-m", "pip", "download", "torch==2.13.0", "torchvision==0.28.0",
                           "--no-deps", "--only-binary=:all:", "-d", str(wheels / "gpu"),
                           "--index-url", "https://download.pytorch.org/whl/cu126"])
    # pip/setuptools/wheel so the venv can be bootstrapped offline too
    subprocess.check_call([sys.executable, "-m", "pip", "download", "pip", "setuptools", "wheel",
                           "-d", str(wheels), "--only-binary=:all:"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--wheels", action="store_true")
    a = ap.parse_args()
    both = not (a.models or a.wheels)
    if a.models or both:
        fetch_paddle()
        fetch_latex()
        fetch_trocr()
        fetch_vlm()
    if a.wheels or both:
        fetch_wheels()


if __name__ == "__main__":
    main()
