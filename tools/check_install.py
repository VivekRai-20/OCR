"""Verifies an installation: packages import, model files exist, and (with --ocr) a test page is read.

    .venv\\Scripts\\python tools\\check_install.py [--ocr]
"""
from __future__ import annotations

import importlib
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from ocrapp import compat

    compat.prepare()
    ok = True
    for mod in ("paddle", "paddleocr", "torch", "transformers", "onnxruntime", "rapid_latex_ocr", "cv2", "fitz",
                "pypdfium2", "docx", "openpyxl", "latex2mathml", "matplotlib", "PySide6.QtWidgets"):
        try:
            importlib.import_module(mod)
            print(f"  ok   {mod}")
        except Exception as e:
            ok = False
            print(f"  FAIL {mod}: {e}")
    from ocrapp.paths import FORMULA, HANDWRITING, PADDLE

    need = [PADDLE / d / "inference.pdiparams" for d in ("det", "det_v4", "rec", "cls", "table", "layout")]
    need += [FORMULA / f for f in ("image_resizer.onnx", "encoder.onnx", "decoder.onnx", "tokenizer.json")]
    need += [HANDWRITING / "trocr-small-handwritten" / "config.json"]
    need += [ROOT / "models" / "vlm" / "glm-ocr" / "model.safetensors"]
    try:
        import torch

        print(f"  AI page reader device: {'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (slow)'}")
    except Exception:
        pass
    for f in need:
        if not f.exists():
            ok = False
            print(f"  MISSING {f.relative_to(ROOT)}")
    print("  models: all present" if all(f.exists() for f in need) else "  models: incomplete")

    if "--ocr" in sys.argv:
        # prove that nothing reaches the network: any connection attempt raises
        def _blocked(*a, **k):
            raise OSError("network access attempted (should be offline)")

        socket.socket.connect = _blocked  # type: ignore[assignment]
        from ocrapp.engine import OCREngine

        sample = ROOT / "tests" / "samples" / "printed_page.png"
        t = time.time()
        doc = OCREngine().process_file(str(sample))
        txt = doc.text
        print(f"  OCR test ({time.time() - t:.0f}s, network blocked): {len(txt)} chars")
        for kind in sorted({b.kind for b in doc.pages[0].blocks}):
            print(f"     block type: {kind}")
        ok = ok and "Arrhenius" in txt
    print("INSTALL OK" if ok else "INSTALL HAS PROBLEMS")
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
