"""
Import-order and compatibility shims. Import this module before paddle.

* On Windows, torch must load its DLLs before paddle does, or the process can
  crash (same finding as CDTRS OCR_new/utils/native_libs.py).
* PaddleOCR 2.8 imports training-only packages (imgaug, lmdb) even for
  inference. On PCs where Windows Application Control / Smart App Control
  blocks one of their native DLLs (seen with scikit-image's _watershed_cy on
  this PC), the whole import fails. Inference never uses them, so a small
  stand-in module is registered when the real one cannot be imported.
"""
from __future__ import annotations

import importlib
import os
import sys
import types
import warnings

_done = False


def _stub(name: str) -> None:
    mod = types.ModuleType(name)
    mod.__dict__["__stub__"] = True

    mod.__file__ = None  # type: ignore[assignment]

    class _Missing:
        def __init__(self, *_a, **_k):
            raise RuntimeError(f"{name} is not available on this PC (training-only dependency).")

    def _getattr(attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Missing

    mod.__getattr__ = _getattr  # type: ignore[attr-defined]
    sys.modules[name] = mod


def prepare() -> None:
    global _done
    if _done:
        return
    _done = True
    warnings.filterwarnings("ignore")
    os.environ.setdefault("FLAGS_minloglevel", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # threaded weight loading crashes (access violation) once PaddlePaddle is loaded
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if sys.platform == "win32":
        try:
            import torch  # noqa: F401  (load torch DLLs before paddle)
        except Exception:
            pass
    for name, subs in (("imgaug", ("imgaug.augmenters",)), ("lmdb", ())):
        try:
            importlib.import_module(name)
        except Exception:
            _stub(name)
            for s in subs:
                _stub(s)
            if subs:
                sys.modules[name].augmenters = sys.modules[subs[0]]  # type: ignore[attr-defined]
