# PyInstaller spec for OfflineOCR.exe  (build with tools\build_exe.bat)
# One-folder build: dist\OfflineOCR\OfflineOCR.exe + _internal\ ; the model and
# data folders (models, user, datasets, output) are copied next to the exe by the
# build script, so the exe works fully offline.
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
SP = Path(sys.prefix) / "Lib" / "site-packages"
PADDLEOCR_DIR = SP / "paddleocr"

datas, binaries, hiddenimports = [], [], []


def add_all(pkg):
    d, b, h = collect_all(pkg)
    datas.extend(d)
    binaries.extend(b)
    hiddenimports.extend(h)


for pkg in ("paddle", "rapid_latex_ocr", "latex2mathml", "docx", "pypdfium2", "pypdfium2_raw",
            "onnxruntime", "shapely", "pyclipper", "rapidfuzz"):
    add_all(pkg)

# paddleocr puts its own folder on sys.path and imports tools/ppocr/ppstructure
# as top-level packages; ship that folder as plain source files so those imports
# resolve from disk (they cannot be analysed: importing them needs the stubs in
# ocrapp/compat.py)
datas.append((str(PADDLEOCR_DIR), "paddleocr"))
# scikit-image: paddleocr imports it at load time; shipped as files too (its
# PyInstaller hook imports every submodule, and Smart App Control blocks some DLLs)
datas.append((str(SP / "skimage"), "skimage"))

# transformers loads model classes lazily by name
hiddenimports += collect_submodules("transformers.models.glm_ocr")
hiddenimports += collect_submodules("transformers.models.glm4v")
hiddenimports += collect_submodules("transformers.models.vision_encoder_decoder")
hiddenimports += collect_submodules("transformers.models.trocr")
hiddenimports += collect_submodules("transformers.models.vit")
hiddenimports += collect_submodules("transformers.models.deit")
hiddenimports += collect_submodules("transformers.models.xlm_roberta")
hiddenimports += collect_submodules("transformers.models.roberta")
hiddenimports += collect_submodules("transformers.models.auto")
hiddenimports += collect_submodules("transformers.generation")
datas += collect_data_files("transformers", include_py_files=True)
hiddenimports += ["accelerate", "safetensors", "tokenizers", "sentencepiece",
                  # imported by the paddleocr sources shipped as data
                  "tqdm", "six", "requests", "bs4", "yaml", "openpyxl", "lxml", "scipy", "scipy.special",
                  "PIL.ImageDraw", "PIL.ImageFont", "attr", "fire", "fontTools", "lazy_loader",
                  "scipy.ndimage", "imageio", "tifffile", "networkx"]

# the app's theme
datas.append((str(ROOT / "ocrapp" / "gui" / "styles"), "ocrapp/gui/styles"))

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # training-only imports of paddleocr: replaced by stubs in ocrapp/compat.py
    # (scikit-image's DLLs are blocked by Smart App Control on this PC)
    excludes=["paddleocr", "tools", "ppocr", "ppstructure", "imgaug", "lmdb", "skimage", "tensorflow", "IPython", "jupyter", "notebook", "pytest",
              "spacy", "sklearn", "paddlex", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OfflineOCR",
    console=False,
    icon=None,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="OfflineOCR")
