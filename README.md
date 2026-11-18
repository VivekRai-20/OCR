# Offline OCR

A Windows desktop OCR app that runs **completely offline**. Everything it needs, including Python packages (`wheels/`), model weights (`models/`) and a Python installer (`installers/`), lives in this folder. To use it on another PC, copy the whole folder (or zip it) and run `install.bat`. No internet is needed on that PC.

## AI page reader (handwritten notes)

Handwritten pages, especially notes that mix prose, chemistry and equations, are read by **GLM-OCR**, a 0.9B vision-language OCR model (MIT licence, stored in `models/vlm/glm-ocr`). It reads the whole page at once and writes formulas as LaTeX and tables as tables. In *Auto* mode it switches on by itself when a page looks handwritten. You can also choose *Always AI page reader* or *Classic only* under **Reader**.

Measured word accuracy on the Chemical Kinetics notes (hand-transcribed pages 1, 2 and 5; `tests/score.py`):

| Page | Classic pipeline | AI page reader |
|---|---|---|
| 1 | 63% | 91% |
| 2 | 66% | 93% |
| 5 | 34% | 87% |

Speed: about 35-45 s per page on an NVIDIA GTX 1650 (4 GB). On a CPU it takes 5-15 minutes per page, so a GPU is strongly recommended. `install.bat` installs the CUDA build of torch from `wheels\gpu` automatically when `nvidia-smi` is present.

## Ruled tables (forms, data sheets)

Tables drawn with lines are rebuilt from the lines themselves (`ocrapp/grid_tables.py`), so any layout works: every row and column, merged cells (row and column spans) and **empty cells**, which stay empty in Excel, Word and HTML. Each cell is read on its own: printed text with PaddleOCR, handwritten entries with the handwriting model. In columns that are mostly numbers, common handwriting slips are corrected (7 read as F, 501-20 → 501.20). Writing beside or around the table is read as normal text. Ruled notebook paper and scanned page edges are not mistaken for tables. Tables without lines still use the SLANet table model.

On the test data sheet (`tests/make_table_samples.py`, `tests/eval_table.py`: 29 × 14 cells, 167 handwritten values):

| | Clean scan | Slanted, blurred photo |
|---|---|---|
| Grid and merged cells | exact | exact |
| Empty cells kept empty | 118/118 | 116/118 |
| Handwritten values read exactly | 92% | 82% |

## What it reads

| Content | How |
|---|---|
| Printed text (prose) | PaddleOCR PP-OCRv4 detection + recognition |
| Handwriting | TrOCR re-reads every line in *Handwritten* mode, and only low-confidence lines in *Auto* mode, keeping whichever reading is more confident |
| Tables | PP-Structure SLANet (row/column spans kept) → Excel cells, Word tables, HTML |
| Formulae | LaTeX-OCR (ONNX), read at several scales, checked against the plain OCR → LaTeX, Word equations |
| Layout | PicoDet CDLA layout model: title, text, table, figure, equation, caption, header, footer; reading order handles two columns |
| Any orientation | Pages are auto-rotated 0/90/180/270°, slanted pages are straightened (any angle up to ±45°), and each text line is cut out with a perspective warp, so slanted, vertical and upside-down lines are read too |

Inputs: PDF (any number of pages), PNG, JPG, TIFF (multi-page), BMP, WEBP, GIF.

## Export formats

Searchable PDF (the page image with an invisible text layer), re-typeset PDF, Word (.docx, with native equations when Microsoft Office is installed), Excel (.xlsx, one sheet per table plus a sheet of all lines with their confidence), CSV, TXT, Markdown, HTML, JSON (full detail: boxes, confidences, alternative readings) and LaTeX.

## Install (offline)

1. Copy this folder to the target PC, for example `C:\OCR`.
2. Double-click **`install.bat`**. It:
   * uses Python 3.12 if it is already installed, or installs it for the current user from `installers\python-3.12.10-amd64.exe`;
   * creates `.venv` in the folder and installs every package from `wheels\` with `pip --no-index`;
   * checks that every package and model loads (`tools\check_install.py`);
   * puts an **Offline OCR** shortcut on the desktop.
3. Start the app with the shortcut or **`run.bat`**.

To check an installation, including a real OCR run with network access blocked:

```bat
.venv\Scripts\python tools\check_install.py --ocr
```

Requirements: Windows 10/11 64-bit, about 4 GB of disk space and 8 GB of RAM. It runs on the CPU, so no GPU is needed.

## Using the app

* **Open files / Open folder**, or drag files onto the window. Press **Run OCR** (F5) for the selected file or **Run all**.
* The page shows coloured boxes per block. Lines in red have low confidence. Click a block, or pick it in the list, to correct it. Text, table cells and LaTeX are all editable, and formulas get a live preview. Your corrections are what gets exported.
* **Content** chooses *Auto*, *Printed*, *Handwritten*, or *Whole image is one formula*. Layout, tables, formulas, auto-rotate, straightening and scan enhancement can be switched on and off, and the *Advanced* section exposes detector thresholds, the handwriting threshold, beam width, threads and max page size.
* **Rotate 90° & re-read** fixes a page whose orientation was guessed wrong.
* **Export** saves the current document in one format. **Export all** writes several formats for every recognised file.

## Fine-tuning (the *Fine-tune* tab)

* **Training data:** correct a document's lines in the OCR tab, then click *Add lines from current document*. The line images and their corrected text are saved to `datasets\handwriting`. You can also import a folder of line images with a `labels.tsv` (`image<TAB>text`) or matching `.txt` files. Text can be edited in the table.
* **Train:** choose a starting model, epochs, batch size, learning rate, validation split, augmentation (slant, rotation, blur, stroke width) and optionally freeze the encoder. Training runs locally, reports validation CER each epoch, and saves the best model to `models\handwriting\<name>` only if it is at least as good as the starting model.
* **Handwriting models:** switch the model in use, evaluate a model's CER/WER on your data, or delete models.
* **Corrections dictionary:** whole-word replacements (for example a name the OCR always misspells), applied to every result without training.

## Command line / batch

```bat
.venv\Scripts\python cli.py scans\ -o output -f pdf docx xlsx txt
.venv\Scripts\python cli.py notes.pdf --mode handwritten -f docx md
```

## Folder layout

```
app.py              GUI entry point          cli.py         batch / command line
ocrapp/             engine, exporters, fine-tuning, GUI
models/             paddle/{det,det_v4,rec,cls,table,layout}, formula/, handwriting/
wheels/             every Python package as a .whl (offline install)
installers/         Python 3.12 installer
datasets/           your fine-tuning data        user/   settings, corrections
tools/              fetch_assets.py (online, refreshes models/wheels), check_install.py
tests/              make_samples.py + sample pages
```

`tools\fetch_assets.py` is the only part that uses the internet. It is for refreshing `models/` and `wheels/` on a connected PC; the app itself never downloads anything, and PaddleOCR's own downloader is disabled.

## Copying to another PC

Copy the folder **without `.venv`** (about 1.1 GB: wheels 530 MB, models 450 MB, Python installer 26 MB). `.venv` is tied to the PC it was built on, and `install.bat` recreates it from `wheels\` in a few minutes.

## Speed and limits (CPU, measured on this PC)

* Printed page: about 10-25 s. The first page after starting takes longer, because the models load.
* Handwritten notebook page: about 1-2 min. Every line goes through the handwriting model; lowering *Handwriting beam width* to 1 roughly halves that time.
* Printed formulas are read reliably. **Handwritten formulas are the weakest part:** the formula model was trained on typeset maths, so handwritten equations come out as approximate text. Correct them in the editor; the corrections are kept in every export.
* The base handwriting model (TrOCR-small, trained on English handwriting) gets common words right but stumbles on unusual words and symbols. Fine-tuning on 100-300 corrected lines of your own handwriting is the most effective way to improve it.

## Notes

* Windows Smart App Control on some PCs blocks a scikit-image DLL that PaddleOCR imports for training only. `ocrapp/compat.py` replaces those training-only modules at inference so the app still starts.
* PaddlePaddle on Windows needs `protobuf<=3.20.2`. That is why `onnxruntime` is pinned to 1.20.1. Do not upgrade either inside `.venv`.
