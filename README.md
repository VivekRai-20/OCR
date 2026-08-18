# Offline OCR & Document Information-Extraction System

A modular, **100% offline** OCR and field-extraction system built with **Python + PaddleOCR**.

Once models are downloaded (one-time), the system operates with **no internet access** — no API calls, no model downloads, no telemetry.

---

## Features

| Feature | Details |
|---|---|
| **Offline operation** | Hard requirement — all processing is local |
| **Document types** | JPG, PNG, TIFF, PDF, DOCX, Fax (image/PDF) |
| **OCR engine** | PaddleOCR (printed + handwritten text) |
| **Preprocessing** | Grayscale, denoise, CLAHE, deskew, auto-rotate, threshold |
| **Extraction** | Configurable regex patterns (no AI model needed) |
| **Independent testing** | OCR and patterns can be tested separately |
| **Extensible** | Abstract engine interface — add TrOCR later without pipeline changes |

---

## Project Structure

```
ocr/
├── main.py                      ← CLI entry point
├── requirements.txt
├── setup_models.py              ← One-time model download helper
├── README.md
│
├── config/
│   └── config.yaml              ← All settings (offline_mode, preprocessing, paths)
│
├── models/
│   └── paddleocr/
│       ├── det/                 ← Text detection model
│       ├── rec/                 ← Text recognition model
│       └── cls/                 ← Angle/orientation classifier
│
├── input/                       ← Drop your documents here
├── output/                      ← All results written here
├── temp/                        ← Intermediate page renders (auto-cleaned)
│
├── document/
│   ├── file_detector.py         ← Extension + magic-byte file type detection
│   ├── pdf_processor.py         ← PDF → page images via pdf2image/Poppler
│   ├── docx_processor.py        ← DOCX native text + embedded image extraction
│   └── image_processor.py       ← Image loading (incl. multi-frame TIFF)
│
├── preprocessing/
│   └── image_preprocessor.py    ← Per-step configurable image preprocessing
│
├── ocr/
│   ├── base_engine.py           ← Abstract OCR engine (future-proof interface)
│   ├── paddle_engine.py         ← PaddleOCR implementation
│   └── ocr_pipeline.py          ← Orchestrates all document types → OCR
│
├── extraction/
│   ├── pattern_manager.py       ← YAML pattern loader + compiler
│   ├── regex_engine.py          ← Regex matching + table output
│   └── patterns.yaml            ← Edit this to add/change extraction fields
│
├── output/
│   └── output_writer.py         ← Writes ocr.txt, ocr.json, extracted.json, result.json
│
├── testing/
│   ├── test_patterns.py         ← Pattern tests (no OCR models required)
│   └── test_ocr.py              ← OCR pipeline tests (models required)
│
├── training/
│   └── README.md                ← Future fine-tuning guide
│
└── utils/
    ├── logger.py                 ← Rotating file + console logger
    └── offline_checker.py        ← Validates models exist; blocks auto-downloads
```

---

## Setup

### 1. Prerequisites

**Python 3.9–3.11** is recommended.

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **GPU users**: Replace `paddlepaddle` with the GPU build:
> ```bash
> pip install paddlepaddle-gpu==2.6.1.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
> pip install paddleocr
> ```
> Then set `use_gpu: true` in `config/config.yaml`.

### 3. Install Poppler (PDF support)

Poppler is required by `pdf2image` to render PDF pages.

**Windows:**
1. Download from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract the archive (e.g. to `C:\poppler\`)
3. Add `C:\poppler\Library\bin` to your system PATH

   **OR** set in `config/config.yaml`:
   ```yaml
   pdf:
     poppler_path: "C:/poppler/Library/bin"
   ```

**Linux:** `sudo apt install poppler-utils`

**macOS:** `brew install poppler`

### 4. Download PaddleOCR models (one-time, requires internet)

```bash
python setup_models.py
```

This downloads the detection, recognition, and classifier models to `models/paddleocr/`.
After this step, the system runs fully offline.

### 5. Verify offline readiness

```bash
python main.py --mode check-offline
```

Expected output:
```
[1] Offline mode flag        ✓ ENABLED
[2] PaddleOCR model dirs     ✓ ALL PRESENT
[3] Patterns YAML file       ✓ FOUND

RESULT: System is READY for fully offline operation.
```

---

## Usage

### OCR Only

Test OCR without any regex extraction:

```bash
python main.py --mode ocr --input input/sample.jpg
python main.py --mode ocr --input input/document.pdf
python main.py --mode ocr --input input/scan.tiff
```

**Output:**
```
output/sample/
├── ocr.txt      ← plain extracted text
└── ocr.json     ← per-region: text, confidence, bbox, page
```

### Pattern Extraction Only

Test regex patterns without running OCR — pass existing text:

```bash
python main.py --mode patterns --input input/sample.txt
```

Or paste text interactively (no file needed):

```bash
python main.py --mode patterns
```

Then paste your OCR text and press Ctrl-Z (Windows) / Ctrl-D (Linux):

```
FIELD            VALUE                              PATTERN
----------------------------------------------------------------
subject          Data Mining and Data Warehousing   Pattern 1
title            Association Rule Mining            Pattern 1
name             Vivek Rai                          Pattern 1
roll_no          42                                 Pattern 1
date             18/08/2026                         Pattern 1
```

### Full Pipeline

```bash
python main.py --mode full --input input/document.pdf
```

**Output:**
```
output/document/
├── ocr.txt          ← raw OCR text
├── ocr.json         ← structured OCR regions
├── extracted.json   ← matched fields with pattern info
└── result.json      ← combined summary
```

### Offline Check

```bash
python main.py --mode check-offline
```

---

## Configuration

All settings live in `config/config.yaml`.

### Key settings

```yaml
offline_mode: true        # NEVER set to false

paddleocr:
  lang: "en"              # Language code
  use_gpu: false          # true only with CUDA-capable GPU
  drop_score: 0.5         # Minimum confidence threshold (0–1)

preprocessing:
  enabled: true
  grayscale: true
  denoise: true
  deskew: true
  threshold: true
  threshold_method: "adaptive"   # or "otsu"
  auto_rotate: true

pdf:
  dpi: 300                # Higher = better quality, slower
  poppler_path: null      # Set if Poppler is not on PATH
```

---

## Adding Extraction Patterns

Edit `extraction/patterns.yaml`:

```yaml
my_new_field:
  patterns:
    - 'My Label\s*[:\-]\s*(.+)'
    - 'Alternative Label\s*[:\-]\s*(.+)'
  flags: [IGNORECASE]
  group: 1
```

Test immediately without restarting:

```bash
python main.py --mode patterns --input input/sample.txt
```

---

## Running Tests

### Pattern tests (no models required)

```bash
python -m pytest testing/test_patterns.py -v
```

### OCR tests (models required)

```bash
python -m pytest testing/test_ocr.py -v
```

OCR tests are automatically skipped with a clear message if models are missing.

### All tests

```bash
python -m pytest testing/ -v
```

---

## Independent Testing Guide

### Test 1: OCR accuracy

```
Image → Preprocessing → PaddleOCR → OCR result (text, bbox, confidence)
```

```bash
python main.py --mode ocr --input input/sample.jpg
```

Check `output/sample/ocr.txt` — if text looks wrong, adjust preprocessing in `config.yaml`.

---

### Test 2: Pattern accuracy

```
Text → Regex → Extracted fields
```

```bash
python main.py --mode patterns --input input/sample.txt
```

If a field is `[not found]`, edit `extraction/patterns.yaml` and add/adjust patterns.

---

### Test 3: End-to-end

```
Document → OCR → Regex → Structured result
```

```bash
python main.py --mode full --input input/document.pdf
```

---

## Output File Reference

### `ocr.json` — per-region OCR data

```json
{
  "file": "/path/to/document.pdf",
  "doc_type": "PDF",
  "pages": 3,
  "elapsed_s": 4.21,
  "regions": [
    {
      "text": "Subject: Data Mining",
      "confidence": 0.9623,
      "bbox": [100, 200, 500, 240],
      "page": 1
    }
  ]
}
```

### `extracted.json` — field extraction detail

```json
{
  "subject": {
    "field": "subject",
    "value": "Data Mining and Data Warehousing",
    "matched_by": "Pattern 1",
    "pattern_str": "Subject\\s*[:\\-]\\s*(.+)",
    "confidence": null
  }
}
```

### `result.json` — combined summary

```json
{
  "file": "/path/to/document.pdf",
  "pages": 3,
  "elapsed_s": 4.21,
  "extracted": {
    "subject": "Data Mining and Data Warehousing",
    "name": "Vivek Rai",
    "roll_no": "42"
  }
}
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ERROR: PaddleOCR model not found locally` | Run `python setup_models.py` |
| `Poppler not installed` | Install Poppler (see Setup §3) |
| `paddleocr not installed` | Run `pip install -r requirements.txt` |
| Poor OCR on dark/noisy scans | Enable `denoise: true`, `threshold: true` in config.yaml |
| Fields not extracted | Edit `extraction/patterns.yaml` and test with `--mode patterns` |
| PDF rendering fails | Check Poppler is on PATH or set `pdf.poppler_path` in config.yaml |

---

## Extending the System

### Add a new OCR engine (e.g. TrOCR)

1. Create `ocr/trocr_engine.py` implementing `BaseOCREngine`
2. Implement `initialize(config)` and `recognize(image, page)` methods
3. In `main.py`, swap `PaddleEngine()` for `TrOCREngine()` — zero changes elsewhere

### Add new extraction fields

1. Edit `extraction/patterns.yaml`
2. Test with: `python main.py --mode patterns --input input/sample.txt`

### Fine-tune for handwriting

See `training/README.md` for detailed instructions.

---

## License

MIT License. See LICENSE file.
