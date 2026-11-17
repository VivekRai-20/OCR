"""Loading inputs (images, PDFs) and geometric / photometric preprocessing."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".jp2", ".pbm", ".pgm", ".ppm"}
PDF_EXT = {".pdf"}
SUPPORTED = IMAGE_EXT | PDF_EXT


def imread(path: str | Path) -> np.ndarray:
    """cv2.imread that works with non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image

        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def load_pages(path: str | Path, dpi: int = 200) -> list[np.ndarray]:
    """All pages of a file as BGR images (multi-page TIFF and PDF supported)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in PDF_EXT:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        pages = []
        for i in range(len(pdf)):
            pil = pdf[i].render(scale=dpi / 72).to_pil().convert("RGB")
            pages.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        pdf.close()
        return pages
    if ext in {".tif", ".tiff", ".gif"}:
        from PIL import Image, ImageSequence

        im = Image.open(path)
        frames = [cv2.cvtColor(np.array(f.convert("RGB")), cv2.COLOR_RGB2BGR) for f in ImageSequence.Iterator(im)]
        if frames:
            return frames
    return [imread(path)]


def load_page(path: str | Path, index: int, dpi: int = 200) -> np.ndarray:
    """One page (PDFs are rendered on demand, so long documents do not fill the memory)."""
    path = Path(path)
    if path.suffix.lower() in PDF_EXT:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            pil = pdf[index].render(scale=dpi / 72).to_pil().convert("RGB")
        finally:
            pdf.close()
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return load_pages(path, dpi)[index]


def page_count(path: str | Path) -> int:
    path = Path(path)
    if path.suffix.lower() in PDF_EXT:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        n = len(pdf)
        pdf.close()
        return n
    return 1


def limit_size(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if max_side and s > max_side:
        f = max_side / s
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return img


def enhance(img: np.ndarray) -> np.ndarray:
    """Gentle cleanup for poor scans / phone photos: denoise + local contrast."""
    den = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def rotate90(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate counter-clockwise by 0/90/180/270 degrees."""
    angle %= 360
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def rotate_free(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate counter-clockwise by any angle, enlarging the canvas (white fill)."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def box_angle(poly) -> tuple[float, float]:
    """(angle in degrees of the box's long top edge, its length). Positive = text rises to the right."""
    p = np.asarray(poly, dtype=np.float64)
    dx, dy = p[1][0] - p[0][0], p[1][1] - p[0][1]
    return -math.degrees(math.atan2(dy, dx)), math.hypot(dx, dy)


def crop_poly(img: np.ndarray, poly, pad: int = 2) -> np.ndarray:
    """Perspective-correct crop of a (possibly slanted) quadrilateral text box.
    Tall crops are turned so the text runs left to right (vertical text lines)."""
    pts = np.asarray(poly, dtype=np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    w, h = max(w, 4), max(h, 4)
    dst = np.float32([[pad, pad], [w + pad, pad], [w + pad, h + pad], [pad, h + pad]])
    m = cv2.getPerspectiveTransform(pts, dst)
    out = cv2.warpPerspective(img, m, (w + 2 * pad, h + 2 * pad), borderMode=cv2.BORDER_REPLICATE,
                              flags=cv2.INTER_CUBIC)
    if out.shape[0] >= out.shape[1] * 1.5:
        out = np.rot90(out)
    return np.ascontiguousarray(out)


def crop_box(img: np.ndarray, bbox, pad: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)].copy()
