"""Export a Document to TXT, Markdown, HTML, JSON, CSV, Word, Excel, PDF and LaTeX."""
from __future__ import annotations

import base64
import csv
import html as _html
import io
import json
import re
from pathlib import Path

import cv2
import numpy as np

from . import imaging
from .model import Block, Document, table_grid
from .tables import cells_to_html, grid_to_html


# ------------------------------------------------------------------ helpers
def _png_bytes(img: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".png", img)
    return enc.tobytes()


def _block_table_html(b: Block) -> str:
    if b.edited_text is not None:
        return grid_to_html(table_grid(b))
    if b.cells:
        return cells_to_html(b.cells)
    return b.html


def _blocks(doc: Document):
    for p in doc.pages:
        for b in p.blocks:
            yield p, b


# --------------------------------------------------------------------- text
def to_txt(doc: Document, dest: Path) -> None:
    out = []
    for p in doc.pages:
        parts = []
        for b in p.blocks:
            if b.kind == "formula":
                parts.append(f"$$ {b.text} $$")
            elif b.text.strip():
                parts.append(b.text)
        out.append("\n\n".join(parts))
    dest.write_text("\n\n\f".join(out), encoding="utf-8")


def to_markdown(doc: Document, dest: Path) -> None:
    out = []
    for p in doc.pages:
        if len(doc.pages) > 1:
            out.append(f"<!-- page {p.index + 1} -->")
        for b in p.blocks:
            t = b.text.strip()
            if b.kind == "title" and t:
                out.append("## " + " ".join(t.split()))
            elif b.kind == "formula":
                out.append(f"$$\n{t}\n$$")
            elif b.kind == "table":
                g = table_grid(b)
                if g:
                    w = max(len(r) for r in g)
                    g = [r + [""] * (w - len(r)) for r in g]
                    esc = lambda c: c.replace("|", "\\|").replace("\n", " ")
                    out.append("| " + " | ".join(esc(c) for c in g[0]) + " |")
                    out[-1] += "\n|" + "---|" * w
                    for r in g[1:]:
                        out[-1] += "\n| " + " | ".join(esc(c) for c in r) + " |"
            elif b.kind == "list" and t:
                out.append("\n".join(("- " + l.lstrip("-•* ")) for l in t.splitlines()))
            elif t:
                out.append(t)
    dest.write_text("\n\n".join(out) + "\n", encoding="utf-8")


def to_html_string(doc: Document, images: bool = True) -> str:
    body = []
    for p in doc.pages:
        body.append(f"<section class='page'><div class='pno'>Page {p.index + 1}</div>")
        for b in p.blocks:
            t = _html.escape(b.text)
            if b.kind == "title":
                body.append(f"<h2>{t}</h2>")
            elif b.kind == "table":
                body.append(_block_table_html(b))
            elif b.kind == "formula":
                body.append(f"<div class='formula'>\\[{t}\\]</div>")
            elif b.kind == "figure":
                if images:
                    data = base64.b64encode(_png_bytes(imaging.crop_box(p.image, b.bbox))).decode()
                    body.append(f"<figure><img src='data:image/png;base64,{data}'/>"
                                + (f"<figcaption>{t}</figcaption>" if t else "") + "</figure>")
                elif t:
                    body.append(f"<p>{t}</p>")
            elif b.kind in ("header", "footer"):
                body.append(f"<p class='{b.kind}'>{t}</p>")
            elif b.kind == "caption":
                body.append(f"<p class='caption'>{t}</p>")
            elif t:
                body.append("<p>" + t.replace("\n", "<br/>") + "</p>")
        body.append("</section>")
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:auto;padding:16px;line-height:1.45}"
           "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #888;padding:3px 6px}"
           ".formula{font-family:Cambria Math,serif;background:#f4f4f4;padding:6px;margin:6px 0}"
           ".page{border-bottom:1px dashed #aaa;padding-bottom:12px}.pno{color:#888;font-size:12px}"
           ".header,.footer,.caption{color:#555;font-size:13px}img{max-width:100%}")
    title = _html.escape(Path(doc.source).name)
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
            f"<style>{css}</style></head><body>{''.join(body)}</body></html>")


def to_html(doc: Document, dest: Path) -> None:
    dest.write_text(to_html_string(doc), encoding="utf-8")


def to_json(doc: Document, dest: Path) -> None:
    data = {"source": doc.source, "pages": []}
    for p in doc.pages:
        w, h = p.size
        data["pages"].append({
            "index": p.index, "width": w, "height": h, "rotation": p.rotation, "skew": round(p.skew, 2),
            "seconds": round(p.seconds, 2),
            "blocks": [{
                "type": b.kind, "bbox": [int(v) for v in b.bbox], "text": b.text,
                "latex": b.latex or None, "table": table_grid(b) if b.kind == "table" else None,
                "lines": [{"text": l.text, "confidence": round(l.conf, 4), "source": l.source,
                           "polygon": [[round(x, 1), round(y, 1)] for x, y in l.poly],
                           **({"alternative": l.alt_text, "alt_confidence": round(l.alt_conf, 4)} if l.alt_text else {})}
                          for l in b.lines],
            } for b in p.blocks],
        })
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def to_csv(doc: Document, dest: Path) -> None:
    """Every table, one after another (blank row between); without tables, one line per row."""
    with open(dest, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        tables = [b for _, b in _blocks(doc) if b.kind == "table"]
        if tables:
            for i, b in enumerate(tables):
                if i:
                    w.writerow([])
                w.writerows(table_grid(b))
        else:
            w.writerow(["page", "block", "type", "text", "confidence"])
            for p, b in _blocks(doc):
                for l in b.lines:
                    w.writerow([p.index + 1, p.blocks.index(b) + 1, b.kind, l.text, round(l.conf, 3)])


def to_latex(doc: Document, dest: Path) -> None:
    def esc(s: str) -> str:
        for a, r in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                     ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
            s = s.replace(a, r)
        return s

    out = [r"\documentclass{article}", r"\usepackage[utf8]{inputenc}", r"\usepackage{amsmath,amssymb}",
           r"\begin{document}"]
    for p, b in _blocks(doc):
        if b.kind == "title":
            out.append(r"\section*{" + esc(b.text) + "}")
        elif b.kind == "formula":
            out.append(r"\[" + b.text + r"\]")
        elif b.kind == "table":
            g = table_grid(b)
            if g:
                n = max(len(r) for r in g)
                out.append(r"\begin{tabular}{" + "|l" * n + "|}\\hline")
                for r in g:
                    out.append(" & ".join(esc(c) for c in r + [""] * (n - len(r))) + r" \\ \hline")
                out.append(r"\end{tabular}")
        elif b.text.strip():
            out.append(esc(b.text) + "\n")
    out.append(r"\end{document}")
    dest.write_text("\n".join(out), encoding="utf-8")


# --------------------------------------------------------------------- Word
_OMML_XSL = None


def _omml(latex: str):
    """LaTeX -> Word native equation (OMML) using Office's MML2OMML.XSL when it is installed."""
    global _OMML_XSL
    try:
        from lxml import etree
        import latex2mathml.converter
    except Exception:
        return None
    if _OMML_XSL is None:
        _OMML_XSL = False
        for base in (r"C:\Program Files\Microsoft Office\root\Office16", r"C:\Program Files (x86)\Microsoft Office\root\Office16",
                     r"C:\Program Files\Microsoft Office\Office16", r"C:\Program Files (x86)\Microsoft Office\Office16"):
            f = Path(base) / "MML2OMML.XSL"
            if f.exists():
                _OMML_XSL = etree.XSLT(etree.parse(str(f)))
                break
    if not _OMML_XSL:
        return None
    try:
        mml = etree.fromstring(latex2mathml.converter.convert(latex).encode())
        return _OMML_XSL(mml).getroot()
    except Exception:
        return None


def _docx_rich_text(para, text: str) -> None:
    """Text with inline $...$ maths (from the AI page reader) -> runs + Word equations."""
    for i, part in enumerate(re.split(r"(\$[^$]+\$)", text)):
        if not part:
            continue
        if part.startswith("$") and part.endswith("$") and len(part) > 2:
            om = _omml(part[1:-1])
            if om is not None:
                para._p.append(om)
                continue
            r = para.add_run(part[1:-1])
            r.font.name = "Cambria Math"
        else:
            para.add_run(part)


def _docx_table(document, grid):
    if not grid:
        return
    n = max(len(r) for r in grid)
    t = document.add_table(rows=len(grid), cols=n)
    t.style = "Table Grid"
    for i, r in enumerate(grid):
        for j, c in enumerate(r):
            t.cell(i, j).text = c


def to_docx(doc: Document, dest: Path) -> None:
    from docx import Document as Docx
    from docx.shared import Inches, Pt

    d = Docx()
    st = d.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    for pi, p in enumerate(doc.pages):
        if pi:
            d.add_page_break()
        for b in p.blocks:
            t = b.text.strip()
            if b.kind == "title" and t:
                d.add_heading(" ".join(t.split()), level=2)
            elif b.kind == "table":
                if b.edited_text is None and b.cells:
                    _docx_table_spans(d, b.cells)
                else:
                    _docx_table(d, table_grid(b))
                d.add_paragraph()
            elif b.kind == "formula":
                para = d.add_paragraph()
                om = _omml(t) if t else None
                if om is not None:
                    para._p.append(om)
                else:
                    r = para.add_run(t)
                    r.font.name = "Cambria Math"
            elif b.kind == "figure":
                crop = imaging.crop_box(p.image, b.bbox)
                if crop.size:
                    width = min(6.0, crop.shape[1] / 150)
                    d.add_picture(io.BytesIO(_png_bytes(crop)), width=Inches(max(width, 1.0)))
                if t:
                    d.add_paragraph().add_run(t).italic = True
            elif b.kind == "list" and t:
                for l in t.splitlines():
                    d.add_paragraph(l.lstrip("-•* "), style="List Bullet")
            elif t:
                para = d.add_paragraph()
                _docx_rich_text(para, t.replace("\n", " ") if b.kind == "text" else t)
                if b.kind in ("header", "footer", "caption"):
                    for r in para.runs:
                        r.font.size = Pt(9)
    d.save(str(dest))


def _docx_table_spans(document, cells):
    from .model import Block as _B

    grid = table_grid(_B("table", [0, 0, 0, 0], cells=cells))
    if not grid:
        return
    n = max(len(r) for r in grid)
    t = document.add_table(rows=len(grid), cols=n)
    t.style = "Table Grid"
    occupied = set()
    r = 0
    for row in cells:
        c = 0
        for cell in row:
            while (r, c) in occupied:
                c += 1
            rs, cs = cell.get("rowspan", 1), cell.get("colspan", 1)
            for dr in range(rs):
                for dc in range(cs):
                    occupied.add((r + dr, c + dc))
            try:
                a = t.cell(r, c)
                if rs > 1 or cs > 1:
                    a = a.merge(t.cell(min(r + rs - 1, len(grid) - 1), min(c + cs - 1, n - 1)))
                a.text = cell["text"]
            except Exception:
                pass
            c += cs
        r += 1


# -------------------------------------------------------------------- Excel
def to_xlsx(doc: Document, dest: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Text"
    ws.append(["Page", "Block", "Type", "Text", "Confidence", "Source"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for p in doc.pages:
        for bi, b in enumerate(p.blocks, 1):
            if b.kind == "formula":
                ws.append([p.index + 1, bi, "formula", b.text, None, "latex"])
            elif b.edited_text is not None or not b.lines:
                ws.append([p.index + 1, bi, b.kind, b.text, None, "edited" if b.edited_text is not None else ""])
            else:
                for l in b.lines:
                    ws.append([p.index + 1, bi, b.kind, l.text, round(l.conf, 3), l.source])
    ws.column_dimensions["D"].width = 90
    thin = Side(style="thin", color="999999")
    n = 0
    for p, b in _blocks(doc):
        if b.kind != "table":
            continue
        n += 1
        t = wb.create_sheet(f"Table {n} (p{p.index + 1})"[:31])
        if b.edited_text is None and b.cells:
            occupied = set()
            r = 1
            for row in b.cells:
                c = 1
                for cell in row:
                    while (r, c) in occupied:
                        c += 1
                    rs, cs = cell.get("rowspan", 1), cell.get("colspan", 1)
                    for dr in range(rs):
                        for dc in range(cs):
                            occupied.add((r + dr, c + dc))
                    t.cell(r, c, _num(cell["text"]))
                    if rs > 1 or cs > 1:
                        t.merge_cells(start_row=r, start_column=c, end_row=r + rs - 1, end_column=c + cs - 1)
                    c += cs
                r += 1
        else:
            for row in table_grid(b):
                t.append([_num(c) for c in row])
        for row in t.iter_rows():
            for c in row:
                c.border = Border(left=thin, right=thin, top=thin, bottom=thin)
                c.alignment = Alignment(wrap_text=True, vertical="top")
        for col in range(1, t.max_column + 1):
            t.column_dimensions[get_column_letter(col)].width = 18
    wb.save(str(dest))


def _num(s: str):
    """Numbers stay numbers in Excel ("1,234.5" -> 1234.5); everything else is text."""
    v = s.strip().replace(",", "")
    if re.fullmatch(r"-?(0|[1-9]\d*)(\.\d+)?", v):
        return float(v) if "." in v else int(v)
    return s


# ---------------------------------------------------------------------- PDF
def to_pdf_searchable(doc: Document, dest: Path) -> None:
    """Each page image with an invisible, selectable/searchable text layer on top."""
    import fitz

    out = fitz.open()
    for p in doc.pages:
        w, h = p.size
        scale = 72 / 150  # page images are treated as 150 dpi for the PDF page size
        page = out.new_page(width=w * scale, height=h * scale)
        ok, jpg = cv2.imencode(".jpg", p.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        page.insert_image(page.rect, stream=jpg.tobytes())
        for b in p.blocks:
            for l in b.lines:
                if not l.text.strip():
                    continue
                x1, y1, x2, y2 = l.bbox
                fs = max(1.0, (y2 - y1) * scale * 0.8)
                tw = fitz.get_text_length(l.text, fontname="helv", fontsize=fs)
                if tw <= 0:
                    continue
                # stretch the invisible text to the width of the line box
                sx = (x2 - x1) * scale / tw
                origin = fitz.Point(x1 * scale, y2 * scale - (y2 - y1) * scale * 0.2)
                try:
                    page.insert_text(origin, l.text, fontname="helv", fontsize=fs, render_mode=3,
                                     morph=(origin, fitz.Matrix(sx, 1)))
                except Exception:
                    pass
    out.save(str(dest), deflate=True, garbage=3)
    out.close()


def to_pdf_text(doc: Document, dest: Path) -> None:
    """A clean, re-typeset PDF of the recognised content (text, tables, figures)."""
    import fitz

    html = to_html_string(doc)
    story = fitz.Story(html=html)
    writer = fitz.DocumentWriter(str(dest))
    mediabox = fitz.paper_rect("a4")
    where = mediabox + (48, 48, -48, -48)
    more = True
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()


EXPORTERS = {
    # key: (label, extension, function)
    "pdf": ("Searchable PDF (image + text layer)", "pdf", to_pdf_searchable),
    "pdf_text": ("PDF (re-typeset text)", "text.pdf", to_pdf_text),
    "docx": ("Word document", "docx", to_docx),
    "xlsx": ("Excel workbook", "xlsx", to_xlsx),
    "csv": ("CSV (tables)", "csv", to_csv),
    "txt": ("Plain text", "txt", to_txt),
    "md": ("Markdown", "md", to_markdown),
    "html": ("HTML", "html", to_html),
    "json": ("JSON (full detail)", "json", to_json),
    "tex": ("LaTeX", "tex", to_latex),
}
EXPORTERS = {k: (v[0], v[1], v[2]) for k, v in EXPORTERS.items()}


def export(doc: Document, fmt: str, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    EXPORTERS[fmt][2](doc, dest)
    return dest
