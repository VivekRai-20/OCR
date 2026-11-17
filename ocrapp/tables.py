"""Table HTML (from SLANet) <-> cell lists with row/col spans."""
from __future__ import annotations

import html as _html
from html.parser import HTMLParser


class _TP(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[dict]] = []
        self._cell: dict | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append([])
            self._cell = {"text": "", "rowspan": int(a.get("rowspan", 1) or 1),
                          "colspan": int(a.get("colspan", 1) or 1), "header": tag == "th"}

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._cell["text"] = " ".join(self._cell["text"].split())
            self.rows[-1].append(self._cell)
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["text"] += data


def parse_table_html(html: str) -> list[list[dict]]:
    p = _TP()
    p.feed(html or "")
    return [r for r in p.rows if r]


def grid_to_html(grid: list[list[str]]) -> str:
    rows = "".join("<tr>" + "".join(f"<td>{_html.escape(c)}</td>" for c in r) + "</tr>" for r in grid)
    return f"<table border='1'>{rows}</table>"


def cells_to_html(cells: list[list[dict]]) -> str:
    out = []
    for r in cells:
        tds = []
        for c in r:
            span = ""
            if c.get("rowspan", 1) > 1:
                span += f" rowspan='{c['rowspan']}'"
            if c.get("colspan", 1) > 1:
                span += f" colspan='{c['colspan']}'"
            tds.append(f"<td{span}>{_html.escape(c['text'])}</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    return "<table border='1'>" + "".join(out) + "</table>"
