"""Per-block review and correction: text, tables (cell grid) and formulas (LaTeX)."""
from __future__ import annotations

import io

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit,
                               QPushButton, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ..model import BLOCK_KINDS, table_grid
from .page_view import KIND_COLORS


def render_latex(latex: str) -> QPixmap | None:
    """Preview of a formula using matplotlib's mathtext (subset of LaTeX). None if it can't."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${latex}$", fontsize=18)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=110, bbox_inches="tight", pad_inches=0.08, facecolor="white", format="png")
        plt.close(fig)
        pm = QPixmap()
        pm.loadFromData(buf.getvalue())
        return pm
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


class BlockEditor(QWidget):
    block_selected = Signal(int)
    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.page = None
        self._loading = False
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_row)
        self.kind = QComboBox()
        self.kind.addItems(BLOCK_KINDS)
        self.kind.currentTextChanged.connect(self._on_kind)
        self.info = QLabel()
        self.info.setStyleSheet("color:#666")
        self.text = QPlainTextEdit()
        self.text.textChanged.connect(self._on_text)
        self.table = QTableWidget()
        self.table.itemChanged.connect(self._on_table)
        self.formula_preview = QLabel()
        self.formula_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formula_preview.setMinimumHeight(60)
        self.formula_preview.setStyleSheet("background:white;border:1px solid #ccc")
        self.stack = QStackedWidget()
        textw = QWidget()
        tl = QVBoxLayout(textw)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.text)
        tl.addWidget(self.formula_preview)
        tablew = QWidget()
        tbl = QVBoxLayout(tablew)
        tbl.setContentsMargins(0, 0, 0, 0)
        tbl.addWidget(self.table)
        row = QHBoxLayout()
        for label, fn in (("+ Row", self._add_row), ("+ Column", self._add_col),
                          ("- Row", self._del_row), ("- Column", self._del_col)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        tbl.addLayout(row)
        self.stack.addWidget(textw)
        self.stack.addWidget(tablew)

        top = QWidget()
        topl = QVBoxLayout(top)
        topl.setContentsMargins(0, 0, 0, 0)
        topl.addWidget(QLabel("Blocks (reading order) - click one to review it"))
        topl.addWidget(self.list)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        h = QHBoxLayout()
        h.addWidget(QLabel("Type:"))
        h.addWidget(self.kind)
        self.btn_delete = QPushButton("Delete block")
        self.btn_delete.clicked.connect(self._delete)
        h.addWidget(self.btn_delete)
        h.addStretch(1)
        bl.addLayout(h)
        bl.addWidget(self.info)
        bl.addWidget(self.stack, 1)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(bottom)
        split.setSizes([250, 400])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

    # --------------------------------------------------------------- data
    def set_page(self, page) -> None:
        self.page = page
        self._loading = True
        self.list.clear()
        for i, b in enumerate(page.blocks if page else []):
            self.list.addItem(self._item_for(i, b))
        self._loading = False
        if page and page.blocks:
            self.list.setCurrentRow(0)
        else:
            self.text.setPlainText("")

    def _item_for(self, i, b) -> QListWidgetItem:
        first = " ".join(b.text.split())[:70]
        conf = f"{b.conf:.0%}" if b.lines else ""
        it = QListWidgetItem(f"{i + 1}. [{b.kind}] {conf}  {first}")
        it.setForeground(QColor(KIND_COLORS.get(b.kind, "#000")))
        if b.lines and b.conf < 0.7:
            it.setBackground(QColor("#fdecea"))
        return it

    def _refresh_item(self) -> None:
        r = self.list.currentRow()
        if self.page and 0 <= r < len(self.page.blocks):
            new = self._item_for(r, self.page.blocks[r])
            it = self.list.item(r)
            it.setText(new.text())
            it.setForeground(new.foreground())

    def select(self, idx: int) -> None:
        self.list.setCurrentRow(idx)

    def current(self):
        r = self.list.currentRow()
        if self.page and 0 <= r < len(self.page.blocks):
            return self.page.blocks[r]
        return None

    # ------------------------------------------------------------ events
    def _on_row(self, r: int) -> None:
        b = self.current()
        if b is None:
            return
        self._loading = True
        self.kind.setCurrentText(b.kind)
        hand = sum(1 for l in b.lines if l.source == "hand")
        low = sum(1 for l in b.lines if l.conf < 0.7)
        self.info.setText(f"{len(b.lines)} lines, {hand} read as handwriting, {low} low-confidence"
                          + ("  |  edit the LaTeX below" if b.kind == "formula" else ""))
        if b.kind == "table":
            grid = table_grid(b)
            cols = max((len(r) for r in grid), default=1)
            self.table.clear()
            self.table.setRowCount(len(grid))
            self.table.setColumnCount(cols)
            for i, row in enumerate(grid):
                for j, c in enumerate(row):
                    self.table.setItem(i, j, QTableWidgetItem(c))
            self.table.resizeColumnsToContents()
            self.stack.setCurrentIndex(1)
        else:
            self.text.setPlainText(b.text)
            self.stack.setCurrentIndex(0)
            self._update_formula_preview(b)
        self._loading = False
        self.block_selected.emit(r)

    def _update_formula_preview(self, b) -> None:
        if b.kind == "formula" and b.text.strip():
            pm = render_latex(b.text.strip())
            if pm is not None:
                self.formula_preview.setPixmap(pm)
                self.formula_preview.setText("")
            else:
                self.formula_preview.setPixmap(QPixmap())
                self.formula_preview.setText("(preview not available for this LaTeX - export keeps it as is)")
            self.formula_preview.show()
        else:
            self.formula_preview.hide()

    def _on_text(self) -> None:
        if self._loading:
            return
        b = self.current()
        if b is None:
            return
        t = self.text.toPlainText()
        if b.kind == "formula":
            b.latex = t
            b.edited_text = None
            self._update_formula_preview(b)
        else:
            rows = t.split("\n")
            if b.lines and len(rows) == len(b.lines):
                # keep line-level corrections (they become fine-tuning data)
                for l, r in zip(b.lines, rows):
                    l.text = r
                b.edited_text = None
            else:
                b.edited_text = t
        self._refresh_item()
        self.changed.emit()

    def _on_table(self, _item) -> None:
        if self._loading:
            return
        b = self.current()
        if b is None or b.kind != "table":
            return
        rows = []
        for i in range(self.table.rowCount()):
            rows.append("\t".join((self.table.item(i, j).text() if self.table.item(i, j) else "")
                                  for j in range(self.table.columnCount())))
        b.edited_text = "\n".join(rows)
        self._refresh_item()
        self.changed.emit()

    def _on_kind(self, kind: str) -> None:
        if self._loading:
            return
        b = self.current()
        if b is None:
            return
        if kind == "formula" and b.kind != "formula" and not b.latex:
            b.latex = b.text
        if kind == "table" and not b.cells and b.edited_text is None:
            b.edited_text = b.text.replace("  ", "\t")
        b.kind = kind
        self._refresh_item()
        self._on_row(self.list.currentRow())
        self.changed.emit()

    def _delete(self) -> None:
        r = self.list.currentRow()
        if self.page and 0 <= r < len(self.page.blocks):
            del self.page.blocks[r]
            self.set_page(self.page)
            self.changed.emit()

    def _add_row(self) -> None:
        self.table.insertRow(self.table.rowCount())
        self._on_table(None)

    def _add_col(self) -> None:
        self.table.insertColumn(self.table.columnCount())
        self._on_table(None)

    def _del_row(self) -> None:
        if self.table.rowCount():
            self.table.removeRow(self.table.currentRow() if self.table.currentRow() >= 0 else self.table.rowCount() - 1)
            self._on_table(None)

    def _del_col(self) -> None:
        if self.table.columnCount() > 1:
            c = self.table.currentColumn()
            self.table.removeColumn(c if c >= 0 else self.table.columnCount() - 1)
            self._on_table(None)
