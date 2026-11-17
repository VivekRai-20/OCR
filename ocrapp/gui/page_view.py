"""Zoomable page image with recognised blocks and lines drawn on top."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsSimpleTextItem

KIND_COLORS = {
    "title": "#8e44ad", "text": "#2e86de", "list": "#16a085", "table": "#e67e22", "formula": "#c0392b",
    "figure": "#7f8c8d", "caption": "#27ae60", "header": "#95a5a6", "footer": "#95a5a6",
}


def to_pixmap(img: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    q = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(q)


class _BlockItem(QGraphicsRectItem):
    def __init__(self, idx: int, rect: QRectF, color: QColor, view: "PageView") -> None:
        super().__init__(rect)
        self.idx = idx
        self.view = view
        self.color = color
        self.setAcceptHoverEvents(True)
        self.set_selected(False)

    def set_selected(self, on: bool) -> None:
        c = QColor(self.color)
        self.setPen(QPen(c, 4 if on else 2))
        c.setAlpha(60 if on else 18)
        self.setBrush(QBrush(c))

    def mousePressEvent(self, e):  # noqa: N802
        self.view.block_clicked.emit(self.idx)
        super().mousePressEvent(e)


class PageView(QGraphicsView):
    block_clicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#E2E8F0"))  # CDTRS slate, like its document preview
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self._items: list[_BlockItem] = []
        self.show_lines = True
        self.show_blocks = True
        self.low_conf = 0.7

    def clear(self) -> None:
        self.scene().clear()
        self._items = []

    def show_image(self, img: np.ndarray) -> None:
        self.clear()
        self.scene().addPixmap(to_pixmap(img))
        self.scene().setSceneRect(0, 0, img.shape[1], img.shape[0])
        self.fit()

    def show_page(self, page) -> None:
        self.clear()
        self.scene().addPixmap(to_pixmap(page.image))
        h, w = page.image.shape[:2]
        self.scene().setSceneRect(0, 0, w, h)
        for i, b in enumerate(page.blocks):
            color = QColor(KIND_COLORS.get(b.kind, "#2e86de"))
            if self.show_lines:
                for l in b.lines:
                    bad = l.conf < self.low_conf
                    pen = QPen(QColor("#e74c3c") if bad else QColor(0, 160, 0, 170), 2 if bad else 1)
                    poly = QPolygonF([QPointF(x, y) for x, y in l.poly])
                    self.scene().addPolygon(poly, pen)
            if self.show_blocks:
                x1, y1, x2, y2 = b.bbox
                item = _BlockItem(i, QRectF(x1, y1, x2 - x1, y2 - y1), color, self)
                self.scene().addItem(item)
                self._items.append(item)
                label = QGraphicsSimpleTextItem(f"{i + 1} {b.kind}")
                label.setBrush(color)
                f = label.font()
                f.setPointSize(max(9, int(h / 110)))
                f.setBold(True)
                label.setFont(f)
                label.setPos(x1, max(0, y1 - f.pointSize() * 1.8))
                self.scene().addItem(label)

    def select_block(self, idx: int) -> None:
        for it in self._items:
            it.set_selected(it.idx == idx)
        for it in self._items:
            if it.idx == idx:
                self.ensureVisible(it, 40, 40)

    def fit(self) -> None:
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, f: float) -> None:
        self.scale(f, f)

    def wheelEvent(self, e):  # noqa: N802
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(1.15 if e.angleDelta().y() > 0 else 1 / 1.15)
        else:
            super().wheelEvent(e)
