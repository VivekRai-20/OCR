"""Main window: open files, run OCR, review/correct, export, fine-tune."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox, QLabel, QMainWindow, QMenu, QMessageBox,
                               QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget,
                               QTextBrowser, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
                               QPlainTextEdit, QFrame, QHBoxLayout, QStackedWidget, QToolBar)

from .. import handwriting, imaging
from ..engine import OCREngine
from ..exporters import EXPORTERS, export, to_html_string
from ..paths import OUTPUT, ROOT
from ..settings import OCRSettings
from .block_editor import BlockEditor
from .finetune_tab import FinetuneTab
from .page_view import PageView
from .workers import FuncWorker, OCRWorker

FILE_FILTER = ("Documents (*.pdf *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.gif *.jp2);;"
               "PDF (*.pdf);;Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.gif);;All files (*)")
MODES = [("auto", "Auto (printed + handwriting)"), ("printed", "Printed text only"),
         ("handwritten", "Handwritten"), ("formula", "Whole image is one formula")]


class SettingsPanel(QScrollArea):
    def __init__(self, s: OCRSettings, on_change) -> None:
        super().__init__()
        self.s = s
        self.on_change = on_change
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)

        g = QGroupBox("Recognition")
        f = QFormLayout(g)
        self.mode = QComboBox()
        for k, label in MODES:
            self.mode.addItem(label, k)
        self.mode.setCurrentIndex([k for k, _ in MODES].index(s.mode) if s.mode in dict(MODES) else 0)
        self.reader = QComboBox()
        for k, label in (("auto", "Auto (AI reader for handwritten pages)"), ("ai", "Always AI page reader (best, slower)"),
                         ("classic", "Classic only (fastest)")):
            self.reader.addItem(label, k)
        self.vlm_dev = QComboBox()
        self.vlm_dev.addItem("GPU if available", "auto")
        self.vlm_dev.addItem("CPU only", "cpu")
        self.layout_cb = QCheckBox("Layout analysis (titles, columns, figures)")
        self.tables = QCheckBox("Tables")
        self.formulas = QCheckBox("Formulas -> LaTeX")
        self.hw_model = QComboBox()
        self.reload_models()
        f.addRow("Content", self.mode)
        f.addRow("Reader", self.reader)
        f.addRow("AI reader runs on", self.vlm_dev)
        f.addRow(self.layout_cb)
        f.addRow(self.tables)
        f.addRow(self.formulas)
        f.addRow("Handwriting model", self.hw_model)
        lay.addWidget(g)

        g2 = QGroupBox("Image preparation")
        f2 = QFormLayout(g2)
        self.orient = QCheckBox("Auto-rotate pages (0/90/180/270)")
        self.deskew = QCheckBox("Straighten slanted pages")
        self.enhance = QCheckBox("Enhance poor scans / photos")
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 600)
        self.dpi.setSingleStep(50)
        self.dpi.setSuffix(" dpi")
        f2.addRow(self.orient)
        f2.addRow(self.deskew)
        f2.addRow(self.enhance)
        f2.addRow("PDF render", self.dpi)
        lay.addWidget(g2)

        g3 = QGroupBox("Advanced")
        f3 = QFormLayout(g3)
        self.det = QComboBox()
        self.det.addItem("PP-OCRv4 (multi-language det)", "det_v4")
        self.det.addItem("PP-OCRv3 English", "det")
        self.hw_thr = QDoubleSpinBox()
        self.hw_thr.setRange(0, 1)
        self.hw_thr.setSingleStep(0.05)
        self.box_thr = QDoubleSpinBox()
        self.box_thr.setRange(0.1, 0.95)
        self.box_thr.setSingleStep(0.05)
        self.unclip = QDoubleSpinBox()
        self.unclip.setRange(1.0, 3.0)
        self.unclip.setSingleStep(0.1)
        self.drop = QDoubleSpinBox()
        self.drop.setRange(0, 0.95)
        self.drop.setSingleStep(0.05)
        self.beams = QSpinBox()
        self.beams.setRange(1, 10)
        self.threads = QSpinBox()
        self.threads.setRange(1, max(1, os.cpu_count() or 4))
        self.maxside = QSpinBox()
        self.maxside.setRange(800, 8000)
        self.maxside.setSingleStep(200)
        self.corr = QCheckBox("Apply corrections dictionary")
        f3.addRow("Text detector", self.det)
        f3.addRow("Re-read as handwriting below", self.hw_thr)
        f3.addRow("Detection box threshold", self.box_thr)
        f3.addRow("Box expansion", self.unclip)
        f3.addRow("Drop lines below confidence", self.drop)
        f3.addRow("Handwriting beam width", self.beams)
        f3.addRow("CPU threads", self.threads)
        f3.addRow("Max page size (px)", self.maxside)
        f3.addRow(self.corr)
        lay.addWidget(g3)
        lay.addStretch(1)
        self.setWidget(w)
        self.load()
        for wdg in (self.mode, self.hw_model, self.det, self.reader, self.vlm_dev):
            wdg.currentIndexChanged.connect(self.save)
        for wdg in (self.layout_cb, self.tables, self.formulas, self.orient, self.deskew, self.enhance, self.corr):
            wdg.toggled.connect(self.save)
        for wdg in (self.dpi, self.hw_thr, self.box_thr, self.unclip, self.drop, self.beams, self.threads, self.maxside):
            wdg.valueChanged.connect(self.save)

    def reload_models(self) -> None:
        cur = self.s.hw_model
        self.hw_model.blockSignals(True)
        self.hw_model.clear()
        self.hw_model.addItem(f"Active ({handwriting.active_model() or 'none'})", "")
        for m in handwriting.list_models():
            self.hw_model.addItem(m, m)
        i = self.hw_model.findData(cur)
        self.hw_model.setCurrentIndex(max(0, i))
        self.hw_model.blockSignals(False)

    def load(self) -> None:
        s = self.s
        self.reader.setCurrentIndex(max(0, self.reader.findData(s.reader)))
        self.vlm_dev.setCurrentIndex(max(0, self.vlm_dev.findData(s.vlm_device)))
        self.layout_cb.setChecked(s.layout)
        self.tables.setChecked(s.tables)
        self.formulas.setChecked(s.formulas)
        self.orient.setChecked(s.auto_orient)
        self.deskew.setChecked(s.deskew)
        self.enhance.setChecked(s.enhance)
        self.dpi.setValue(s.pdf_dpi)
        self.det.setCurrentIndex(max(0, self.det.findData(s.det_model)))
        self.hw_thr.setValue(s.hw_threshold)
        self.box_thr.setValue(s.det_box_thresh)
        self.unclip.setValue(s.det_unclip)
        self.drop.setValue(s.drop_score)
        self.beams.setValue(s.hw_beams)
        self.threads.setValue(min(s.threads, self.threads.maximum()))
        self.maxside.setValue(s.max_side)
        self.corr.setChecked(s.apply_corrections)

    def save(self, *_):
        s = self.s
        s.mode = self.mode.currentData()
        s.reader = self.reader.currentData()
        s.vlm_device = self.vlm_dev.currentData()
        s.layout, s.tables, s.formulas = self.layout_cb.isChecked(), self.tables.isChecked(), self.formulas.isChecked()
        s.auto_orient, s.deskew, s.enhance = self.orient.isChecked(), self.deskew.isChecked(), self.enhance.isChecked()
        s.pdf_dpi = self.dpi.value()
        s.det_model = self.det.currentData()
        s.hw_model = self.hw_model.currentData() or ""
        s.hw_threshold, s.det_box_thresh = self.hw_thr.value(), self.box_thr.value()
        s.det_unclip, s.drop_score = self.unclip.value(), self.drop.value()
        s.hw_beams, s.threads, s.max_side = self.beams.value(), self.threads.value(), self.maxside.value()
        s.apply_corrections = self.corr.isChecked()
        s.save()
        self.on_change()


class ExportDialog(QDialog):
    def __init__(self, parent=None, folder: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Export all documents")
        lay = QVBoxLayout(self)
        self.checks = {}
        for k, (label, ext, _) in EXPORTERS.items():
            cb = QCheckBox(f"{label} (.{ext})")
            cb.setChecked(k in ("pdf", "docx", "xlsx", "txt"))
            self.checks[k] = cb
            lay.addWidget(cb)
        self.folder = QPushButton(folder or str(OUTPUT))
        self.folder.clicked.connect(self._pick)
        lay.addWidget(QLabel("Output folder:"))
        lay.addWidget(self.folder)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _pick(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.folder.text())
        if d:
            self.folder.setText(d)

    def formats(self) -> list[str]:
        return [k for k, cb in self.checks.items() if cb.isChecked()]


def _card(title: str, body: QWidget, hint: str = "") -> QFrame:
    """White rounded CDTRS 'contentCard' with an optional section title."""
    card = QFrame()
    card.setObjectName("contentCard")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(12, 10, 12, 12)
    lay.setSpacing(6)
    if title:
        t = QLabel(title)
        t.setObjectName("sectionTitle")
        lay.addWidget(t)
    if hint:
        h = QLabel(hint)
        h.setObjectName("hintLabel")
        lay.addWidget(h)
    lay.addWidget(body, 1)
    return card


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Offline OCR")
        self.resize(1500, 920)
        self.qs = QSettings(str(ROOT / "user" / "window.ini"), QSettings.Format.IniFormat)
        self.settings = OCRSettings.load()
        self.engine = OCREngine(self.settings)
        self.docs: dict[str, object] = {}     # path -> Document or None
        self.cur_path: str | None = None
        self.cur_page = 0
        self.worker: OCRWorker | None = None

        # ---- left: files + settings
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._on_tree)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.panel = SettingsPanel(self.settings, self._settings_changed)
        left = QSplitter(Qt.Orientation.Vertical)
        fw = _card("Documents", self.tree, "Drag and drop PDFs or images here")
        left.addWidget(fw)
        left.addWidget(_card("Recognition settings", self.panel))
        left.setSizes([300, 600])

        # ---- centre: page view
        self.view = PageView()
        self.view.block_clicked.connect(lambda i: self.editor.select(i))

        # ---- right: blocks / text / preview
        self.editor = BlockEditor()
        self.editor.block_selected.connect(self.view.select_block)
        self.editor.changed.connect(self._doc_changed)
        self.plain = QPlainTextEdit()
        self.plain.setReadOnly(True)
        self.preview = QTextBrowser()
        self.right = QTabWidget()
        self.right.addTab(self.editor, "Review && correct")
        self.right.addTab(self.plain, "Plain text")
        self.right.addTab(self.preview, "Preview")
        self.right.currentChanged.connect(lambda _: self._refresh_text())

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(_card("Page", self.view))
        split.addWidget(_card("Recognised content", self.right))
        split.setSizes([380, 640, 520])

        self.finetune = FinetuneTab(lambda: self.docs.get(self.cur_path) if self.cur_path else None)
        self.finetune.models_changed.connect(self._models_changed)

        self.status = QLabel("Ready - open a PDF or image")
        self.bar = QProgressBar()
        self.bar.setMaximumWidth(320)
        self.bar.setTextVisible(False)
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.bar)
        self._build_actions()

        # ---- CDTRS-style frame: navy sidebar | header + page content
        ocr_page = QWidget()
        ol = QVBoxLayout(ocr_page)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(10)
        ol.addWidget(self.toolbar)
        ol.addWidget(split, 1)
        ft_page = QWidget()
        fl2 = QVBoxLayout(ft_page)
        fl2.setContentsMargins(0, 0, 0, 0)
        fl2.addWidget(_card("", self.finetune), 1)
        self.tabs = QStackedWidget()  # same indices as before: 0 = OCR, 1 = Fine-tune
        self.tabs.addWidget(ocr_page)
        self.tabs.addWidget(ft_page)

        self.page_title = QLabel()
        self.page_title.setObjectName("pageTitle")
        self.page_subtitle = QLabel()
        self.page_subtitle.setObjectName("pageSubtitle")
        content = QWidget()
        content.setObjectName("contentArea")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(20, 16, 20, 12)
        cl.setSpacing(10)
        cl.addWidget(self.page_title)
        cl.addWidget(self.page_subtitle)
        cl.addWidget(self.tabs, 1)

        root = QWidget()
        rl = QHBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._build_sidebar())
        rl.addWidget(content, 1)
        self.setCentralWidget(root)
        self._show_section(0)
        self.setAcceptDrops(True)
        geo = self.qs.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)

    # ------------------------------------------------------------ sidebar
    SECTIONS = [
        ("📄  OCR Workspace", "OCR Workspace",
         "Open PDFs or images, run OCR, review and correct the result, then export it."),
        ("🧠  Fine-tune", "Fine-tune",
         "Train the handwriting model on your own writing and manage word corrections."),
    ]

    def _build_sidebar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("sidebar")
        bar.setMinimumWidth(200)
        bar.setMaximumWidth(240)
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(12, 14, 12, 14)
        lay.setSpacing(5)
        title = QLabel("Offline OCR")
        title.setObjectName("sidebarTitle")
        lay.addWidget(title)
        sub = QLabel("Document reader · works without internet")
        sub.setObjectName("sidebarUser")
        sub.setWordWrap(True)
        lay.addWidget(sub)
        lay.addSpacing(12)
        self.nav_buttons = []
        for i, (label, _, _) in enumerate(self.SECTIONS):
            b = QPushButton(label)
            b.setObjectName("sidebarButton")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=i: self._show_section(i))
            lay.addWidget(b)
            self.nav_buttons.append(b)
        lay.addStretch(1)
        about = QPushButton("ℹ  About")
        about.setObjectName("logoutButton")
        about.setCursor(Qt.CursorShape.PointingHandCursor)
        about.clicked.connect(self.about)
        lay.addWidget(about)
        foot = QLabel("All models run on this PC")
        foot.setObjectName("sidebarFooter")
        lay.addWidget(foot)
        return bar

    def _show_section(self, i: int) -> None:
        self.tabs.setCurrentIndex(i)
        for j, b in enumerate(self.nav_buttons):
            b.setChecked(j == i)
        _, title, subtitle = self.SECTIONS[i]
        self.page_title.setText(title)
        self.page_subtitle.setText(subtitle)

    # ------------------------------------------------------------ actions
    def _build_actions(self) -> None:
        tb = QToolBar("Main")
        tb.setObjectName("actionBar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.toolbar = tb

        def act(text, fn, key=None, tip=""):
            a = QAction(text, self)
            a.triggered.connect(fn)
            if key:
                a.setShortcut(QKeySequence(key))
            a.setToolTip(tip or text)
            tb.addAction(a)
            return a

        act("Open files...", self.open_files, "Ctrl+O")
        act("Open folder...", self.open_folder)
        tb.addSeparator()
        self.a_run = act("▶  Run OCR", self.run_current, "F5", "Recognise the selected file")
        tb.widgetForAction(self.a_run).setObjectName("primaryAction")
        self.a_run_all = act("Run all", self.run_all, "Ctrl+F5", "Recognise every file in the list")
        self.a_stop = act("Stop", self.stop)
        self.a_stop.setEnabled(False)
        tb.addSeparator()
        exp = QToolButton()
        exp.setText("Export")
        exp.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        m = QMenu(exp)
        for k, (label, ext, _) in EXPORTERS.items():
            m.addAction(f"{label} (.{ext})...", lambda k=k: self.export_current(k))
        exp.setMenu(m)
        tb.addWidget(exp)
        act("Export all...", self.export_all)
        act("Copy text", self.copy_text, "Ctrl+Shift+C")
        tb.addSeparator()
        act("Rotate 90° && re-read", self.rotate_reread)
        act("Fit", self.view.fit)
        act("Zoom +", lambda: self.view.zoom(1.25), "Ctrl+=")
        act("Zoom -", lambda: self.view.zoom(0.8), "Ctrl+-")
        self.cb_lines = QCheckBox("Show lines")
        self.cb_lines.setChecked(True)
        self.cb_lines.toggled.connect(self._toggle_overlay)
        tb.addWidget(self.cb_lines)

    # ----------------------------------------------------------- files
    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802
        paths = []
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            paths += sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in imaging.SUPPORTED) if p.is_dir() else [str(p)]
        self.add_files(paths)

    def open_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Open documents", self.qs.value("last_dir", ""), FILE_FILTER)
        if files:
            self.qs.setValue("last_dir", str(Path(files[0]).parent))
            self.add_files(files)

    def open_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open folder", self.qs.value("last_dir", ""))
        if d:
            self.qs.setValue("last_dir", d)
            self.add_files(sorted(str(f) for f in Path(d).rglob("*") if f.suffix.lower() in imaging.SUPPORTED))

    def add_files(self, files: list[str]) -> None:
        first = None
        for f in files:
            if Path(f).suffix.lower() not in imaging.SUPPORTED or f in self.docs:
                continue
            self.docs[f] = None
            it = QTreeWidgetItem([Path(f).name])
            it.setData(0, Qt.ItemDataRole.UserRole, (f, -1))
            it.setToolTip(0, f)
            self.tree.addTopLevelItem(it)
            first = first or it
        if first:
            self.tree.setCurrentItem(first)

    def _item_for(self, path: str) -> QTreeWidgetItem | None:
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole)[0] == path:
                return it
        return None

    def _tree_menu(self, pos) -> None:
        it = self.tree.itemAt(pos)
        if not it:
            return
        path = it.data(0, Qt.ItemDataRole.UserRole)[0]
        m = QMenu(self)
        m.addAction("Run OCR", lambda: self._start([path]))
        m.addAction("Remove from list", lambda: self._remove(path))
        m.addAction("Open containing folder", lambda: os.startfile(str(Path(path).parent)))
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def _remove(self, path: str) -> None:
        it = self._item_for(path)
        if it:
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(it))
        self.docs.pop(path, None)
        if self.cur_path == path:
            self.cur_path = None
            self.view.clear()
            self.editor.set_page(None)

    def _on_tree(self, it, _prev=None) -> None:
        if not it:
            return
        path, page = it.data(0, Qt.ItemDataRole.UserRole)
        self.cur_path = path
        doc = self.docs.get(path)
        if doc is None:
            try:
                pages = imaging.load_pages(path, 72) if path.lower().endswith(".pdf") else imaging.load_pages(path)
                self.view.show_image(pages[0])
                self.status.setText(f"{Path(path).name}: {len(pages)} page(s), not recognised yet - press Run OCR (F5)")
            except Exception as e:
                self.status.setText(f"Cannot open {path}: {e}")
            self.editor.set_page(None)
            self.plain.setPlainText("")
            self.preview.setHtml("")
            return
        self.cur_page = max(0, page) if page >= 0 else 0
        self._show_page()

    def _show_page(self) -> None:
        doc = self.docs.get(self.cur_path)
        if not doc or not doc.pages:
            return
        self.cur_page = min(self.cur_page, len(doc.pages) - 1)
        p = doc.pages[self.cur_page]
        self.view.show_page(p)
        self.view.fit()
        self.editor.set_page(p)
        self._refresh_text()
        lines = sum(len(b.lines) for b in p.blocks)
        low = sum(1 for b in p.blocks for l in b.lines if l.conf < 0.7)
        self.status.setText(f"{Path(self.cur_path).name} page {p.index + 1}: {len(p.blocks)} blocks, {lines} lines "
                            f"({low} low-confidence, red), rotated {p.rotation}°, deskewed {p.skew:+.1f}°, {p.seconds:.1f}s"
                            + ("  |  " + "; ".join(p.notes) if p.notes else ""))

    def _refresh_text(self) -> None:
        doc = self.docs.get(self.cur_path) if self.cur_path else None
        if not doc:
            return
        if self.right.currentWidget() is self.plain:
            self.plain.setPlainText(doc.text)
        elif self.right.currentWidget() is self.preview:
            self.preview.setHtml(to_html_string(doc, images=True))

    def _doc_changed(self) -> None:
        self.view.show_page(self.docs[self.cur_path].pages[self.cur_page])
        self.view.select_block(self.editor.list.currentRow())

    def _toggle_overlay(self, on: bool) -> None:
        self.view.show_lines = on
        self._show_page()

    def _settings_changed(self) -> None:
        self.engine.s = self.settings

    def _models_changed(self) -> None:
        self.panel.reload_models()
        self.engine.unload_handwriting()

    # ------------------------------------------------------------ OCR
    def run_current(self) -> None:
        if self.cur_path:
            self._start([self.cur_path])
        else:
            self.open_files()

    def run_all(self) -> None:
        if self.docs:
            self._start(list(self.docs))

    def _start(self, files: list[str]) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "OCR is already running.")
            return
        self.worker = OCRWorker(self.engine, files)
        self.worker.progress.connect(self._progress)
        self.worker.doc_done.connect(self._doc_done)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._finished)
        self.a_run.setEnabled(False)
        self.a_run_all.setEnabled(False)
        self.a_stop.setEnabled(True)
        self.status.setText("Loading models (first run takes a minute) ...")
        self.worker.start()

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()
            self.status.setText("Stopping after the current page ...")

    def _progress(self, msg: str, frac: float) -> None:
        self.status.setText(msg)
        self.bar.setValue(int(frac * 100))

    def _doc_done(self, path: str, doc) -> None:
        self.docs[path] = doc
        it = self._item_for(path)
        if it:
            it.takeChildren()
            it.setText(0, "✓ " + Path(path).name)
            for p in doc.pages:
                c = QTreeWidgetItem([f"Page {p.index + 1}"])
                c.setData(0, Qt.ItemDataRole.UserRole, (path, doc.pages.index(p)))
                it.addChild(c)
            it.setExpanded(len(doc.pages) > 1)
        if path == self.cur_path:
            self.cur_page = 0
            self._show_page()

    def _failed(self, path: str, err: str) -> None:
        QMessageBox.warning(self, "OCR failed", f"{Path(path).name}:\n{err[:1500]}")

    def _finished(self) -> None:
        self.a_run.setEnabled(True)
        self.a_run_all.setEnabled(True)
        self.a_stop.setEnabled(False)
        self.bar.setValue(100)
        if self.docs.get(self.cur_path):
            self._show_page()

    def rotate_reread(self) -> None:
        doc = self.docs.get(self.cur_path) if self.cur_path else None
        if not doc:
            return
        p = doc.pages[self.cur_page]
        img = imaging.rotate90(p.image, 270)  # 90° clockwise
        s = self.settings
        orient, deskew = s.auto_orient, s.deskew
        idx = self.cur_page

        def job(log, prog, stop):
            s.auto_orient, s.deskew = False, False
            try:
                return self.engine.process_image(img, index=p.index)
            finally:
                s.auto_orient, s.deskew = orient, deskew

        w = FuncWorker(job)
        w.result.connect(lambda page: (doc.pages.__setitem__(idx, page), self._show_page()))
        w.failed.connect(lambda e: QMessageBox.warning(self, "Failed", e[:1500]))
        self.status.setText("Re-reading rotated page ...")
        self._rot_worker = w
        w.start()

    # --------------------------------------------------------- export
    def export_current(self, fmt: str) -> None:
        doc = self.docs.get(self.cur_path) if self.cur_path else None
        if not doc:
            QMessageBox.information(self, "Export", "Run OCR on a file first.")
            return
        label, ext, _ = EXPORTERS[fmt]
        default = str(Path(self.qs.value("export_dir", str(OUTPUT))) / f"{Path(self.cur_path).stem}.{ext}")
        dest, _ = QFileDialog.getSaveFileName(self, f"Export {label}", default, f"{label} (*.{ext.split('.')[-1]})")
        if not dest:
            return
        self.qs.setValue("export_dir", str(Path(dest).parent))
        try:
            export(doc, fmt, dest)
            self.status.setText(f"Saved {dest}")
            if QMessageBox.question(self, "Exported", f"Saved {dest}\n\nOpen it now?") == QMessageBox.StandardButton.Yes:
                os.startfile(dest)
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))

    def export_all(self) -> None:
        done = {p: d for p, d in self.docs.items() if d}
        if not done:
            QMessageBox.information(self, "Export", "No recognised documents yet.")
            return
        dlg = ExportDialog(self, self.qs.value("export_dir", str(OUTPUT)))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        folder = Path(dlg.folder.text())
        self.qs.setValue("export_dir", str(folder))
        n = 0
        for p, d in done.items():
            for fmt in dlg.formats():
                try:
                    export(d, fmt, folder / f"{Path(p).stem}.{EXPORTERS[fmt][1]}")
                    n += 1
                except Exception as e:
                    QMessageBox.warning(self, "Export failed", f"{Path(p).name} -> {fmt}: {e}")
        self.status.setText(f"Wrote {n} files to {folder}")
        os.startfile(str(folder))

    def copy_text(self) -> None:
        doc = self.docs.get(self.cur_path) if self.cur_path else None
        if doc:
            QApplication.clipboard().setText(doc.text)
            self.status.setText("Text copied to clipboard")

    def about(self) -> None:
        QMessageBox.about(self, "Offline OCR", (
            "<b>Offline OCR</b><br>Runs entirely on this PC - no internet needed.<br><br>"
            "Text: PaddleOCR PP-OCRv4<br>Layout + tables: PP-Structure (PicoDet CDLA, SLANet)<br>"
            "Handwriting: TrOCR (fine-tunable)<br>Formulas: LaTeX-OCR (ONNX)<br><br>"
            f"Project folder: {ROOT}"))

    def closeEvent(self, e):  # noqa: N802
        self.qs.setValue("geometry", self.saveGeometry())
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        super().closeEvent(e)
