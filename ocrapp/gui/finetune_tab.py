"""Fine-tune tab: handwriting models, training data, training, and the corrections dictionary."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

from .. import finetune, handwriting
from ..settings import load_corrections, save_corrections
from .workers import FuncWorker


class FinetuneTab(QWidget):
    models_changed = Signal()

    def __init__(self, get_document, parent=None) -> None:
        super().__init__(parent)
        self.get_document = get_document
        self.worker = None
        tabs = QTabWidget()
        tabs.addTab(self._models_page(), "Handwriting models")
        tabs.addTab(self._data_page(), "Training data")
        tabs.addTab(self._train_page(), "Train")
        tabs.addTab(self._corr_page(), "Corrections dictionary")
        lay = QVBoxLayout(self)
        intro = QLabel(
            "Teach the app your handwriting (or any script/font it misreads): correct lines in the OCR tab, "
            "add them as training data, then train a new model here. Training runs on this PC, offline. "
            "The corrections dictionary fixes words the OCR repeatedly gets wrong, without training.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        lay.addWidget(tabs)
        self.refresh()

    # ----------------------------------------------------------- models
    def _models_page(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        self.models = QTableWidget(0, 5)
        self.models.setHorizontalHeaderLabels(["Model", "Active", "Kind", "Training lines", "Validation CER"])
        self.models.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.models.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.models.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        l.addWidget(self.models)
        row = QHBoxLayout()
        for label, fn in (("Use selected model", self._activate), ("Evaluate on training data", self._evaluate),
                          ("Delete selected", self._delete_model), ("Refresh", self.refresh)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        l.addLayout(row)
        return w

    def _selected_model(self) -> str:
        r = self.models.currentRow()
        return self.models.item(r, 0).text() if r >= 0 else ""

    def _activate(self) -> None:
        m = self._selected_model()
        if m:
            handwriting.set_active_model(m)
            self.refresh()
            self.models_changed.emit()

    def _delete_model(self) -> None:
        m = self._selected_model()
        if not m or m == "trocr-small-handwritten":
            QMessageBox.information(self, "Delete", "The base model cannot be deleted.")
            return
        if QMessageBox.question(self, "Delete model", f"Delete model '{m}' from disk?") == QMessageBox.StandardButton.Yes:
            finetune.delete_model(m)
            if handwriting.active_model() == "":
                handwriting.set_active_model("trocr-small-handwritten")
            self.refresh()
            self.models_changed.emit()

    def _evaluate(self) -> None:
        m = self._selected_model()
        if not m:
            return
        self._run(lambda log, prog, stop: finetune.evaluate(m, log), f"Evaluating {m} ...")

    # ------------------------------------------------------------- data
    def _data_page(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        self.data_info = QLabel()
        l.addWidget(self.data_info)
        self.samples = QTableWidget(0, 2)
        self.samples.setHorizontalHeaderLabels(["Line image", "Correct text (edit here)"])
        self.samples.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.samples.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.samples.setColumnWidth(0, 420)
        self.samples.setIconSize(QPixmap(400, 48).size())
        l.addWidget(self.samples)
        row = QHBoxLayout()
        self.only_hw = QCheckBox("only lines read as handwriting")
        for label, fn in (("Add lines from current document", self._add_from_doc),):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addWidget(self.only_hw)
        for label, fn in (("Import folder...", self._import), ("Save text edits", self._save_samples),
                          ("Delete selected", self._delete_samples), ("Reload", self._load_samples)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        l.addLayout(row)
        hint = QLabel("Import folder: line images with a labels.tsv (image<TAB>text), or image + same-name .txt files.")
        hint.setStyleSheet("color:#666")
        l.addWidget(hint)
        return w

    def _load_samples(self) -> None:
        s = finetune.load_samples()
        self.samples.setRowCount(0)
        show = s[-500:]  # the newest 500 (keeps the table fast)
        self.samples.setRowCount(len(show))
        for i, x in enumerate(show):
            it = QTableWidgetItem()
            pm = QPixmap(str(x.path))
            if not pm.isNull():
                it.setIcon(QIcon(pm.scaledToHeight(44, Qt.TransformationMode.SmoothTransformation)))
            it.setData(Qt.ItemDataRole.UserRole, x.file)
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.samples.setItem(i, 0, it)
            self.samples.setItem(i, 1, QTableWidgetItem(x.text))
            self.samples.setRowHeight(i, 50)
        self.data_info.setText(f"{len(s)} labelled line images in datasets/handwriting"
                               + (" (showing the newest 500)" if len(s) > 500 else ""))

    def _save_samples(self) -> None:
        edits = {self.samples.item(i, 0).data(Qt.ItemDataRole.UserRole): self.samples.item(i, 1).text()
                 for i in range(self.samples.rowCount())}
        s = finetune.load_samples()
        for x in s:
            if x.file in edits:
                x.text = edits[x.file]
        finetune.save_samples([x for x in s if x.text.strip()])
        self._load_samples()

    def _delete_samples(self) -> None:
        rows = {i.row() for i in self.samples.selectedIndexes()}
        files = {self.samples.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows}
        if files:
            finetune.delete_samples(files)
            self._load_samples()

    def _add_from_doc(self) -> None:
        doc = self.get_document()
        if doc is None or not doc.pages:
            QMessageBox.information(self, "No document", "Run OCR on a document first, correct its lines, then add them.")
            return
        n = finetune.add_from_document(doc, self.only_hw.isChecked())
        self._load_samples()
        QMessageBox.information(self, "Added", f"Added {n} line images. Check their text in this table before training.")

    def _import(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder with line images and labels")
        if d:
            n = finetune.import_folder(d)
            self._load_samples()
            QMessageBox.information(self, "Imported", f"Imported {n} labelled line images.")

    # ------------------------------------------------------------ train
    def _train_page(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        box = QGroupBox("Training settings")
        f = QFormLayout(box)
        self.base = QComboBox()
        self.name = QLineEdit("my-handwriting")
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 200)
        self.epochs.setValue(8)
        self.bs = QSpinBox()
        self.bs.setRange(1, 64)
        self.bs.setValue(4)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6)
        self.lr.setRange(1e-6, 1e-3)
        self.lr.setSingleStep(1e-5)
        self.lr.setValue(3e-5)
        self.val = QSpinBox()
        self.val.setRange(5, 50)
        self.val.setValue(15)
        self.val.setSuffix(" %")
        self.aug = QCheckBox("augment (slant, rotation, blur, stroke width)")
        self.aug.setChecked(True)
        self.freeze = QCheckBox("freeze image encoder (faster, needs less data)")
        f.addRow("Start from model", self.base)
        f.addRow("New model name", self.name)
        f.addRow("Epochs", self.epochs)
        f.addRow("Batch size", self.bs)
        f.addRow("Learning rate", self.lr)
        f.addRow("Validation split", self.val)
        f.addRow("", self.aug)
        f.addRow("", self.freeze)
        l.addWidget(box)
        row = QHBoxLayout()
        self.btn_train = QPushButton("Start training")
        self.btn_train.clicked.connect(self._train)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self.worker and self.worker.stop())
        row.addWidget(self.btn_train)
        row.addWidget(self.btn_stop)
        row.addStretch(1)
        l.addLayout(row)
        self.bar = QProgressBar()
        l.addWidget(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        l.addWidget(self.log, 1)
        return w

    def _train(self) -> None:
        name = "".join(c for c in self.name.text().strip() if c.isalnum() or c in "-_.") or "my-handwriting"
        cfg = finetune.TrainConfig(base=self.base.currentText(), name=name, epochs=self.epochs.value(),
                                   batch_size=self.bs.value(), lr=self.lr.value(), val_fraction=self.val.value() / 100,
                                   augment=self.aug.isChecked(), freeze_encoder=self.freeze.isChecked())
        self.log.clear()
        self._run(lambda log, prog, stop: finetune.train(cfg, log, prog, stop), "Training ...", training=True)

    def _run(self, fn, msg: str, training: bool = False) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "Another job is still running.")
            return
        self.log.appendPlainText(msg)
        self.bar.setValue(0)
        self.worker = FuncWorker(fn)
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.prog.connect(lambda f: self.bar.setValue(int(f * 100)))
        self.worker.failed.connect(lambda e: (self.log.appendPlainText("ERROR: " + e),
                                              QMessageBox.warning(self, "Failed", e.split("\n")[0])))
        self.worker.result.connect(self._done)
        self.worker.finished.connect(lambda: (self.btn_train.setEnabled(True), self.btn_stop.setEnabled(False)))
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(training)
        self.worker.start()

    def _done(self, r) -> None:
        self.bar.setValue(100)
        self.refresh()
        if isinstance(r, dict) and r.get("model"):
            if QMessageBox.question(self, "Training finished",
                                    f"New model '{r['model']}': validation CER {r['base_cer']:.1%} -> {r['best_cer']:.1%}.\n"
                                    "Use it for handwriting recognition now?") == QMessageBox.StandardButton.Yes:
                handwriting.set_active_model(r["model"])
                self.refresh()
            self.models_changed.emit()

    # ------------------------------------------------------- corrections
    def _corr_page(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("Whole-word replacements applied to every OCR result (case-sensitive)."))
        self.corr = QTableWidget(0, 2)
        self.corr.setHorizontalHeaderLabels(["OCR reads", "Replace with"])
        self.corr.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        l.addWidget(self.corr)
        row = QHBoxLayout()
        for label, fn in (("Add row", lambda: self.corr.insertRow(self.corr.rowCount())),
                          ("Delete selected", lambda: [self.corr.removeRow(r) for r in sorted(
                              {i.row() for i in self.corr.selectedIndexes()}, reverse=True)]),
                          ("Save", self._save_corr)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        l.addLayout(row)
        return w

    def _load_corr(self) -> None:
        c = load_corrections()
        self.corr.setRowCount(len(c))
        for i, (a, b) in enumerate(c.items()):
            self.corr.setItem(i, 0, QTableWidgetItem(a))
            self.corr.setItem(i, 1, QTableWidgetItem(b))

    def _save_corr(self) -> None:
        d = {}
        for i in range(self.corr.rowCount()):
            a, b = self.corr.item(i, 0), self.corr.item(i, 1)
            if a and a.text().strip():
                d[a.text().strip()] = b.text() if b else ""
        save_corrections(d)
        QMessageBox.information(self, "Saved", f"{len(d)} corrections saved. They apply to the next OCR run.")

    # ------------------------------------------------------------- misc
    def refresh(self) -> None:
        active = handwriting.active_model()
        names = handwriting.list_models()
        self.models.setRowCount(len(names))
        for i, n in enumerate(names):
            meta = handwriting.model_meta(n)
            cer = (meta.get("metrics") or {}).get("val_cer")
            vals = [n, "yes" if n == active else "", meta.get("kind", ""),
                    str(meta.get("training_samples", "")), f"{cer:.1%}" if isinstance(cer, (int, float)) else ""]
            for j, v in enumerate(vals):
                self.models.setItem(i, j, QTableWidgetItem(v))
        cur = self.base.currentText()
        self.base.clear()
        self.base.addItems(names)
        if cur in names:
            self.base.setCurrentText(cur)
        self._load_samples()
        self._load_corr()
