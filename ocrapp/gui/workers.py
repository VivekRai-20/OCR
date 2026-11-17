"""Background threads so the window stays responsive while models run."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal


class OCRWorker(QThread):
    progress = Signal(str, float)       # message, overall fraction
    doc_done = Signal(str, object)      # path, Document
    failed = Signal(str, str)           # path, error text

    def __init__(self, engine, files: list[str], pages: dict | None = None) -> None:
        super().__init__()
        self.engine = engine
        self.files = files
        self.pages = pages or {}
        self._stop = False

    def stop(self) -> None:
        self._stop = True
        self.engine.cancelled = True

    def run(self) -> None:
        n = len(self.files)
        for i, f in enumerate(self.files):
            if self._stop:
                break

            def cb(msg: str, frac: float, i=i, f=f) -> None:
                name = f.replace("\\", "/").split("/")[-1]
                self.progress.emit(f"[{i + 1}/{n}] {name} - {msg}", (i + frac) / n)

            try:
                doc = self.engine.process_file(f, progress=cb, pages=self.pages.get(f))
                self.doc_done.emit(f, doc)
            except Exception as e:  # report and continue with the next file
                self.failed.emit(f, f"{e}\n\n{traceback.format_exc()}")
        self.progress.emit("Ready", 1.0)


class FuncWorker(QThread):
    """Runs fn(log, progress, should_stop) in the background."""
    log = Signal(str)
    prog = Signal(float)
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            r = self.fn(self.log.emit, self.prog.emit, lambda: self._stop)
            self.result.emit(r)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")
