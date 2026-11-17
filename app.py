"""Offline OCR - desktop app entry point.  Run:  run.bat  (or  .venv\\Scripts\\python app.py)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocrapp import compat  # noqa: E402

compat.prepare()  # torch before paddle, offline flags, import shims

from PySide6.QtWidgets import QApplication  # noqa: E402


def selftest(path: str) -> None:
    """OfflineOCR.exe --selftest <file>: OCR one file without the window and write
    the result to user/selftest.log (used to check a packaged build)."""
    import time
    import traceback

    from ocrapp.paths import USER

    log = USER / "selftest.log"
    try:
        from ocrapp.engine import OCREngine
        from ocrapp.exporters import export

        t = time.time()
        doc = OCREngine().process_file(path)
        export(doc, "docx", USER / "selftest.docx")
        log.write_text(f"OK {time.time() - t:.0f}s notes={[p.notes for p in doc.pages]}\n\n{doc.text}",
                       encoding="utf-8")
    except Exception:
        log.write_text("FAILED\n" + traceback.format_exc(), encoding="utf-8")


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--selftest":
        selftest(sys.argv[2])
        return
    app = QApplication(sys.argv)
    app.setApplicationName("Offline OCR")
    from ocrapp.gui import theme

    theme.apply(app)  # CDTRS look (palette + stylesheet)
    from ocrapp.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.add_files(sys.argv[1:])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
