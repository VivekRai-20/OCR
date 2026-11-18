"""Offline OCR - desktop app entry point.  Run:  run.bat  (or  .venv\\Scripts\\python app.py)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocrapp import compat  # noqa: E402

compat.prepare()  # torch before paddle, offline flags, import shims

from PySide6.QtWidgets import QApplication  # noqa: E402


def main() -> None:
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
