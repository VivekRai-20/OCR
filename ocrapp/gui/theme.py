"""CDTRS look for the OCR app: the CDTRS light palette and stylesheet
(styles/theme.qss is CDTRS's frontend/styles/theme.qss plus a few rules for
widgets CDTRS does not use)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QPalette

STYLES = Path(__file__).resolve().parent / "styles"
PAGE_BG = "#F5F7FA"
TEXT = "#1F2937"


def apply_light_theme(app) -> None:
    """Same fixed light palette as CDTRS (ui/light_theme.py), so Windows dark mode
    does not turn panels black."""
    palette = QPalette()
    colours = {
        QPalette.ColorRole.Window: PAGE_BG,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: "#FFFFFF",
        QPalette.ColorRole.AlternateBase: "#F8FAFC",
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: "#F1F5F9",
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.BrightText: "#FFFFFF",
        QPalette.ColorRole.Highlight: "#2563EB",
        QPalette.ColorRole.HighlightedText: "#FFFFFF",
        QPalette.ColorRole.ToolTipBase: "#0F172A",
        QPalette.ColorRole.ToolTipText: "#FFFFFF",
        QPalette.ColorRole.PlaceholderText: "#94A3B8",
        QPalette.ColorRole.Link: "#2563EB",
        QPalette.ColorRole.Light: "#FFFFFF",
        QPalette.ColorRole.Midlight: "#E2E8F0",
        QPalette.ColorRole.Mid: "#CBD5E1",
        QPalette.ColorRole.Dark: "#94A3B8",
        QPalette.ColorRole.Shadow: "#475569",
    }
    for role, colour in colours.items():
        palette.setColor(QPalette.ColorGroup.All, role, QColor(colour))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#94A3B8"))
    app.setPalette(palette)
    try:
        from PySide6.QtCore import Qt

        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    except Exception:
        pass


def apply(app) -> None:
    app.setStyle("Fusion")
    apply_light_theme(app)
    app.setFont(QFont("Segoe UI", 10))
    qss = (STYLES / "theme.qss").read_text(encoding="utf-8")
    app.setStyleSheet(qss.replace("@STYLES@", STYLES.as_posix()))
