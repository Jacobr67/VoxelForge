"""
src/gui/components/export_panel.py

Export panel: format selector (.schematic / .schem / .litematica),
output path chooser, and Export button. Disabled until a voxelised
structure is ready. Emits export_requested signal with format + path.
"""

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QLineEdit, QFileDialog,
    QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt


FORMAT_INFO = {
    ".schematic": "Legacy MCEdit format — Minecraft 1.12 and older",
    ".schem":     "Modern WorldEdit / FAWE — Minecraft 1.13+  ✓ Recommended",
    ".litematica":"Litematica mod — in-game holographic overlay",
}

EXPORT_BTN_READY = """
    QPushButton {
        background-color: #1a5c2a;
        color: #c8f0d8;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        font-size: 13px;
        font-weight: bold;
        letter-spacing: 0.5px;
    }
    QPushButton:hover  { background-color: #207a38; }
    QPushButton:pressed { background-color: #154820; }
"""

EXPORT_BTN_DISABLED = """
    QPushButton {
        background-color: #162018;
        color: #2a4030;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        font-size: 13px;
        font-weight: bold;
    }
"""

INPUT_STYLE = """
    QLineEdit {
        background-color: #1a1a1a;
        color: #c0c0c0;
        border: 1px solid #2e2e2e;
        border-radius: 3px;
        padding: 5px 8px;
        font-size: 12px;
    }
    QLineEdit:focus { border: 1px solid #5a9fd4; }
"""

COMBO_STYLE = """
    QComboBox {
        background-color: #1a1a1a;
        color: #e8e8e8;
        border: 1px solid #2e2e2e;
        border-radius: 3px;
        padding: 5px 10px;
        font-size: 12px;
    }
    QComboBox::drop-down { border: none; width: 20px; }
    QComboBox QAbstractItemView {
        background-color: #1e1e1e;
        color: #e8e8e8;
        selection-background-color: #1e5c8a;
        border: 1px solid #2a2a2a;
    }
"""


class ExportPanel(QWidget):
    """
    Format selection and export controls.

    Signals:
        export_requested(str, str): Emitted with (format_ext, output_path)
                                    when user clicks Export.
    """

    export_requested = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ready = False
        self._build_ui()
        self._set_ready(False)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_ready(self, ready: bool):
        """Enable or disable export controls based on pipeline state."""
        self._ready = ready
        self._set_ready(ready)

    def set_status(self, message: str, colour: str = "#a0a0a0"):
        self._status_label.setText(message)
        self._status_label.setStyleSheet(
            f"color: {colour}; font-size: 11px;"
        )

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Section label
        section_label = QLabel("EXPORT")
        section_label.setStyleSheet(
            "color: #5a9fd4; font-size: 10px; font-weight: bold; "
            "letter-spacing: 2px;"
        )
        layout.addWidget(section_label)

        # Format selector
        fmt_row = QHBoxLayout()
        fmt_label = QLabel("Format")
        fmt_label.setStyleSheet("color: #909090; font-size: 12px;")
        fmt_label.setFixedWidth(56)
        fmt_row.addWidget(fmt_label)

        self._format_combo = QComboBox()
        for fmt in FORMAT_INFO:
            self._format_combo.addItem(fmt)
        self._format_combo.setCurrentText(".schem")
        self._format_combo.setStyleSheet(COMBO_STYLE)
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        fmt_row.addWidget(self._format_combo, stretch=1)
        layout.addLayout(fmt_row)

        # Format hint
        self._fmt_hint = QLabel(FORMAT_INFO[".schem"])
        self._fmt_hint.setStyleSheet("color: #4a7a5a; font-size: 10px;")
        self._fmt_hint.setWordWrap(True)
        layout.addWidget(self._fmt_hint)

        # Output path
        path_row = QHBoxLayout()
        path_label = QLabel("Save to")
        path_label.setStyleSheet("color: #909090; font-size: 12px;")
        path_label.setFixedWidth(56)
        path_row.addWidget(path_label)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Choose output location…")
        self._path_edit.setStyleSheet(INPUT_STYLE)
        path_row.addWidget(self._path_edit, stretch=1)

        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(32)
        browse_btn.setFixedHeight(32)
        browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #222222;
                color: #a0a0a0;
                border: 1px solid #2e2e2e;
                border-radius: 3px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2a2a2a; color: #e0e0e0; }
        """)
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        # Export button
        self._export_btn = QPushButton("⬇  Export Structure")
        self._export_btn.setMinimumHeight(40)
        self._export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_btn.clicked.connect(self._on_export_clicked)
        layout.addWidget(self._export_btn)

        # Status
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #505050; font-size: 11px;")
        layout.addWidget(self._status_label)

    # ── Interaction ────────────────────────────────────────────────────────────

    def _set_ready(self, ready: bool):
        self._export_btn.setEnabled(ready)
        self._export_btn.setStyleSheet(
            EXPORT_BTN_READY if ready else EXPORT_BTN_DISABLED
        )
        if not ready:
            self.set_status("Generate a structure first.", "#404040")

    def _on_format_changed(self, fmt: str):
        self._fmt_hint.setText(FORMAT_INFO.get(fmt, ""))
        # Update file extension in path if already set
        current = self._path_edit.text()
        if current:
            p = Path(current)
            self._path_edit.setText(str(p.with_suffix(fmt)))

    def _browse(self):
        fmt = self._format_combo.currentText()
        filter_str = {
            ".schematic": "Schematic Files (*.schematic)",
            ".schem":     "Schem Files (*.schem)",
            ".litematica":"Litematica Files (*.litematica)",
        }.get(fmt, "All Files (*)")

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Structure As",
            str(Path.home() / f"structure{fmt}"),
            filter_str,
        )
        if path:
            # Ensure correct extension
            p = Path(path)
            if p.suffix != fmt:
                path = str(p.with_suffix(fmt))
            self._path_edit.setText(path)

    def _on_export_clicked(self):
        output_path = self._path_edit.text().strip()
        if not output_path:
            self.set_status("⚠  Please choose a save location.", "#e8a44a")
            return

        fmt = self._format_combo.currentText()
        self.export_requested.emit(fmt, output_path)