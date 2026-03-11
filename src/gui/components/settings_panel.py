"""
src/gui/components/settings_panel.py

Generation settings panel: voxel resolution, texture resolution,
seed input, and hollow toggle.
Exposes a get_settings() method returning a clean dict consumed by
the generation pipeline.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QCheckBox, QComboBox,
    QFrame,
)
from PyQt6.QtCore import Qt


# ── Shared style helpers ───────────────────────────────────────────────────────

SECTION_LABEL_STYLE = (
    "color: #5a9fd4; font-size: 10px; font-weight: bold; letter-spacing: 2px;"
)
ROW_LABEL_STYLE = "color: #909090; font-size: 12px;"

SPINBOX_STYLE = """
    QSpinBox {
        background-color: #1a1a1a;
        color: #e8e8e8;
        border: 1px solid #2e2e2e;
        border-radius: 3px;
        padding: 3px 6px;
        font-size: 12px;
    }
    QSpinBox:focus { border: 1px solid #5a9fd4; }
"""

CHECKBOX_STYLE = """
    QCheckBox {
        color: #c0c0c0;
        font-size: 12px;
        spacing: 8px;
    }
    QCheckBox::indicator {
        width: 15px; height: 15px;
        border: 1px solid #3a3a3a;
        border-radius: 3px;
        background: #1a1a1a;
    }
    QCheckBox::indicator:checked {
        background: #1e5c8a;
        border-color: #5a9fd4;
    }
"""


class SettingsPanel(QWidget):
    """
    Settings panel for generation and voxelisation parameters.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_settings(self) -> dict:
        """
        Return all current settings as a dict consumed by the pipeline.

        Returns:
            {
                "voxel_resolution":   int,   # 16–256
                "texture_resolution": int,   # 512 / 1024 / 2048
                "seed":               int,   # 0 = random
                "hollow":             bool,
            }
        """
        return {
            "voxel_resolution":   self._resolution_spin.value(),
            "texture_resolution": int(self._tex_res_combo.currentText()),
            "seed":               self._seed_spin.value(),
            "hollow":             self._hollow_check.isChecked(),
        }

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ── Voxelisation ──────────────────────────────────────────────────────
        layout.addWidget(self._section("VOXELISATION"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        # Resolution
        grid.addWidget(self._row_label("Resolution"), 0, 0)
        res_row = QHBoxLayout()
        self._resolution_spin = QSpinBox()
        self._resolution_spin.setRange(16, 256)
        self._resolution_spin.setValue(64)
        self._resolution_spin.setSuffix("  blocks")
        self._resolution_spin.setStyleSheet(SPINBOX_STYLE)
        self._resolution_spin.setFixedWidth(110)
        res_row.addWidget(self._resolution_spin)
        res_row.addStretch()
        grid.addLayout(res_row, 0, 1)

        # Hollow toggle
        self._hollow_check = QCheckBox("Hollow interior")
        self._hollow_check.setChecked(False)
        self._hollow_check.setStyleSheet(CHECKBOX_STYLE)
        grid.addWidget(self._hollow_check, 1, 0, 1, 2)

        layout.addLayout(grid)

        layout.addWidget(self._divider())

        # ── AI Generation ─────────────────────────────────────────────────────
        layout.addWidget(self._section("AI GENERATION"))

        gen_grid = QGridLayout()
        gen_grid.setHorizontalSpacing(12)
        gen_grid.setVerticalSpacing(10)

        # Texture resolution
        gen_grid.addWidget(self._row_label("Texture res."), 0, 0)
        self._tex_res_combo = QComboBox()
        self._tex_res_combo.addItems(["512", "1024", "2048"])
        self._tex_res_combo.setCurrentText("1024")
        self._tex_res_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a1a;
                color: #e8e8e8;
                border: 1px solid #2e2e2e;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 12px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #e8e8e8;
                selection-background-color: #1e5c8a;
            }
        """)
        self._tex_res_combo.setFixedWidth(80)
        gen_grid.addWidget(self._tex_res_combo, 0, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        # Seed
        gen_grid.addWidget(self._row_label("Seed"), 1, 0)
        seed_row = QHBoxLayout()
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(0)
        self._seed_spin.setSpecialValueText("Random")
        self._seed_spin.setStyleSheet(SPINBOX_STYLE)
        self._seed_spin.setFixedWidth(110)
        seed_row.addWidget(self._seed_spin)
        seed_row.addStretch()
        gen_grid.addLayout(seed_row, 1, 1)

        layout.addLayout(gen_grid)
        layout.addStretch()

    # ── Widget factories ───────────────────────────────────────────────────────

    def _section(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(SECTION_LABEL_STYLE)
        return label

    def _row_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(ROW_LABEL_STYLE)
        return label

    def _divider(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #2a2a2a;")
        return line