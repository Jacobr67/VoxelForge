"""
src/gui/components/palette_selector.py

Block type filter panel. Each chip represents a category of blocks that
can be EXCLUDED from the colour mapping output. All categories are included
by default — toggling a chip OFF removes that block type from the palette.

This is the inverse of the old "groups" model:
  Old: activate groups to ADD blocks
  New: deactivate filters to REMOVE block types

The active set passed to ColourMapper is the union of all non-excluded
palette JSON files.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt


# ── Filter definitions ─────────────────────────────────────────────────────────
# Each filter maps to a block palette JSON in assets/block_palettes/.
# When a filter is ENABLED (default), its blocks are included.
# When DISABLED, its blocks are excluded from the output.

BLOCK_FILTERS = [
    {
        "id":          "natural",
        "label":       "Natural",
        "description": "Include natural terrain blocks: dirt, grass, sand, gravel, clay…",
        "icon":        "🌿",
    },
    {
        "id":          "stone_and_ores",
        "label":       "Stone & Ores",
        "description": "Include stone variants, ore blocks, and metal blocks.",
        "icon":        "⛰",
    },
    {
        "id":          "wood_and_leaves",
        "label":       "Wood & Leaves",
        "description": "Include log, plank, and leaf blocks from all wood types.",
        "icon":        "🌲",
    },
    {
        "id":          "coloured_blocks",
        "label":       "Coloured",
        "description": "Include wool, concrete, terracotta, and glazed terracotta.",
        "icon":        "🎨",
    },
    {
        "id":          "survival_obtainable",
        "label":       "Survival Only",
        "description": "Restrict palette to survival-obtainable blocks only.",
        "icon":        "⛏",
    },
    {
        "id":          "full_blocks",
        "label":       "Full Blocks",
        "description": "Include only solid full-cube blocks (no slabs, stairs, etc.).",
        "icon":        "▪",
    },
]

# All filters enabled by default
_DEFAULT_ENABLED = {f["id"] for f in BLOCK_FILTERS}


CHIP_ON_STYLE = """
    QPushButton {
        background-color: #1a3d1a;
        color: #90e890;
        border: 1px solid #3a7a3a;
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 11px;
        font-weight: bold;
        text-align: left;
    }
    QPushButton:hover { background-color: #224a22; }
"""

CHIP_OFF_STYLE = """
    QPushButton {
        background-color: #1a1a1a;
        color: #555555;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 11px;
        text-align: left;
        text-decoration: line-through;
    }
    QPushButton:hover {
        background-color: #222222;
        color: #888888;
        border-color: #3a3a3a;
    }
"""


class PaletteSelector(QWidget):
    """
    Block type filter chips.

    Each chip can be toggled ON (included) or OFF (excluded).
    At least one filter must remain ON.

    Signals:
        selection_changed(list): Emitted with list of ENABLED filter IDs.
    """

    selection_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        # All enabled by default
        self._enabled: set[str] = set(_DEFAULT_ENABLED)
        self._buttons: dict[str, QPushButton] = {}
        self._build_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_active_groups(self) -> list[str]:
        """Return list of enabled (included) filter IDs for ColourMapper."""
        return list(self._enabled)

    def get_excluded_groups(self) -> list[str]:
        """Return list of disabled (excluded) filter IDs."""
        all_ids = {f["id"] for f in BLOCK_FILTERS}
        return list(all_ids - self._enabled)

    def set_enabled_filters(self, filter_ids: list[str]):
        """Programmatically set which filters are enabled."""
        self._enabled = set(filter_ids)
        if not self._enabled:
            # Prevent empty palette
            self._enabled = set(_DEFAULT_ENABLED)
        self._refresh_buttons()
        self._update_summary()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header
        header_row = QHBoxLayout()
        title = QLabel("BLOCK FILTERS")
        title.setStyleSheet(
            "color: #5a9fd4; font-size: 10px; font-weight: bold; letter-spacing: 2px;"
        )
        header_row.addWidget(title)
        header_row.addStretch()

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("color: #505050; font-size: 10px;")
        header_row.addWidget(self._summary_label)
        layout.addLayout(header_row)

        # Subtitle
        subtitle = QLabel("Toggle OFF to exclude block types from the output.")
        subtitle.setStyleSheet("color: #404040; font-size: 10px;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # 2-column chip grid
        grid = QGridLayout()
        grid.setSpacing(6)

        for i, filt in enumerate(BLOCK_FILTERS):
            btn = QPushButton(f"{filt['icon']}  {filt['label']}")
            btn.setToolTip(filt["description"])
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setCheckable(False)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setMinimumHeight(34)

            fid = filt["id"]
            btn.clicked.connect(lambda _, f=fid: self._toggle(f))
            self._buttons[fid] = btn

            row, col = divmod(i, 2)
            grid.addWidget(btn, row, col)

        layout.addLayout(grid)

        self._refresh_buttons()
        self._update_summary()

    # ── Interaction ────────────────────────────────────────────────────────────

    def _toggle(self, filter_id: str):
        if filter_id in self._enabled:
            # Don't allow turning off the last active filter
            if len(self._enabled) <= 1:
                return
            self._enabled.discard(filter_id)
        else:
            self._enabled.add(filter_id)

        self._refresh_buttons()
        self._update_summary()
        self.selection_changed.emit(list(self._enabled))

    def _refresh_buttons(self):
        for fid, btn in self._buttons.items():
            if fid in self._enabled:
                btn.setStyleSheet(CHIP_ON_STYLE)
            else:
                btn.setStyleSheet(CHIP_OFF_STYLE)

    def _update_summary(self):
        excluded = len(BLOCK_FILTERS) - len(self._enabled)
        if excluded == 0:
            self._summary_label.setText("all included")
            self._summary_label.setStyleSheet("color: #3a7a3a; font-size: 10px;")
        else:
            word = "type" if excluded == 1 else "types"
            self._summary_label.setText(f"{excluded} {word} excluded")
            self._summary_label.setStyleSheet("color: #8a5a20; font-size: 10px;")