"""
src/gui/components/palette_selector.py

Block palette group selector. Displays all available palette groups
as toggleable chips. The active selection is passed to ColourMapper.
Shows a live count of how many blocks are in the combined active palette.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt


# ── Palette group definitions ──────────────────────────────────────────────────

PALETTE_GROUPS = [
    {
        "id":          "full_blocks",
        "label":       "Full Blocks",
        "description": "Solid cube blocks only",
        "icon":        "▪",
    },
    {
        "id":          "survival_obtainable",
        "label":       "Survival Mode",
        "description": "No creative-only blocks",
        "icon":        "⛏",
    },
    {
        "id":          "natural",
        "label":       "Natural",
        "description": "Dirt, grass, stone, sand…",
        "icon":        "🌿",
    },
    {
        "id":          "stone_and_ores",
        "label":       "Stone & Ores",
        "description": "Stone variants and ore blocks",
        "icon":        "⛰",
    },
    {
        "id":          "wood_and_leaves",
        "label":       "Wood & Leaves",
        "description": "Log, plank, and leaf blocks",
        "icon":        "🌲",
    },
    {
        "id":          "coloured_blocks",
        "label":       "Coloured",
        "description": "Wool, concrete, terracotta…",
        "icon":        "🎨",
    },
]

CHIP_ACTIVE_STYLE = """
    QPushButton {
        background-color: #1e5c8a;
        color: #e8f4ff;
        border: 1px solid #5a9fd4;
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 11px;
        font-weight: bold;
        text-align: left;
    }
    QPushButton:hover { background-color: #2471aa; }
"""

CHIP_INACTIVE_STYLE = """
    QPushButton {
        background-color: #1a1a1a;
        color: #707070;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 11px;
        text-align: left;
    }
    QPushButton:hover {
        background-color: #222222;
        color: #a0a0a0;
        border-color: #3a3a3a;
    }
"""


class PaletteSelector(QWidget):
    """
    Grid of toggleable palette group chips.

    Signals:
        selection_changed(list): Emitted with the list of active group IDs
                                 whenever the selection changes.
    """

    selection_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active: set[str] = {"full_blocks", "survival_obtainable"}
        self._buttons: dict[str, QPushButton] = {}
        self._build_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_active_groups(self) -> list[str]:
        """Return list of currently active palette group IDs."""
        return list(self._active)

    def set_active_groups(self, group_ids: list[str]):
        """Programmatically set active groups."""
        self._active = set(group_ids)
        self._refresh_buttons()
        self._update_count()

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Section header
        header_row = QHBoxLayout()
        section_label = QLabel("BLOCK PALETTE")
        section_label.setStyleSheet(
            "color: #5a9fd4; font-size: 10px; font-weight: bold; "
            "letter-spacing: 2px;"
        )
        header_row.addWidget(section_label)
        header_row.addStretch()

        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #505050; font-size: 10px;")
        header_row.addWidget(self._count_label)
        layout.addLayout(header_row)

        # 2-column grid of chips
        grid = QGridLayout()
        grid.setSpacing(6)

        for i, group in enumerate(PALETTE_GROUPS):
            btn = QPushButton(f"{group['icon']}  {group['label']}")
            btn.setToolTip(group["description"])
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setCheckable(False)
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed
            )
            btn.setMinimumHeight(34)

            gid = group["id"]
            btn.clicked.connect(lambda _, g=gid: self._toggle(g))
            self._buttons[gid] = btn

            row, col = divmod(i, 2)
            grid.addWidget(btn, row, col)

        layout.addLayout(grid)

        # Helper note
        note = QLabel("Groups combine by union — more groups = more block choices.")
        note.setStyleSheet("color: #404040; font-size: 10px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        self._refresh_buttons()
        self._update_count()

    # ── Interaction ────────────────────────────────────────────────────────────

    def _toggle(self, group_id: str):
        if group_id in self._active:
            # Prevent deselecting all groups
            if len(self._active) > 1:
                self._active.discard(group_id)
        else:
            self._active.add(group_id)

        self._refresh_buttons()
        self._update_count()
        self.selection_changed.emit(list(self._active))

    def _refresh_buttons(self):
        for gid, btn in self._buttons.items():
            if gid in self._active:
                btn.setStyleSheet(CHIP_ACTIVE_STYLE)
            else:
                btn.setStyleSheet(CHIP_INACTIVE_STYLE)

    def _update_count(self):
        n = len(self._active)
        groups_word = "group" if n == 1 else "groups"
        self._count_label.setText(f"{n} {groups_word} active")