"""
src/gui/components/prompt_panel.py

The primary user input panel. Contains the text prompt field,
a generate button, and a status label for live feedback.
Emits a signal when the user triggers generation.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QPushButton, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont, QColor


class PromptPanel(QWidget):
    """
    Top panel: prompt text input + Generate button.

    Signals:
        generate_requested(str): Emitted with the prompt text when the
                                 user clicks Generate or presses Ctrl+Enter.
    """

    generate_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_status(self, message: str, colour: str = "#a0a0a0"):
        """Update the status label below the prompt box."""
        self._status_label.setText(message)
        self._status_label.setStyleSheet(f"color: {colour}; font-size: 11px;")

    def set_enabled(self, enabled: bool):
        """Enable or disable the panel during generation."""
        self._prompt_edit.setEnabled(enabled)
        self._generate_btn.setEnabled(enabled)
        if not enabled:
            self._generate_btn.setText("⏳  Generating...")
        else:
            self._generate_btn.setText("⚡  Generate Structure")

    def get_prompt(self) -> str:
        return self._prompt_edit.toPlainText().strip()

    def clear_prompt(self):
        self._prompt_edit.clear()

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Section label
        section_label = QLabel("PROMPT")
        section_label.setStyleSheet(
            "color: #5a9fd4; font-size: 10px; font-weight: bold; "
            "letter-spacing: 2px;"
        )
        layout.addWidget(section_label)

        # Prompt text edit
        self._prompt_edit = QTextEdit()
        self._prompt_edit.setPlaceholderText(
            "Describe a Minecraft structure...\n\n"
            "e.g. \"A medieval stone castle with towers and a drawbridge\"\n"
            "     \"A futuristic space station with solar panels\"\n"
            "     \"A cozy hobbit hole built into a grassy hillside\""
        )
        self._prompt_edit.setMinimumHeight(110)
        self._prompt_edit.setMaximumHeight(160)
        self._prompt_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._prompt_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #e8e8e8;
                border: 1px solid #2e2e2e;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.5;
            }
            QTextEdit:focus {
                border: 1px solid #5a9fd4;
            }
        """)
        # Ctrl+Enter triggers generation
        self._prompt_edit.installEventFilter(self)
        layout.addWidget(self._prompt_edit)

        # Bottom row: status label + generate button
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)

        self._status_label = QLabel("Ready.")
        self._status_label.setStyleSheet("color: #606060; font-size: 11px;")
        bottom_row.addWidget(self._status_label, stretch=1)

        self._generate_btn = QPushButton("⚡  Generate Structure")
        self._generate_btn.setMinimumWidth(180)
        self._generate_btn.setMinimumHeight(36)
        self._generate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e5c8a;
                color: #e8f4ff;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-size: 13px;
                font-weight: bold;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: #2471aa;
            }
            QPushButton:pressed {
                background-color: #155080;
            }
            QPushButton:disabled {
                background-color: #1a3a52;
                color: #4a7a9b;
            }
        """)
        self._generate_btn.clicked.connect(self._on_generate_clicked)
        bottom_row.addWidget(self._generate_btn)

        layout.addLayout(bottom_row)

    # ── Event handling ─────────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        """Intercept Ctrl+Enter in the prompt box to trigger generation."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt

        if obj is self._prompt_edit and event.type() == QEvent.Type.KeyPress:
            key_event = event
            if (key_event.key() == Qt.Key.Key_Return and
                    key_event.modifiers() == Qt.KeyboardModifier.ControlModifier):
                self._on_generate_clicked()
                return True
        return super().eventFilter(obj, event)

    def _on_generate_clicked(self):
        prompt = self.get_prompt()
        if prompt:
            self.generate_requested.emit(prompt)
        else:
            self.set_status("⚠  Please enter a prompt first.", "#e8a44a")