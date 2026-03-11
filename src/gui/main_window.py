"""
src/gui/main_window.py

Root application window for VoxelForge. Wires all components together
and orchestrates the full generation pipeline on a background thread
so the UI stays responsive during API calls and voxelisation.

Layout:
  ┌─────────────────────────────────────────────────────────┐
  │  [Logo / Title bar]                                     │
  ├──────────────────────┬──────────────────────────────────┤
  │  Left panel          │  Centre: 3D Preview              │
  │  ─ PromptPanel       │                                  │
  │  ─ SettingsPanel     │                                  │
  │  ─ PaletteSelector   ├──────────────────────────────────┤
  │  ─ ExportPanel       │  Bottom: status bar              │
  └──────────────────────┴──────────────────────────────────┘
"""

import logging
import os
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QScrollArea, QFrame, QSizePolicy, QStatusBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QIcon

from src.gui.components.prompt_panel    import PromptPanel
from src.gui.components.settings_panel  import SettingsPanel
from src.gui.components.palette_selector import PaletteSelector
from src.gui.components.export_panel    import ExportPanel
from src.gui.components.preview_3d      import Preview3D

logger = logging.getLogger(__name__)


# ── Background worker ──────────────────────────────────────────────────────────

class GenerationWorker(QObject):
    """
    Runs the full pipeline (API → mesh → voxelise → colour map)
    on a background thread. Emits signals back to the UI thread.
    """

    progress    = pyqtSignal(str)          # Status message updates
    mesh_ready  = pyqtSignal(object)       # ProcessedMesh
    voxel_ready = pyqtSignal(object, object)  # VoxelGrid, colour_map ndarray
    finished    = pyqtSignal(object)       # MappedGrid
    error       = pyqtSignal(str)          # Error message

    def __init__(self, prompt: str, settings: dict, active_groups: list):
        super().__init__()
        self.prompt        = prompt
        self.settings      = settings
        self.active_groups = active_groups

    def run(self):
        try:
            self._run_pipeline()
        except Exception as e:
            logger.exception("Pipeline error")
            self.error.emit(str(e))

    def _run_pipeline(self):
        from src.api.trellis_client      import TrellisClient
        from src.voxeliser.mesh_processor import MeshProcessor
        from src.voxeliser.voxel_grid     import Voxeliser
        from src.voxeliser.colour_mapper  import ColourMapper
        from src.utils.config_loader      import ConfigLoader

        # ── Load API key ──────────────────────────────────────────────────────
        self.progress.emit("Loading configuration...")
        config = ConfigLoader()
        api_key = config.get_api_key()

        # ── Call Trellis API ──────────────────────────────────────────────────
        self.progress.emit("Connecting to NVIDIA Trellis API...")
        client   = TrellisClient(api_key=api_key)
        glb_bytes = client.generate_from_text(
            prompt             = self.prompt,
            seed               = self.settings["seed"],
            on_progress        = self.progress.emit,
        )

        # ── Process mesh ──────────────────────────────────────────────────────
        self.progress.emit("Processing 3D mesh...")
        processor      = MeshProcessor()
        processed_mesh = processor.load_glb(glb_bytes)
        self.mesh_ready.emit(processed_mesh)

        # ── Voxelise ──────────────────────────────────────────────────────────
        self.progress.emit("Voxelising mesh...")
        voxeliser  = Voxeliser(resolution=self.settings["voxel_resolution"])
        voxel_grid = voxeliser.voxelise(processed_mesh)

        if self.settings.get("hollow"):
            self.progress.emit("Hollowing structure...")
            voxel_grid = voxeliser.hollow(voxel_grid)

        self.voxel_ready.emit(voxel_grid, processed_mesh.colour_map)

        # ── Colour mapping ────────────────────────────────────────────────────
        self.progress.emit("Mapping colours to Minecraft blocks...")
        mapper     = ColourMapper(active_groups=self.active_groups)
        mapped     = mapper.map(voxel_grid, processed_mesh.colour_map)

        self.progress.emit("✓ Structure ready to export.")
        self.finished.emit(mapped)


class ExportWorker(QObject):
    """Runs file export on a background thread."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str)   # Output path
    error    = pyqtSignal(str)

    def __init__(self, mapped_grid, fmt: str, output_path: str):
        super().__init__()
        self.mapped_grid = mapped_grid
        self.fmt         = fmt
        self.output_path = output_path

    def run(self):
        try:
            self._export()
        except Exception as e:
            logger.exception("Export error")
            self.error.emit(str(e))

    def _export(self):
        from src.exporter.schematic_writer  import SchematicWriter
        from src.exporter.schem_writer      import SchemWriter
        from src.exporter.litematica_writer import LitematicaWriter

        self.progress.emit(f"Writing {self.fmt} file...")

        path = Path(self.output_path)

        if self.fmt == ".schematic":
            writer = SchematicWriter()
        elif self.fmt == ".schem":
            writer = SchemWriter()
        elif self.fmt == ".litematica":
            writer = LitematicaWriter()
        else:
            raise ValueError(f"Unknown export format: {self.fmt}")

        writer.write(self.mapped_grid, path)
        self.finished.emit(str(path))


# ── Main window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """
    VoxelForge main application window.
    """

    APP_TITLE   = "VoxelForge"
    MIN_WIDTH   = 1000
    MIN_HEIGHT  = 680

    def __init__(self):
        super().__init__()
        self._mapped_grid = None
        self._gen_thread  = None
        self._exp_thread  = None

        self._setup_window()
        self._build_ui()

    # ── Window setup ───────────────────────────────────────────────────────────

    def _setup_window(self):
        self.setWindowTitle(self.APP_TITLE)
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.resize(1280, 780)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #111111;
                color: #d0d0d0;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            }
            QScrollBar:vertical {
                background: #151515;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #2a2a2a;
                border-radius: 4px;
            }
            QStatusBar {
                background-color: #0d0d0d;
                color: #505050;
                font-size: 11px;
                border-top: 1px solid #1e1e1e;
            }
            QToolTip {
                background-color: #1e1e1e;
                color: #c0c0c0;
                border: 1px solid #3a3a3a;
                padding: 4px 8px;
                font-size: 11px;
            }
        """)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Title bar
        root_layout.addWidget(self._build_title_bar())

        # Main content area
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        # Left panel
        content.addWidget(self._build_left_panel(), stretch=0)

        # Vertical divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setStyleSheet("color: #1e1e1e;")
        content.addWidget(divider)

        # 3D preview (takes remaining space)
        content.addWidget(self._build_preview_area(), stretch=1)

        root_layout.addLayout(content, stretch=1)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready. Enter a prompt and click Generate.")

    def _build_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(44)
        bar.setStyleSheet(
            "background-color: #0d0d0d; border-bottom: 1px solid #1e1e1e;"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)

        # Logo / wordmark
        logo = QLabel("◈  VOXELFORGE")
        logo.setStyleSheet(
            "color: #5a9fd4; font-size: 14px; font-weight: bold; "
            "letter-spacing: 3px;"
        )
        layout.addWidget(logo)
        layout.addStretch()

        # Version tag
        ver = QLabel("v1.0.0")
        ver.setStyleSheet("color: #303030; font-size: 11px;")
        layout.addWidget(ver)

        return bar

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        container.setFixedWidth(300)
        container.setStyleSheet("background-color: #131313;")

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(300)
        scroll.setStyleSheet("border: none; background-color: #131313;")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(20)

        # Components
        self._prompt_panel = PromptPanel()
        self._prompt_panel.generate_requested.connect(self._on_generate)
        layout.addWidget(self._prompt_panel)

        layout.addWidget(self._h_divider())

        self._settings_panel = SettingsPanel()
        layout.addWidget(self._settings_panel)

        layout.addWidget(self._h_divider())

        self._palette_selector = PaletteSelector()
        layout.addWidget(self._palette_selector)

        layout.addWidget(self._h_divider())

        self._export_panel = ExportPanel()
        self._export_panel.export_requested.connect(self._on_export)
        layout.addWidget(self._export_panel)

        layout.addStretch()

        return scroll

    def _build_preview_area(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background-color: #0e0e0e;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Preview mode label
        self._preview_label = QLabel("NO PREVIEW")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setFixedHeight(28)
        self._preview_label.setStyleSheet(
            "color: #2a2a2a; font-size: 10px; letter-spacing: 2px; "
            "background-color: #0d0d0d; border-bottom: 1px solid #1a1a1a;"
        )
        layout.addWidget(self._preview_label)

        self._preview = Preview3D()
        layout.addWidget(self._preview, stretch=1)

        # Preview hint
        hint = QLabel(
            "Left-drag: orbit   ·   Right-drag: pan   ·   Scroll: zoom"
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setFixedHeight(24)
        hint.setStyleSheet(
            "color: #252525; font-size: 10px; "
            "background-color: #0d0d0d; border-top: 1px solid #1a1a1a;"
        )
        layout.addWidget(hint)

        return container

    # ── Pipeline orchestration ─────────────────────────────────────────────────

    def _on_generate(self, prompt: str):
        if self._gen_thread and self._gen_thread.isRunning():
            return

        settings      = self._settings_panel.get_settings()
        active_groups = self._palette_selector.get_active_groups()

        self._prompt_panel.set_enabled(False)
        self._export_panel.set_ready(False)
        self._mapped_grid = None
        self._preview.clear()
        self._set_preview_label("GENERATING MESH...", "#5a9fd4")

        worker = GenerationWorker(prompt, settings, active_groups)
        thread = QThread()
        worker.moveToThread(thread)

        worker.progress.connect(self._on_pipeline_progress)
        worker.mesh_ready.connect(self._on_mesh_ready)
        worker.voxel_ready.connect(self._on_voxel_ready)
        worker.finished.connect(self._on_generation_done)
        worker.error.connect(self._on_pipeline_error)
        thread.started.connect(worker.run)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_gen_thread_finished)

        self._gen_thread  = thread
        self._gen_worker  = worker   # keep reference alive until thread ends
        thread.start()

    def _on_pipeline_progress(self, message: str):
        self._prompt_panel.set_status(message, "#a0c8e8")
        self._status_bar.showMessage(message)

    def _on_mesh_ready(self, processed_mesh):
        self._preview.load_mesh(processed_mesh)
        self._set_preview_label("MESH PREVIEW", "#5a9fd4")

    def _on_voxel_ready(self, voxel_grid, colour_map):
        self._preview.load_voxels(voxel_grid, colour_map)
        self._set_preview_label("VOXEL PREVIEW", "#4aaa6a")

    def _on_generation_done(self, mapped_grid):
        self._mapped_grid = mapped_grid
        self._prompt_panel.set_enabled(True)
        self._prompt_panel.set_status("✓ Structure ready to export.", "#4aaa6a")
        self._export_panel.set_ready(True)
        self._export_panel.set_status("Ready to export.", "#4aaa6a")
        self._status_bar.showMessage(
            f"Generation complete. "
            f"{mapped_grid.occupied.sum():,} voxels mapped."
        )

    def _on_pipeline_error(self, message: str):
        self._prompt_panel.set_enabled(True)
        self._prompt_panel.set_status(f"⚠ Error: {message}", "#e85050")
        self._set_preview_label("ERROR", "#e85050")
        self._status_bar.showMessage(f"Error: {message}")
        logger.error(f"Pipeline error: {message}")

    def _on_export(self, fmt: str, output_path: str):
        if not self._mapped_grid:
            return
        if self._exp_thread and self._exp_thread.isRunning():
            return

        self._export_panel.set_status("Exporting...", "#a0c8e8")
        self._status_bar.showMessage(f"Writing {fmt} file...")

        worker = ExportWorker(self._mapped_grid, fmt, output_path)
        thread = QThread()
        worker.moveToThread(thread)

        worker.progress.connect(
            lambda msg: self._status_bar.showMessage(msg)
        )
        worker.finished.connect(self._on_export_done)
        worker.error.connect(self._on_export_error)
        thread.started.connect(worker.run)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_exp_thread_finished)

        self._exp_thread = thread
        self._exp_worker = worker
        thread.start()

    def _on_export_done(self, output_path: str):
        self._export_panel.set_status(
            f"✓ Exported: {Path(output_path).name}", "#4aaa6a"
        )
        self._status_bar.showMessage(
            f"Export complete: {output_path}"
        )

    def _on_export_error(self, message: str):
        self._export_panel.set_status(f"⚠ Export failed: {message}", "#e85050")
        self._status_bar.showMessage(f"Export error: {message}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _on_gen_thread_finished(self):
        """Null out gen thread/worker references so the next generation can start."""
        self._gen_thread = None
        self._gen_worker = None

    def _on_exp_thread_finished(self):
        """Null out export thread/worker references so the next export can start."""
        self._exp_thread = None
        self._exp_worker = None

    def _set_preview_label(self, text: str, colour: str = "#2a2a2a"):
        self._preview_label.setText(text)
        self._preview_label.setStyleSheet(
            f"color: {colour}; font-size: 10px; letter-spacing: 2px; "
            f"background-color: #0d0d0d; border-bottom: 1px solid #1a1a1a;"
        )

    def _h_divider(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #1e1e1e;")
        return line