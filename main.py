"""
main.py — VoxelForge entry point

Boots the PyQt6 application, applies the qt-material theme, and
launches the main window. All heavy work happens on background threads
inside MainWindow — this file stays deliberately thin.

Usage:
    python main.py
    python main.py --debug     # enables DEBUG console logging
"""

import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        prog="VoxelForge",
        description="AI-powered text-to-Minecraft structure generator.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level console logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Logging (must be first — before any other imports) ─────────────────
    from src.utils.logger import setup_logging
    setup_logging(debug=args.debug)

    import logging
    logger = logging.getLogger(__name__)
    logger.info("VoxelForge starting up.")

    # ── 2. Config ─────────────────────────────────────────────────────────────
    from src.utils.config_loader import ConfigLoader
    config = ConfigLoader()

    # ── 3. Qt application ─────────────────────────────────────────────────────
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("VoxelForge")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("VoxelForge")

    # ── 4. Apply qt-material theme ────────────────────────────────────────────
    try:
        from qt_material import apply_stylesheet
        theme      = config.get_theme()
        theme_file = f"{theme}.xml"
        apply_stylesheet(app, theme=theme_file)
        logger.info(f"Applied theme: {theme_file}")
    except Exception as e:
        logger.warning(
            f"Could not apply qt-material theme: {e}. "
            "Falling back to default Qt style."
        )

    # ── 5. Launch main window ─────────────────────────────────────────────────
    from src.gui.main_window import MainWindow
    window = MainWindow()
    window.show()

    logger.info("Main window displayed. Entering event loop.")
    exit_code = app.exec()

    logger.info(f"VoxelForge exiting with code {exit_code}.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()