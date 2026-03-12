"""
main.py — VoxelForge entry point

Boots the PyQt6 application, applies the qt-material theme, and
launches the main window with a branded splash screen.

Usage:
    python main.py
    python main.py --debug         # enables DEBUG console logging
    python main.py --debugsplash   # show splash only (UI debug mode)
"""

import sys
import argparse
from src.gui.splash_screen import SplashScreen


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

    parser.add_argument(
        "--debugsplash",
        action="store_true",
        help="Launch only the splash screen for UI debugging.",
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
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)
    app.setApplicationName("VoxelForge")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("VoxelForge")

    # ── 4. Apply qt-material theme ────────────────────────────────────────────
    try:
        from qt_material import apply_stylesheet
        theme = config.get_theme()
        theme_file = f"{theme}.xml"
        apply_stylesheet(app, theme=theme_file)
        logger.info(f"Applied theme: {theme_file}")
    except Exception as e:
        logger.warning(
            f"Could not apply qt-material theme: {e}. "
            "Falling back to default Qt style."
        )

    # ── 5. Show splash screen ─────────────────────────────────────────────────
    splash = SplashScreen()
    splash.show()

    # Force Qt to draw splash immediately
    app.processEvents()

    # ── DEBUG SPLASH MODE ─────────────────────────────────────────────────────
    if args.debugsplash:
        logger.info("Running in splash debug mode.")
        logger.info("Splash will stay open until Ctrl+C is pressed.")
        exit_code = app.exec()
        sys.exit(exit_code)

    # ── 6. Create main window ─────────────────────────────────────────────────
    from src.gui.main_window import MainWindow
    window = MainWindow()

    # Function to finish startup
    def finish_startup():
        splash.close()
        window.show()
        logger.info("Main window displayed. Entering event loop.")

    # Give splash a small visible duration for polish
    QTimer.singleShot(1500, finish_startup)

    exit_code = app.exec()

    logger.info(f"VoxelForge exiting with code {exit_code}.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()