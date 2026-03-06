"""
src/utils/logger.py

Configures app-wide logging for VoxelForge.
Call setup_logging() once at startup in main.py.

Log output:
  - Console: WARNING and above (clean for end users)
  - File:    DEBUG and above → logs/voxelforge.log (full detail for debugging)

Log file rotates at 2 MB, keeping the last 3 files so it never
grows unbounded on a user's machine.
"""

import logging
import logging.handlers
from pathlib import Path


LOG_DIR      = Path("logs")
LOG_FILE     = LOG_DIR / "voxelforge.log"
MAX_BYTES    = 2 * 1024 * 1024   # 2 MB
BACKUP_COUNT = 3

LOG_FORMAT_FILE    = "%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s"
LOG_FORMAT_CONSOLE = "%(levelname)s: %(message)s"
DATE_FORMAT        = "%Y-%m-%d %H:%M:%S"


def setup_logging(debug: bool = False):
    """
    Initialise logging for the entire application.

    Args:
        debug: If True, sets console output to DEBUG level as well.
               Useful during development; never set True in a release build.

    Call this once at the very top of main.py before importing anything else.
    """
    LOG_DIR.mkdir(exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)   # capture everything at root

    # ── File handler (rotating) ───────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes    = MAX_BYTES,
        backupCount = BACKUP_COUNT,
        encoding    = "utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(LOG_FORMAT_FILE, datefmt=DATE_FORMAT)
    )
    root_logger.addHandler(file_handler)

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.WARNING)
    console_handler.setFormatter(
        logging.Formatter(LOG_FORMAT_CONSOLE)
    )
    root_logger.addHandler(console_handler)

    # Silence noisy third-party loggers at WARNING unless in debug mode
    if not debug:
        for noisy in ("trimesh", "PIL", "OpenGL", "urllib3", "requests"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        f"Logging initialised. Log file: {LOG_FILE.resolve()}"
    )