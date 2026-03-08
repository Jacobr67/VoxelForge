"""
src/utils/config_loader.py

Loads application settings from config/settings.yaml and the NVIDIA
API key from either a .env file or an environment variable.

Priority for API key resolution:
  1. Environment variable: NVIDIA_API_KEY
  2. .env file in project root: NVIDIA_API_KEY=nvapi-...

Settings are loaded once and cached. Call ConfigLoader() anywhere;
it reads from disk on first instantiation per process.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Paths relative to project root (two levels up from this file)
_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_ENV_FILE      = _PROJECT_ROOT / ".env"
_SETTINGS_FILE = _PROJECT_ROOT / "config" / "settings.yaml"

# Defaults used when settings.yaml is absent or a key is missing
_DEFAULTS: dict[str, Any] = {
    "voxel_resolution":   64,
    "texture_resolution": 1024,
    "geometry_fidelity":  0.5,
    "surface_fidelity":   0.5,
    "seed":               0,
    "hollow":             False,
    "log_level":          "WARNING",
    "theme":              "dark_teal",
}


class ConfigLoader:
    """
    Loads and provides access to all application configuration.

    Usage:
        config = ConfigLoader()
        api_key  = config.get_api_key()
        settings = config.get_settings()
        res      = config.get("voxel_resolution", default=64)
    """

    _instance_settings: dict | None = None   # class-level cache

    def __init__(self):
        # Load .env file once (safe to call multiple times — dotenv is idempotent)
        load_dotenv(dotenv_path=_ENV_FILE, override=False)

        if ConfigLoader._instance_settings is None:
            ConfigLoader._instance_settings = self._load_settings()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_api_key(self) -> str:
        """
        Retrieve the NVIDIA NGC API key.

        Checks environment variable first, then .env file (loaded at init).

        Returns:
            The API key string (starts with 'nvapi-').

        Raises:
            EnvironmentError: If no API key can be found.
        """
        key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if key:
            logger.debug("API key loaded from environment.")
            return key

        raise EnvironmentError(
            "NVIDIA API key not found.\n\n"
            "Add it to your .env file in the project root:\n"
            "  NVIDIA_API_KEY=nvapi-your-key-here\n\n"
            "Get your key at: https://build.nvidia.com/microsoft/trellis"
        )

    def get_settings(self) -> dict:
        """Return the full settings dict (merged defaults + settings.yaml)."""
        return dict(ConfigLoader._instance_settings)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a single setting value by key."""
        return ConfigLoader._instance_settings.get(key, default)

    def get_theme(self) -> str:
        """Return the qt-material theme name, e.g. 'dark_teal'."""
        return self.get("theme", "dark_teal")

    def is_debug(self) -> bool:
        """True if log_level is set to DEBUG in settings.yaml."""
        return self.get("log_level", "WARNING").upper() == "DEBUG"

    # ── Private ────────────────────────────────────────────────────────────────

    def _load_settings(self) -> dict:
        """
        Load settings.yaml and merge with defaults.
        Missing keys fall back to _DEFAULTS silently.
        """
        settings = dict(_DEFAULTS)

        if not _SETTINGS_FILE.exists():
            logger.warning(
                f"settings.yaml not found at {_SETTINGS_FILE}. "
                "Using all defaults."
            )
            return settings

        try:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}

            for key in _DEFAULTS:
                if key in loaded:
                    settings[key] = loaded[key]

            logger.info(f"Settings loaded from {_SETTINGS_FILE}")

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse settings.yaml: {e}. Using defaults.")
        except Exception as e:
            logger.error(f"Failed to read settings.yaml: {e}. Using defaults.")

        return settings