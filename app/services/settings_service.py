"""Service for reading and persisting tool-level application settings."""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.schemas.settings import Settings, SettingsResponse, SettingsUpdate

# Environment variable that overrides the default settings file location
_SETTINGS_PATH_ENV = "REDUBBER_SETTINGS_PATH"
_DEFAULT_SETTINGS_FILENAME = "settings.json"


def get_settings_path() -> Path:
    """Return the path to the settings JSON file.

    Respects the ``REDUBBER_SETTINGS_PATH`` environment variable.
    Falls back to ``./settings.json`` relative to the process working directory.

    Returns:
        Absolute or relative path to the settings file.
    """
    env_override = os.environ.get(_SETTINGS_PATH_ENV)
    if env_override:
        return Path(env_override)
    return Path(_DEFAULT_SETTINGS_FILENAME)


def _mask_api_key(raw_key: str) -> str:
    """Return a masked representation of an OpenAI API key.

    Args:
        raw_key: The full API key value as stored on disk.

    Returns:
        Empty string if *raw_key* is empty, otherwise ``"sk-...{last4}"``.
    """
    if not raw_key:
        return ""
    last4 = raw_key[-4:]
    return f"sk-...{last4}"


def _load_raw() -> Settings:
    """Load settings from disk, returning defaults if the file is missing or invalid.

    The ``working_directory`` field follows this priority:
    1. Non-empty value saved in settings.json  (explicit user override)
    2. ``REDUBBER_WORKING_DIR`` environment variable
    3. Empty string (app working dir fallback)

    Returns:
        A ``Settings`` instance populated from the JSON file or defaults.
    """
    path = get_settings_path()
    if not path.exists():
        return Settings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        settings = Settings.model_validate(data)
        # Env-var fallbacks for workspace fields (saved empty string → env var takes over)
        updates: dict[str, str] = {}
        if not settings.projects_root_path:
            env_root = os.environ.get("REDUBBER_PROJECTS_ROOT", "")
            if env_root:
                updates["projects_root_path"] = env_root
        if not settings.working_directory:
            env_wd = os.environ.get("REDUBBER_WORKING_DIR", "")
            if env_wd:
                updates["working_directory"] = env_wd
        if updates:
            settings = settings.model_copy(update=updates)
        return settings
    except (json.JSONDecodeError, ValueError):
        return Settings()


def _persist(settings: Settings) -> None:
    """Persist settings to disk as JSON.

    Creates parent directories if they don't exist.

    Args:
        settings: The settings instance to write.
    """
    path = get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        settings.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _to_response(settings: Settings) -> SettingsResponse:
    """Build a ``SettingsResponse`` with the API key masked.

    Args:
        settings: Internal settings (may contain full API key).

    Returns:
        Response-safe settings with the key masked.
    """
    data = settings.model_dump()
    data["openai_api_key"] = _mask_api_key(settings.openai_api_key)
    return SettingsResponse.model_validate(data)


def get_settings() -> SettingsResponse:
    """Load current settings from disk and return them with the API key masked.

    Returns:
        Current settings. Defaults are returned if no file exists yet.
    """
    raw = _load_raw()
    return _to_response(raw)


def update_settings(update: SettingsUpdate) -> SettingsResponse:
    """Merge *update* into current settings, persist, and return the result.

    Only fields that are explicitly set (non-``None``) in *update* are
    changed; all other fields retain their current values.

    Args:
        update: Partial settings update containing the fields to change.

    Returns:
        Updated settings with the API key masked.

    Raises:
        ValueError: If ``working_directory`` is set to a non-empty path that
            does not exist on the filesystem.
    """
    raw = _load_raw()
    current = raw.model_dump()

    # Apply only the fields that were explicitly provided
    patch = update.model_dump(exclude_none=True)
    current.update(patch)

    updated = Settings.model_validate(current)

    # Validate working_directory: if set, the path must exist on disk
    if updated.working_directory:
        wd = Path(updated.working_directory)
        if not wd.exists():
            raise ValueError(
                f"working_directory does not exist on disk: {updated.working_directory}"
            )

    _persist(updated)
    return _to_response(updated)
