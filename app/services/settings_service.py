"""Service for reading and persisting tool-level application settings."""

from __future__ import annotations

import os
from pathlib import Path

from app.schemas.settings import Settings, SettingsResponse, SettingsUpdate


def _get_db():
    from app.core.config import settings as _config
    from database import DatabaseManager
    return DatabaseManager(_config.database_url)


def _mask_api_key(raw_key: str) -> str:
    if not raw_key:
        return ""
    last4 = raw_key[-4:]
    return f"sk-...{last4}"


def _to_response(settings: Settings) -> SettingsResponse:
    data = settings.model_dump()
    data["openai_api_key"] = _mask_api_key(settings.openai_api_key)
    return SettingsResponse.model_validate(data)


def _load_raw() -> Settings:
    """Load settings from DB, applying env-var fallbacks for workspace fields."""
    db = _get_db()
    row = db.get_app_settings()

    if row:
        # Remove DB metadata fields before validating
        row.pop("id", None)
        row.pop("updated_at", None)
        settings = Settings.model_validate(row)
    else:
        settings = Settings()

    # Env-var fallbacks for workspace fields (empty in DB → env var takes over)
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


def _persist(settings: Settings) -> None:
    """Persist settings to the database."""
    db = _get_db()
    db.save_app_settings(settings.model_dump())


def get_openai_api_key() -> str:
    """Return the OpenAI API key — DB value takes priority, env var as fallback."""
    raw = _load_raw()
    if raw.openai_api_key:
        return raw.openai_api_key
    return os.environ.get("OPENAI_API_KEY", "")


def get_settings() -> SettingsResponse:
    """Load current settings from DB and return with the API key masked."""
    raw = _load_raw()
    return _to_response(raw)


def update_settings(update: SettingsUpdate) -> SettingsResponse:
    """Merge *update* into current settings, persist to DB, and return result.

    Raises:
        ValueError: If ``working_directory`` is set to a non-empty path that
            does not exist on the filesystem.
    """
    raw = _load_raw()
    current = raw.model_dump()

    patch = update.model_dump(exclude_none=True)
    current.update(patch)

    updated = Settings.model_validate(current)

    if updated.working_directory:
        wd = Path(updated.working_directory)
        if not wd.exists():
            raise ValueError(
                f"working_directory does not exist on disk: {updated.working_directory}"
            )

    _persist(updated)
    return _to_response(updated)
