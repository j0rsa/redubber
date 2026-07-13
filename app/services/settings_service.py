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
    data["env_overrides"] = list(get_env_overrides().keys())
    return SettingsResponse.model_validate(data)


def get_env_overrides() -> dict[str, str]:
    """Return env vars that override settings fields, keyed by field name.

    When an env var is set it always wins over the DB value, and the UI
    should show that field as read-only.
    """
    overrides: dict[str, str] = {}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        overrides["openai_api_key"] = api_key
    working_dir = os.environ.get("REDUBBER_WORKING_DIR", "")
    if working_dir:
        overrides["working_directory"] = working_dir
    projects_root = os.environ.get("REDUBBER_PROJECTS_ROOT", "")
    if projects_root:
        overrides["projects_root_path"] = projects_root
    return overrides


def _load_raw() -> Settings:
    """Load settings from DB, with env vars winning over DB values."""
    db = _get_db()
    row = db.get_app_settings()

    if row:
        row.pop("id", None)
        row.pop("updated_at", None)
        settings = Settings.model_validate(row)
    else:
        settings = Settings()

    # Env vars always win — they represent deployment-level configuration
    # that should not be overridden via the UI.
    overrides = get_env_overrides()
    if overrides:
        settings = settings.model_copy(update=overrides)

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
    # Strip fields that are locked by env vars — they cannot be changed via the UI
    env_locked = set(get_env_overrides().keys())
    patch = {k: v for k, v in patch.items() if k not in env_locked}
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
