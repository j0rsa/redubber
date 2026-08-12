"""Application configuration using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


@lru_cache(maxsize=1)
def get_package_version() -> str:
    """Return the app version from ``pyproject.toml`` (single source of truth).

    Prefer reading ``pyproject.toml`` so bumps apply without reinstalling the
    package (Docker also uses ``poetry install --no-root``). Fall back to
    importlib metadata when the file is unavailable.
    """
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return str(data["tool"]["poetry"]["version"])
    except Exception:
        pass

    try:
        return version("redubber")
    except PackageNotFoundError:
        return "0.0.0"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # Config directory — redubber.db and settings.json are stored here
    redubber_config_path: str = ""

    # OpenAI
    openai_api_key: str = ""

    # Task Queue
    max_concurrent_redubs: int = 1
    task_queue_max_size: int = 100

    # API
    api_title: str = "Redubber API"
    # Populated from pyproject.toml — do not hardcode a second copy here.
    api_version: str = get_package_version()
    log_level: str = "INFO"

    # CORS (comma-separated origins)
    cors_origins: str = "http://localhost:5173,http://localhost:5174,http://localhost:4173"

    @property
    def database_url(self) -> str:
        if self.redubber_config_path:
            return str(Path(self.redubber_config_path) / "redubber.db")
        return "./redubber.db"


# Global settings instance
settings = Settings()
