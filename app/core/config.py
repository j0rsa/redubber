"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # Database
    database_url: str = "./redubber.db"

    # Storage
    mounted_storage: str = "./storage"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_voice: str = "nova"

    # Task Queue
    max_concurrent_redubs: int = 1
    task_queue_max_size: int = 100
    tts_max_concurrent: int = 100  # Async TTS concurrency

    # API
    api_title: str = "Redubber API"
    api_version: str = "2.0.0"
    log_level: str = "INFO"

    # CORS (comma-separated origins)
    cors_origins: str = "http://localhost:5173,http://localhost:5174,http://localhost:4173"

    @property
    def storage_path(self) -> Path:
        """Get Path object for mounted storage."""
        return Path(self.mounted_storage)


# Global settings instance
settings = Settings()
