"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # Database
    database_url: str = "./redubber.db"

    # OpenAI
    openai_api_key: str = ""

    # Task Queue
    max_concurrent_redubs: int = 1
    task_queue_max_size: int = 100

    # API
    api_title: str = "Redubber API"
    api_version: str = "2.0.1"
    log_level: str = "INFO"

    # CORS (comma-separated origins)
    cors_origins: str = "http://localhost:5173,http://localhost:5174,http://localhost:4173"



# Global settings instance
settings = Settings()
