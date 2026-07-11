"""Dependency injection providers for FastAPI routes."""

from __future__ import annotations

from database import DatabaseManager
from file_scanner import FileScanner

from app.core.config import settings


def get_db() -> DatabaseManager:
    """Provide DatabaseManager instance.

    Returns:
        Initialized DatabaseManager connected to the configured database.
    """
    return DatabaseManager(settings.database_url)


def get_scanner() -> FileScanner:
    """Provide FileScanner instance.

    Returns:
        Initialized FileScanner for detecting video and subtitle files.
    """
    return FileScanner()
