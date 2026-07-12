"""Pytest configuration and shared fixtures for API tests."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup test environment with temporary directories.

    Automatically applied to all tests. Overrides settings to use
    temporary directories instead of production paths.

    Args:
        tmp_path: Pytest's temporary directory fixture.
        monkeypatch: Pytest's monkeypatch fixture for patching.
    """
    # Patch the settings object to use temporary directories
    from app.core.config import settings

    # Create storage directory
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "database_url", str(storage_dir / "redubber.db"))
    monkeypatch.setattr(settings, "openai_api_key", "test-key-not-used")


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provide FastAPI test client for API endpoint testing.

    Creates a fresh application instance for each test to ensure
    isolation between tests. Uses temporary directories from
    setup_test_environment fixture.

    Yields:
        TestClient configured with the FastAPI app.
    """
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_project_data() -> dict[str, str]:
    """Provide sample project creation data.

    Returns:
        Dictionary with valid project creation fields.
    """
    return {
        "path": "/test/videos/project1",
    }


@pytest.fixture
def sample_task_data() -> dict[str, str | int]:
    """Provide sample task creation data.

    Returns:
        Dictionary with valid task submission fields.
    """
    return {
        "video_path": "/test/videos/sample.mp4",
        "project_id": 1,
    }


@pytest.fixture
def sample_voice_settings() -> dict[str, str]:
    """Provide sample voice settings data.

    Returns:
        Dictionary with valid voice configuration fields.
    """
    return {
        "voice": "nova",
        "instructions": "Speak with a clear, professional tone.",
    }
