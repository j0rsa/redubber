"""Integration test fixtures for end-to-end workflow tests.

Provides test database, video files, and FastAPI client with full lifespan.
"""

from __future__ import annotations


from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import create_app
from database import DatabaseManager


@pytest.fixture(scope="session")
def integration_test_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create persistent test directory for integration tests.

    Args:
        tmp_path_factory: Pytest's temp path factory for session-scoped directories.

    Returns:
        Path to the integration test directory.
    """
    return tmp_path_factory.mktemp("integration")


@pytest.fixture(scope="session")
def integration_db(
    integration_test_dir: Path,
) -> Generator[DatabaseManager, None, None]:
    """Create test database for integration tests.

    Session-scoped to allow tests to build on each other's state when needed.

    Args:
        integration_test_dir: Base directory for integration tests.

    Yields:
        DatabaseManager instance configured for testing.
    """
    db_path = integration_test_dir / "test.db"
    db = DatabaseManager(str(db_path))
    yield db
    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def test_video_dir(integration_test_dir: Path) -> Path:
    """Create directory with test video files.

    Args:
        integration_test_dir: Base directory for integration tests.

    Returns:
        Path to the test videos directory with sample files.
    """
    video_dir = integration_test_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy video files (or copy small test videos if available)
    test_videos = [
        "test_en.mp4",
        "test_fr.mp4",
        "test_de.mp4",
        "sample.mp4",
    ]

    for video_name in test_videos:
        video_path = video_dir / video_name
        # Create a minimal fake video file for testing
        video_path.write_bytes(b"fake video content for testing")

    return video_dir


@pytest.fixture(scope="session")
def test_subtitle_dir(integration_test_dir: Path) -> Path:
    """Create directory with test subtitle files.

    Args:
        integration_test_dir: Base directory for integration tests.

    Returns:
        Path to the test subtitles directory.
    """
    subtitle_dir = integration_test_dir / "subtitles"
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy subtitle files
    test_subtitles = [
        "test_en.srt",
        "test_fr.srt",
        "sample.vtt",
    ]

    for subtitle_name in test_subtitles:
        subtitle_path = subtitle_dir / subtitle_name
        subtitle_path.write_text(
            """1
00:00:00,000 --> 00:00:05,000
Test subtitle content

2
00:00:05,000 --> 00:00:10,000
More test content
"""
        )

    return subtitle_dir


@pytest.fixture
def integration_client(
    integration_test_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    """FastAPI test client with lifespan for integration tests.

    Creates a fresh application instance with isolated temporary directories.

    Args:
        integration_test_dir: Base directory for integration tests.
        monkeypatch: Pytest's monkeypatch fixture for patching.

    Yields:
        TestClient configured with the FastAPI app and lifespan.
    """
    # Create storage directory
    storage_dir = integration_test_dir / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Patch settings to use test directories
    monkeypatch.setattr(settings, "database_url", str(storage_dir / "test_redubber.db"))
    monkeypatch.setattr(settings, "openai_api_key", "test-key-integration")
    monkeypatch.setattr(settings, "max_concurrent_redubs", 2)
    monkeypatch.setattr(settings, "task_queue_max_size", 10)

    # Create app and test client
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client



@pytest.fixture
def sample_project_path(test_video_dir: Path) -> str:
    """Provide path to sample project for testing.

    Args:
        test_video_dir: Directory containing test video files.

    Returns:
        String path to the test project directory.
    """
    return str(test_video_dir)


@pytest.fixture
def sample_video_path(test_video_dir: Path) -> str:
    """Provide path to sample video file for testing.

    Args:
        test_video_dir: Directory containing test video files.

    Returns:
        String path to a test video file.
    """
    return str(test_video_dir / "sample.mp4")
