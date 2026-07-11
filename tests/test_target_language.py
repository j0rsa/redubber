"""Tests for per-project target language settings.

Covers:
- DatabaseManager.get_target_language / set_target_language
- ProjectResponse.target_language field and default
- PUT /api/projects/{id}/target-language endpoint
- GET /api/projects/{id} returns target_language
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from database import DatabaseManager


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """Provide a fresh FastAPI test client for each test.

    The autouse ``setup_test_environment`` fixture in conftest.py already
    patches settings to use a temp database, so every call here gets a
    clean isolated db.
    """
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def project_id(client: TestClient, tmp_path: Path) -> int:
    """Create a real project directory and return the created project ID."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    response = client.post("/api/projects/", json={"path": str(project_dir)})
    assert response.status_code == 201, response.text
    return response.json()["id"]


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------


class TestDatabaseTargetLanguage:
    """Unit tests for DatabaseManager target language methods."""

    def test_default_target_language_is_eng(self, tmp_path: Path) -> None:
        """New projects should default to 'eng' when no value is stored."""
        db = DatabaseManager(str(tmp_path / "test.db"))
        project_id = db.add_project(path=str(tmp_path / "project"), name="test")

        assert db.get_target_language(project_id) == "eng"

    def test_set_and_get_target_language(self, tmp_path: Path) -> None:
        """set_target_language should persist and be readable."""
        db = DatabaseManager(str(tmp_path / "test.db"))
        project_id = db.add_project(path=str(tmp_path / "project"), name="test")

        db.set_target_language(project_id, "spa")

        assert db.get_target_language(project_id) == "spa"

    def test_set_target_language_overwrites_previous(self, tmp_path: Path) -> None:
        """Calling set_target_language twice should keep the latest value."""
        db = DatabaseManager(str(tmp_path / "test.db"))
        project_id = db.add_project(path=str(tmp_path / "project"), name="test")

        db.set_target_language(project_id, "fra")
        db.set_target_language(project_id, "deu")

        assert db.get_target_language(project_id) == "deu"

    def test_migration_adds_column_to_existing_db(self, tmp_path: Path) -> None:
        """DatabaseManager should add target_language column on first init of a legacy db."""
        import sqlite3

        db_path = str(tmp_path / "legacy.db")

        # Bootstrap a minimal database without the target_language column
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE projects (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "path TEXT UNIQUE NOT NULL, name TEXT NOT NULL, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "INSERT INTO projects (path, name) VALUES ('/test', 'test')"
            )
            conn.commit()

        # Instantiating DatabaseManager should run the migration silently
        db = DatabaseManager(db_path)

        # Column must now exist and have the default value
        assert db.get_target_language(1) == "eng"


# ---------------------------------------------------------------------------
# Schema layer
# ---------------------------------------------------------------------------


class TestProjectResponseSchema:
    """Unit tests for the ProjectResponse Pydantic model."""

    def test_target_language_field_defaults_to_eng(self) -> None:
        """ProjectResponse.target_language should default to 'eng'."""
        from app.schemas.models import ProjectResponse
        from datetime import datetime

        pr = ProjectResponse(
            id=1,
            path="/some/path",
            name="project",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert pr.target_language == "eng"

    def test_target_language_field_accepts_custom_value(self) -> None:
        """ProjectResponse.target_language should accept any ISO code."""
        from app.schemas.models import ProjectResponse
        from datetime import datetime

        pr = ProjectResponse(
            id=1,
            path="/some/path",
            name="project",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            target_language="zho",
        )

        assert pr.target_language == "zho"


# ---------------------------------------------------------------------------
# API layer
# ---------------------------------------------------------------------------


class TestUpdateTargetLanguageEndpoint:
    """Tests for PUT /api/projects/{id}/target-language."""

    def test_update_target_language_success(
        self, client: TestClient, project_id: int
    ) -> None:
        """Should return 200 with updated project including new target_language."""
        response = client.put(
            f"/api/projects/{project_id}/target-language",
            json={"target_language": "spa"},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["target_language"] == "spa"
        assert body["id"] == project_id

    def test_update_target_language_not_found(self, client: TestClient) -> None:
        """Should return 404 for a non-existent project."""
        response = client.put(
            "/api/projects/99999/target-language",
            json={"target_language": "fra"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_update_target_language_too_short(
        self, client: TestClient, project_id: int
    ) -> None:
        """Should return 422 when target_language is shorter than min_length=2."""
        response = client.put(
            f"/api/projects/{project_id}/target-language",
            json={"target_language": "e"},
        )

        assert response.status_code == 422

    def test_update_target_language_too_long(
        self, client: TestClient, project_id: int
    ) -> None:
        """Should return 422 when target_language exceeds max_length=10."""
        response = client.put(
            f"/api/projects/{project_id}/target-language",
            json={"target_language": "x" * 11},
        )

        assert response.status_code == 422

    def test_get_project_returns_target_language(
        self, client: TestClient, project_id: int
    ) -> None:
        """GET /api/projects/{id} should include target_language after update."""
        client.put(
            f"/api/projects/{project_id}/target-language",
            json={"target_language": "fra"},
        )

        response = client.get(f"/api/projects/{project_id}")

        assert response.status_code == 200
        assert response.json()["target_language"] == "fra"

    def test_new_project_has_default_target_language_eng(
        self, client: TestClient, project_id: int
    ) -> None:
        """GET on a freshly created project should return target_language='eng'."""
        response = client.get(f"/api/projects/{project_id}")

        assert response.status_code == 200
        assert response.json()["target_language"] == "eng"

    def test_list_projects_includes_target_language(
        self, client: TestClient, project_id: int
    ) -> None:
        """GET /api/projects/ list should include target_language on each item."""
        response = client.get("/api/projects/")

        assert response.status_code == 200
        projects = response.json()
        assert any(p["id"] == project_id for p in projects)
        for project in projects:
            assert "target_language" in project
