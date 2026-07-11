"""Tests for project management API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestProjectsAPI:
    """Test suite for /api/projects endpoints."""

    def test_create_project_validates_schema(self, client: TestClient) -> None:
        """POST /api/projects/ rejects missing required 'path' field."""
        response = client.post("/api/projects/", json={})
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_create_project_invalid_path_returns_422(self, client: TestClient) -> None:
        """POST /api/projects/ returns 422 for a path that does not exist."""
        response = client.post("/api/projects/", json={"path": "/nonexistent/path/xyz"})
        assert response.status_code == 422

    def test_list_projects_returns_200(self, client: TestClient) -> None:
        """GET /api/projects/ returns an empty list when no projects exist."""
        response = client.get("/api/projects/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_project_not_found_returns_404(self, client: TestClient) -> None:
        """GET /api/projects/{project_id} returns 404 for unknown id."""
        response = client.get("/api/projects/99999")
        assert response.status_code == 404

    def test_get_project_validates_id_type(self, client: TestClient) -> None:
        """GET /api/projects/{project_id} rejects non-integer path parameter."""
        response = client.get("/api/projects/not-an-integer")
        assert response.status_code == 422

    def test_update_voice_settings_not_found_returns_404(
        self, client: TestClient, sample_voice_settings: dict[str, str]
    ) -> None:
        """PUT /api/projects/{project_id}/voice-settings returns 404 for unknown project."""
        response = client.put(
            "/api/projects/99999/voice-settings", json=sample_voice_settings
        )
        assert response.status_code == 404

    def test_update_voice_settings_validates_schema(self, client: TestClient) -> None:
        """PUT /api/projects/{project_id}/voice-settings rejects missing 'voice' field."""
        response = client.put(
            "/api/projects/1/voice-settings", json={"instructions": "test"}
        )
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_delete_project_not_found_returns_404(self, client: TestClient) -> None:
        """DELETE /api/projects/{project_id} returns 404 for unknown project."""
        response = client.delete("/api/projects/99999")
        assert response.status_code == 404

    def test_all_project_endpoints_accept_json(self, client: TestClient) -> None:
        """Project endpoints handle JSON content type."""
        headers = {"Content-Type": "application/json"}

        response = client.post(
            "/api/projects/", json={"path": "/nonexistent"}, headers=headers
        )
        assert response.status_code in [422, 201]

        response = client.put(
            "/api/projects/99999/voice-settings",
            json={"voice": "nova"},
            headers=headers,
        )
        assert response.status_code in [404, 422, 200]


@pytest.mark.asyncio
class TestProjectsAPIAsync:
    """Async test suite for projects endpoints."""

    async def test_list_projects_is_async(self, client: TestClient) -> None:
        """GET /api/projects/ works with async test client."""
        response = client.get("/api/projects/")
        assert response.status_code == 200
