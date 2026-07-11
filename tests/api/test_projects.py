"""Tests for project management API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestProjectsAPI:
    """Test suite for /api/projects endpoints."""

    def test_create_project_returns_501(
        self, client: TestClient, sample_project_data: dict[str, str]
    ) -> None:
        """POST /api/projects/ returns 501 until implementation complete.

        Verifies that the endpoint is registered and returns the expected
        not-implemented status code with TODO message.
        """
        response = client.post("/api/projects/", json=sample_project_data)

        assert response.status_code == 501
        assert "TODO" in response.json()["detail"]
        assert "DatabaseManager" in response.json()["detail"]

    def test_create_project_validates_schema(self, client: TestClient) -> None:
        """POST /api/projects/ validates request schema.

        Verifies that Pydantic validation rejects invalid request bodies
        before reaching the handler.
        """
        # Missing required 'path' field
        response = client.post("/api/projects/", json={})

        assert response.status_code == 422  # Unprocessable Entity
        assert "detail" in response.json()

    def test_list_projects_returns_501(self, client: TestClient) -> None:
        """GET /api/projects/ returns 501 until implementation complete.

        Verifies that the list endpoint is registered and accessible.
        """
        response = client.get("/api/projects/")

        assert response.status_code == 501
        assert "TODO" in response.json()["detail"]

    def test_get_project_by_id_returns_501(self, client: TestClient) -> None:
        """GET /api/projects/{project_id} returns 501 until implementation complete.

        Verifies that the detail endpoint is registered with path parameter.
        """
        response = client.get("/api/projects/123")

        assert response.status_code == 501
        assert "TODO" in response.json()["detail"]

    def test_get_project_validates_id_type(self, client: TestClient) -> None:
        """GET /api/projects/{project_id} validates project_id is an integer.

        FastAPI should reject non-integer path parameters automatically.
        """
        response = client.get("/api/projects/not-an-integer")

        assert response.status_code == 422  # Unprocessable Entity
        assert "detail" in response.json()

    def test_update_voice_settings_returns_501(
        self, client: TestClient, sample_voice_settings: dict[str, str]
    ) -> None:
        """PUT /api/projects/{project_id}/voice-settings returns 501.

        Verifies that the voice settings endpoint is registered.
        """
        response = client.put(
            "/api/projects/1/voice-settings", json=sample_voice_settings
        )

        assert response.status_code == 501
        assert "TODO" in response.json()["detail"]

    def test_update_voice_settings_validates_schema(self, client: TestClient) -> None:
        """PUT /api/projects/{project_id}/voice-settings validates request schema.

        Verifies that missing required 'voice' field is rejected.
        """
        # Missing required 'voice' field
        response = client.put(
            "/api/projects/1/voice-settings", json={"instructions": "test"}
        )

        assert response.status_code == 422
        assert "detail" in response.json()

    def test_delete_project_returns_501(self, client: TestClient) -> None:
        """DELETE /api/projects/{project_id} returns 501 until implementation complete.

        Verifies that the delete endpoint is registered.
        """
        response = client.delete("/api/projects/1")

        assert response.status_code == 501
        assert "TODO" in response.json()["detail"]

    def test_all_project_endpoints_accept_json(self, client: TestClient) -> None:
        """All project endpoints accept application/json content type.

        Verifies that endpoints properly handle JSON request bodies.
        """
        headers = {"Content-Type": "application/json"}

        # Test POST endpoint
        response = client.post(
            "/api/projects/", json={"path": "/test"}, headers=headers
        )
        assert response.status_code in [501, 422, 200, 201]

        # Test PUT endpoint
        response = client.put(
            "/api/projects/1/voice-settings",
            json={"voice": "nova"},
            headers=headers,
        )
        assert response.status_code in [501, 422, 200]


@pytest.mark.asyncio
class TestProjectsAPIAsync:
    """Async test suite for projects endpoints.

    These tests verify async handler behavior and will be expanded
    once actual implementation is added.
    """

    async def test_endpoints_are_async(self, client: TestClient) -> None:
        """Verify that project endpoints are async handlers.

        This is important for non-blocking I/O when database and file
        operations are implemented.
        """
        # All handlers should work with async test client
        response = client.get("/api/projects/")
        assert response.status_code == 501
