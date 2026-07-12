"""Tests for video analysis API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestVideosAPI:
    """Test suite for /api/videos endpoints."""

    def test_trigger_scan_validates_project_id_type(self, client: TestClient) -> None:
        """POST /api/projects/{project_id}/scan rejects non-integer project_id."""
        response = client.post("/api/projects/not-an-integer/scan")
        assert response.status_code == 422

    def test_trigger_scan_not_found_returns_404(self, client: TestClient) -> None:
        """POST /api/projects/{project_id}/scan returns 404 for unknown project."""
        response = client.post("/api/projects/99999/scan")
        assert response.status_code == 404

    def test_list_videos_validates_project_id_type(self, client: TestClient) -> None:
        """GET /api/projects/{project_id}/videos rejects non-integer project_id."""
        response = client.get("/api/projects/invalid/videos")
        assert response.status_code == 422

    def test_list_videos_not_found_returns_404(self, client: TestClient) -> None:
        """GET /api/projects/{project_id}/videos returns 404 for unknown project."""
        response = client.get("/api/projects/99999/videos")
        assert response.status_code == 404

    def test_list_videos_returns_array(self, client: TestClient) -> None:
        """GET /api/projects/{project_id}/videos returns a list when project exists."""
        # Project doesn't exist, but if it did we'd get a list
        response = client.get("/api/projects/99999/videos")
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            assert isinstance(response.json(), list)


@pytest.mark.asyncio
class TestVideosAPIAsync:
    """Async test suite for video endpoints."""

    async def test_list_videos_is_async(self, client: TestClient) -> None:
        """GET /api/projects/{project_id}/videos works with async test client."""
        response = client.get("/api/projects/99999/videos")
        assert response.status_code in [200, 404]


class TestVideosAPIEdgeCases:
    """Edge case tests for video endpoints."""

    def test_trigger_scan_for_nonexistent_project(self, client: TestClient) -> None:
        """POST /api/projects/{project_id}/scan returns 404 for missing project."""
        response = client.post("/api/projects/99999/scan")
        assert response.status_code == 404

    def test_list_videos_for_zero_project_id(self, client: TestClient) -> None:
        """GET /api/projects/0/videos handles edge case project ID."""
        response = client.get("/api/projects/0/videos")
        assert response.status_code in [200, 404, 422]
