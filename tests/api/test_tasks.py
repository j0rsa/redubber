"""Tests for task management API endpoints."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient


class TestTasksAPI:
    """Test suite for /api/tasks endpoints."""

    def test_submit_redub_task_validates_schema(self, client: TestClient) -> None:
        """POST /api/redub rejects requests missing required fields."""
        response = client.post("/api/redub", json={"project_id": 1})
        assert response.status_code == 422

        response = client.post("/api/redub", json={"video_path": "/test/video.mp4"})
        assert response.status_code == 422

    def test_submit_redub_task_returns_202_or_400(self, client: TestClient) -> None:
        """POST /api/redub returns 202 on success or 400 if file missing."""
        response = client.post(
            "/api/redub",
            json={"video_path": "/nonexistent/video.mp4", "project_id": 1},
        )
        assert response.status_code in [202, 400]

    def test_submit_redub_extra_fields_ignored(self, client: TestClient) -> None:
        """POST /api/redub ignores unknown fields."""
        response = client.post(
            "/api/redub",
            json={
                "video_path": "/test/video.mp4",
                "project_id": 1,
                "unknown_field": "ignored",
            },
        )
        assert response.status_code in [202, 400]

    def test_get_task_status_not_found_returns_404(self, client: TestClient) -> None:
        """GET /api/tasks/{task_id} returns 404 for unknown task."""
        response = client.get(f"/api/tasks/{uuid.uuid4()}")
        assert response.status_code == 404

    def test_get_task_status_accepts_uuid_format(self, client: TestClient) -> None:
        """GET /api/tasks/{task_id} accepts UUID-format task IDs."""
        response = client.get(f"/api/tasks/{uuid.uuid4()}")
        assert response.status_code == 404  # not found, not 422

    def test_cancel_nonexistent_task_returns_404(self, client: TestClient) -> None:
        """POST /api/tasks/{task_id}/cancel returns 404 for unknown task."""
        response = client.post("/api/tasks/nonexistent-task-id/cancel")
        assert response.status_code == 404

    def test_list_tasks_returns_200(self, client: TestClient) -> None:
        """GET /api/tasks returns empty list when no tasks exist."""
        response = client.get("/api/tasks")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_tasks_accepts_status_filter(self, client: TestClient) -> None:
        """GET /api/tasks?status=queued accepts status query parameter."""
        response = client.get("/api/tasks?status=queued")
        assert response.status_code == 200

    def test_list_tasks_accepts_project_id_filter(self, client: TestClient) -> None:
        """GET /api/tasks?project_id=123 accepts project_id query parameter."""
        response = client.get("/api/tasks?project_id=123")
        assert response.status_code == 200

    def test_list_tasks_validates_project_id_type(self, client: TestClient) -> None:
        """GET /api/tasks?project_id=invalid rejects non-integer project_id."""
        response = client.get("/api/tasks?project_id=not-an-integer")
        assert response.status_code == 422


@pytest.mark.asyncio
class TestTasksAPIAsync:
    """Async test suite for task management endpoints."""

    async def test_list_tasks_is_async(self, client: TestClient) -> None:
        """GET /api/tasks works with async test client."""
        response = client.get("/api/tasks")
        assert response.status_code == 200
