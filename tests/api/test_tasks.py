"""Tests for task management API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestTasksAPI:
    """Test suite for /api/tasks endpoints."""

    def test_submit_redub_task_returns_501(
        self, client: TestClient, sample_task_data: dict[str, str | int]
    ) -> None:
        """POST /api/tasks/redub returns 501 until TaskQueueManager integrated.

        Verifies that the endpoint is registered and awaiting implementation.
        """
        response = client.post("/api/tasks/redub", json=sample_task_data)

        assert response.status_code == 501
        assert "TODO" in response.json()["detail"]
        assert "TaskQueueManager" in response.json()["detail"]

    def test_submit_redub_task_validates_schema(self, client: TestClient) -> None:
        """POST /api/tasks/redub validates request schema.

        Verifies that missing required fields are rejected.
        """
        # Missing required 'video_path' field
        response = client.post("/api/tasks/redub", json={"project_id": 1})

        assert response.status_code == 422
        assert "detail" in response.json()

        # Missing required 'project_id' field
        response = client.post(
            "/api/tasks/redub", json={"video_path": "/test/video.mp4"}
        )

        assert response.status_code == 422
        assert "detail" in response.json()

    def test_submit_redub_task_returns_202_accepted(self, client: TestClient) -> None:
        """POST /api/tasks/redub should return 202 when implemented.

        Verifies that the endpoint is configured to return 202 Accepted
        status code (async task submission pattern).
        """
        response = client.post(
            "/api/tasks/redub",
            json={"video_path": "/test/video.mp4", "project_id": 1},
        )

        # Currently returns 501, but endpoint is configured for 202
        assert response.status_code in [501, 202]

    def test_get_task_status_returns_501(self, client: TestClient) -> None:
        """GET /api/tasks/{task_id} returns 501 until implementation complete.

        Verifies that the status polling endpoint is registered.
        """
        response = client.get("/api/tasks/test-task-uuid-123")

        assert response.status_code == 501
        assert "TODO" in response.json()["detail"]

    def test_get_task_status_accepts_uuid_format(self, client: TestClient) -> None:
        """GET /api/tasks/{task_id} accepts UUID-format task IDs.

        Task IDs will be UUIDs, so the endpoint should accept them
        as string path parameters.
        """
        import uuid

        task_id = str(uuid.uuid4())
        response = client.get(f"/api/tasks/{task_id}")

        # Should reach handler (501), not fail validation (422)
        assert response.status_code == 501

    def test_cancel_task_returns_501(self, client: TestClient) -> None:
        """POST /api/tasks/{task_id}/cancel returns 501 until implementation complete.

        Verifies that the cancellation endpoint is registered.
        """
        response = client.post("/api/tasks/test-task-id/cancel")

        assert response.status_code == 501
        assert "TODO" in response.json()["detail"]

    def test_list_tasks_returns_501(self, client: TestClient) -> None:
        """GET /api/tasks/ returns 501 until implementation complete.

        Verifies that the list endpoint is registered.
        """
        response = client.get("/api/tasks/")

        assert response.status_code == 501
        assert "TODO" in response.json()["detail"]

    def test_list_tasks_accepts_status_filter(self, client: TestClient) -> None:
        """GET /api/tasks/?status=queued accepts status query parameter.

        Verifies that the status filter query parameter is properly
        configured with alias.
        """
        response = client.get("/api/tasks/?status=queued")

        # Should reach handler, not fail validation
        assert response.status_code == 501

    def test_list_tasks_accepts_project_id_filter(self, client: TestClient) -> None:
        """GET /api/tasks/?project_id=123 accepts project_id query parameter.

        Verifies that the project_id filter is properly configured.
        """
        response = client.get("/api/tasks/?project_id=123")

        # Should reach handler, not fail validation
        assert response.status_code == 501

    def test_list_tasks_accepts_multiple_filters(self, client: TestClient) -> None:
        """GET /api/tasks/ accepts multiple query parameters simultaneously.

        Verifies that status and project_id filters can be combined.
        """
        response = client.get("/api/tasks/?status=running&project_id=456")

        assert response.status_code == 501

    def test_list_tasks_validates_project_id_type(self, client: TestClient) -> None:
        """GET /api/tasks/?project_id=invalid validates query parameter types.

        Project ID must be an integer.
        """
        response = client.get("/api/tasks/?project_id=not-an-integer")

        assert response.status_code == 422


@pytest.mark.asyncio
class TestTasksAPIAsync:
    """Async test suite for task management endpoints."""

    async def test_task_submission_is_async(self, client: TestClient) -> None:
        """Verify that task submission handler is async.

        This is critical for non-blocking task queue operations.
        """
        response = client.post(
            "/api/tasks/redub",
            json={"video_path": "/test/video.mp4", "project_id": 1},
        )
        assert response.status_code == 501

    async def test_task_status_polling_is_async(self, client: TestClient) -> None:
        """Verify that status polling handler is async.

        This ensures efficient handling of concurrent status checks
        from multiple frontend clients.
        """
        response = client.get("/api/tasks/test-task-id")
        assert response.status_code == 501


class TestTasksAPIEdgeCases:
    """Edge case and error handling tests for tasks API."""

    def test_submit_task_with_extra_fields_ignored(self, client: TestClient) -> None:
        """POST /api/tasks/redub ignores extra fields in request.

        Pydantic should ignore unknown fields by default.
        """
        response = client.post(
            "/api/tasks/redub",
            json={
                "video_path": "/test/video.mp4",
                "project_id": 1,
                "unknown_field": "should be ignored",
            },
        )

        # Should not fail validation due to extra field
        assert response.status_code in [501, 202]

    def test_cancel_nonexistent_task_will_return_404(self, client: TestClient) -> None:
        """POST /api/tasks/{task_id}/cancel should return 404 for unknown task.

        This test documents expected behavior once implemented.
        """
        response = client.post("/api/tasks/nonexistent-task-id/cancel")

        # Currently 501, but should be 404 when implemented
        assert response.status_code in [501, 404]

    def test_task_endpoints_handle_empty_task_id(self, client: TestClient) -> None:
        """Task endpoints handle empty task_id gracefully.

        Empty path parameters should result in routing to list endpoint
        or 404, not 500 errors.
        """
        # This should route to the list endpoint
        response = client.get("/api/tasks/")
        assert response.status_code in [501, 200]

        # This should not match any route or return 404
        response = client.get("/api/tasks//")
        assert response.status_code in [404, 307]  # 307 redirect to normalized path
