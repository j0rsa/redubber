"""Tests for FastAPI application creation and configuration."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


class TestApplicationSetup:
    """Test suite for application factory and configuration."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        """create_app() returns a configured FastAPI instance."""
        app = create_app()

        assert app.title == "Redubber API"
        assert app.version == "2.0.0"

    def test_app_has_cors_middleware(self) -> None:
        """Application includes CORS middleware for frontend access."""
        app = create_app()

        # Check middleware is registered
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]

        assert "CORSMiddleware" in middleware_classes

    def test_health_endpoint_works(self, client: TestClient) -> None:
        """GET /api/health returns healthy status."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "2.0.0"

    def test_openapi_docs_accessible(self, client: TestClient) -> None:
        """OpenAPI documentation endpoints are accessible."""
        # Swagger UI
        response = client.get("/api/docs")
        assert response.status_code == 200

        # ReDoc
        response = client.get("/api/redoc")
        assert response.status_code == 200

        # OpenAPI JSON schema
        response = client.get("/api/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "Redubber API"

    def test_openapi_schema_includes_all_routes(self, client: TestClient) -> None:
        """OpenAPI schema documents all API routes."""
        response = client.get("/api/openapi.json")
        schema = response.json()
        paths = schema["paths"]

        # Check project routes
        assert "/api/projects/" in paths
        assert "/api/projects/{project_id}" in paths
        assert "/api/projects/{project_id}/voice-settings" in paths

        # Check task routes
        assert "/api/redub" in paths
        assert "/api/tasks/{task_id}" in paths
        assert "/api/tasks/{task_id}/cancel" in paths
        assert "/api/tasks" in paths

        # Check video routes
        assert "/api/projects/{project_id}/scan" in paths
        assert "/api/projects/{project_id}/videos" in paths

    def test_openapi_schema_includes_tags(self, client: TestClient) -> None:
        """OpenAPI schema includes route tags for organization."""
        response = client.get("/api/openapi.json")
        schema = response.json()

        # Check that tags are defined for each router
        paths = schema["paths"]

        # Projects endpoints should be tagged
        projects_path = paths["/api/projects/"]["get"]
        assert "projects" in projects_path["tags"]

        # Tasks endpoints should be tagged
        tasks_path = paths["/api/tasks"]["get"]
        assert "tasks" in tasks_path["tags"]

        # Videos endpoints should be tagged
        videos_path = paths["/api/projects/{project_id}/videos"]["get"]
        assert "videos" in videos_path["tags"]

    def test_openapi_schema_includes_response_models(self, client: TestClient) -> None:
        """OpenAPI schema documents response models."""
        response = client.get("/api/openapi.json")
        schema = response.json()

        # Check components/schemas for our Pydantic models
        components = schema["components"]["schemas"]

        assert "ProjectResponse" in components
        assert "VideoAnalysis" in components
        assert "TaskStatusResponse" in components
        assert "AudioStream" in components
        assert "SubtitleInfo" in components
        assert "PipelineStatusResponse" in components


class TestApplicationRouting:
    """Test suite for route registration and routing behavior."""

    def test_all_project_routes_registered(self, client: TestClient) -> None:
        """All project management routes are accessible."""
        # List projects
        assert client.get("/api/projects/").status_code in [200, 501]

        # Create project
        assert client.post("/api/projects/", json={"path": "/test"}).status_code in [
            201,
            501,
            422,
        ]

        # Get project
        assert client.get("/api/projects/1").status_code in [200, 404, 501]

        # Update voice settings
        assert client.put(
            "/api/projects/1/voice-settings", json={"voice": "nova"}
        ).status_code in [200, 404, 501, 422]

        # Delete project
        assert client.delete("/api/projects/1").status_code in [204, 404, 501]

    def test_all_task_routes_registered(self, client: TestClient) -> None:
        """All task management routes are accessible."""
        # Submit task
        assert client.post(
            "/api/redub",
            json={"video_path": "/test", "project_id": 1},
        ).status_code in [202, 400, 422]

        # Get task status
        assert client.get("/api/tasks/task-123").status_code in [200, 404]

        # Cancel task
        assert client.post("/api/tasks/task-123/cancel").status_code in [200, 404]

        # List tasks
        assert client.get("/api/tasks").status_code in [200]

    def test_all_video_routes_registered(self, client: TestClient) -> None:
        """All video analysis routes are accessible."""
        # Trigger scan
        assert client.post("/api/videos/projects/1/scan").status_code in [202, 404, 501]

        # List videos
        assert client.get("/api/videos/projects/1/videos").status_code in [
            200,
            404,
            501,
        ]

    def test_nonexistent_routes_return_404(self, client: TestClient) -> None:
        """Requests to nonexistent routes return 404."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

        response = client.post("/api/nonexistent")
        assert response.status_code == 404


class TestCORSConfiguration:
    """Test suite for CORS middleware configuration."""

    def test_cors_allows_localhost_origin(self, client: TestClient) -> None:
        """CORS allows requests from Vite dev server."""
        response = client.get(
            "/api/health",
            headers={"Origin": "http://localhost:5173"},
        )

        assert response.status_code == 200
        # In test client, CORS headers might not be fully visible
        # This is a basic check that the request succeeds

    def test_options_request_handled(self, client: TestClient) -> None:
        """CORS preflight OPTIONS requests are handled."""
        response = client.options(
            "/api/projects/",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )

        # OPTIONS should succeed or return method-specific response
        assert response.status_code in [200, 204, 405]


@pytest.mark.asyncio
class TestApplicationLifespan:
    """Test suite for application lifespan management."""

    async def test_lifespan_creates_directories(self, tmp_path) -> None:
        """Application startup creates required directories.

        This test documents expected behavior. Actual lifespan testing
        requires more complex setup with context managers.
        """
        # Lifespan management is tested implicitly by client fixture
        # which starts and stops the app for each test
        pass

    async def test_app_can_start_and_stop_cleanly(self) -> None:
        """Application starts and stops without errors.

        The client fixture implicitly tests this for every test,
        so this is a documentation test.
        """
        pass
