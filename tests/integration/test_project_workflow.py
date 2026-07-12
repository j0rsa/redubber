"""Integration tests for project management workflows.

Tests end-to-end project creation, scanning, listing, updating, and deletion.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


@pytest.mark.integration
def test_create_project_and_scan(
    integration_client: TestClient, sample_project_path: str
) -> None:
    """Test creating a project and triggering a file scan.

    Verifies:
    - Project creation returns 201
    - Project data is persisted to database
    - Scan can be triggered
    - Videos are discovered and indexed

    Args:
        integration_client: FastAPI test client with lifespan.
        sample_project_path: Path to test project directory.
    """
    # Create project
    response = integration_client.post(
        "/api/projects/",
        json={"path": sample_project_path},
    )
    assert response.status_code == 201
    project_data = response.json()
    assert "id" in project_data
    assert project_data["path"] == sample_project_path
    project_id = project_data["id"]

    # Trigger scan
    scan_response = integration_client.post(f"/api/projects/{project_id}/scan")
    assert scan_response.status_code in (200, 202)  # Accept either immediate or async

    # Wait for scan to complete (with timeout)
    max_wait = 10  # seconds
    start_time = time.time()
    scan_completed = False

    while time.time() - start_time < max_wait:
        status_response = integration_client.get(f"/api/projects/{project_id}")
        if status_response.status_code == 200:
            project = status_response.json()
            # Check if scan has results (videos indexed)
            if project.get("video_count", 0) > 0:
                scan_completed = True
                break
        time.sleep(0.5)

    assert scan_completed, "Scan did not complete within timeout"

    # Verify videos were indexed
    videos_response = integration_client.get(f"/api/projects/{project_id}/videos")
    assert videos_response.status_code == 200
    videos = videos_response.json()
    assert len(videos) > 0, "No videos found after scan"


@pytest.mark.integration
def test_list_projects(integration_client: TestClient, test_video_dir) -> None:
    """Test listing multiple projects with proper ordering.

    Verifies:
    - Multiple projects can be created
    - List endpoint returns all projects
    - Projects are ordered by updated_at (newest first)

    Args:
        integration_client: FastAPI test client with lifespan.
        test_video_dir: Directory containing test video files.
    """
    # Create multiple projects
    project_paths = [
        str(test_video_dir / "project1"),
        str(test_video_dir / "project2"),
        str(test_video_dir / "project3"),
    ]

    created_ids = []
    for path in project_paths:
        response = integration_client.post("/api/projects/", json={"path": path})
        assert response.status_code == 201
        created_ids.append(response.json()["id"])
        time.sleep(0.1)  # Ensure different timestamps

    # List projects
    list_response = integration_client.get("/api/projects/")
    assert list_response.status_code == 200
    projects = list_response.json()

    # Verify all projects are in the list
    project_ids = [p["id"] for p in projects]
    for pid in created_ids:
        assert pid in project_ids

    # Verify ordering (newest first)
    # The most recently created should be first
    assert projects[0]["id"] == created_ids[-1]


@pytest.mark.integration
def test_update_voice_settings(
    integration_client: TestClient, sample_project_path: str
) -> None:
    """Test updating and persisting voice settings.

    Verifies:
    - Voice settings can be updated
    - Settings are persisted to database
    - Settings can be retrieved correctly

    Args:
        integration_client: FastAPI test client with lifespan.
        sample_project_path: Path to test project directory.
    """
    # Create project
    response = integration_client.post(
        "/api/projects/", json={"path": sample_project_path}
    )
    assert response.status_code == 201
    project_id = response.json()["id"]

    # Update voice settings
    voice_settings = {
        "voice": "nova",
        "instructions": "Speak with a clear, professional tone.",
    }
    update_response = integration_client.put(
        f"/api/projects/{project_id}/voice-settings",
        json=voice_settings,
    )
    assert update_response.status_code == 200

    # Retrieve and verify settings
    get_response = integration_client.get(f"/api/projects/{project_id}/voice-settings")
    assert get_response.status_code == 200
    retrieved_settings = get_response.json()
    assert retrieved_settings["voice"] == voice_settings["voice"]
    assert retrieved_settings["instructions"] == voice_settings["instructions"]


@pytest.mark.integration
def test_delete_project(
    integration_client: TestClient, sample_project_path: str
) -> None:
    """Test deleting a project and verifying database cleanup.

    Verifies:
    - Project can be deleted
    - Associated data is removed from database
    - Deleted project cannot be retrieved

    Args:
        integration_client: FastAPI test client with lifespan.
        sample_project_path: Path to test project directory.
    """
    # Create project
    response = integration_client.post(
        "/api/projects/", json={"path": sample_project_path}
    )
    assert response.status_code == 201
    project_id = response.json()["id"]

    # Delete project
    delete_response = integration_client.delete(f"/api/projects/{project_id}")
    assert delete_response.status_code in (200, 204)

    # Verify project is gone
    get_response = integration_client.get(f"/api/projects/{project_id}")
    assert get_response.status_code == 404


@pytest.mark.integration
def test_scan_concurrency(
    integration_client: TestClient, sample_project_path: str
) -> None:
    """Test concurrent scan requests with proper protection.

    Verifies:
    - Multiple concurrent scan requests are handled gracefully
    - No race conditions or database corruption
    - All requests complete successfully or return appropriate errors

    Args:
        integration_client: FastAPI test client with lifespan.
        sample_project_path: Path to test project directory.
    """
    # Create project
    response = integration_client.post(
        "/api/projects/", json={"path": sample_project_path}
    )
    assert response.status_code == 201
    project_id = response.json()["id"]

    # Send concurrent scan requests
    def trigger_scan() -> int:
        """Trigger a scan and return status code."""
        scan_response = integration_client.post(f"/api/projects/{project_id}/scan")
        return scan_response.status_code

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(trigger_scan) for _ in range(5)]
        results = [f.result() for f in futures]

    # All requests should complete with valid status codes
    for status_code in results:
        assert status_code in (200, 202, 409), f"Unexpected status code: {status_code}"

    # At least one should succeed
    assert any(code in (200, 202) for code in results)


@pytest.mark.integration
def test_project_metadata_persistence(
    integration_client: TestClient, sample_project_path: str
) -> None:
    """Test that project metadata persists across API calls.

    Verifies:
    - Created timestamp is set correctly
    - Updated timestamp changes on modifications
    - Project data remains consistent

    Args:
        integration_client: FastAPI test client with lifespan.
        sample_project_path: Path to test project directory.
    """
    # Create project
    create_response = integration_client.post(
        "/api/projects/", json={"path": sample_project_path}
    )
    assert create_response.status_code == 201
    project_id = create_response.json()["id"]
    created_at = create_response.json().get("created_at")
    assert created_at is not None

    # Get project immediately
    get_response = integration_client.get(f"/api/projects/{project_id}")
    assert get_response.status_code == 200
    initial_data = get_response.json()
    initial_updated_at = initial_data.get("updated_at")

    # Wait and update voice settings
    time.sleep(0.5)
    voice_update = {
        "voice": "alloy",
        "instructions": "Test instructions",
    }
    integration_client.put(
        f"/api/projects/{project_id}/voice-settings",
        json=voice_update,
    )

    # Get project again and verify updated_at changed
    final_response = integration_client.get(f"/api/projects/{project_id}")
    assert final_response.status_code == 200
    final_data = final_response.json()
    final_updated_at = final_data.get("updated_at")

    # Verify timestamps
    assert final_data.get("created_at") == created_at  # Created timestamp unchanged
    assert final_updated_at != initial_updated_at  # Updated timestamp changed
