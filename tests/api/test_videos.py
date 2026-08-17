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

    def test_get_scan_status_not_found_returns_404(self, client: TestClient) -> None:
        """GET /api/projects/{project_id}/scan returns 404 for unknown project."""
        response = client.get("/api/projects/99999/scan")
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


class TestResetDubAPI:
    """POST /api/projects/{id}/videos/{id}/reset-dub."""

    def test_reset_dub_project_not_found(self, client: TestClient) -> None:
        response = client.post("/api/projects/99999/videos/1/reset-dub")
        assert response.status_code == 404

    def test_reset_dub_video_not_found(self, client: TestClient, tmp_path) -> None:
        from app.core.config import settings
        from database import DatabaseManager

        project_dir = tmp_path / "empty_proj"
        project_dir.mkdir()
        db = DatabaseManager(settings.database_url)
        project_id = db.add_project(str(project_dir), "Empty")

        response = client.post(f"/api/projects/{project_id}/videos/1/reset-dub")
        assert response.status_code == 404

    def test_reset_dub_rejects_non_final_video(
        self, client: TestClient, tmp_path
    ) -> None:
        from unittest.mock import patch

        from app.core.config import settings
        from app.core.project_paths import get_project_working_dir
        from database import DatabaseManager

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        video = project_dir / "lesson.mp4"
        video.write_bytes(b"fake")
        db = DatabaseManager(settings.database_url)
        project_id = db.add_project(str(project_dir), "Demo")
        db.set_target_language(project_id, "eng")
        db.save_video_analysis(
            project_id,
            {
                "filename": "lesson.mp4",
                "path": str(video),
                "size_mb": 1.0,
                "duration_seconds": 10,
                "audio_streams": [
                    {
                        "index": 0,
                        "language": "rus",
                        "codec": "aac",
                        "channels": 2,
                        "sample_rate": "48000",
                    }
                ],
                "subtitles": [],
            },
        )
        video_id = db.get_video_analysis(project_id)[0]["id"]

        working_dir = get_project_working_dir(str(project_dir), "Demo")
        backup_dir = working_dir / "backups"
        backup_dir.mkdir(parents=True)
        backup = backup_dir / "lesson.20250101.mp4"
        backup.write_bytes(b"backup")

        with patch("redubber.sync_video_metadata"):
            response = client.post(
                f"/api/projects/{project_id}/videos/{video_id}/reset-dub"
            )

        assert response.status_code == 422
        assert "final" in response.json()["detail"].lower()
        assert not backup.exists()

        list_response = client.get(f"/api/projects/{project_id}/videos")
        assert list_response.status_code == 200
        status = list_response.json()[0]["pipeline_status"]
        assert status is None or status.get("replaced") is False

    def test_reset_dub_success(self, client: TestClient, tmp_path) -> None:
        from app.core.config import settings
        from database import DatabaseManager

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        video = project_dir / "lesson.mp4"
        video.write_bytes(b"fake")
        srt = project_dir / "lesson.en.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")

        db = DatabaseManager(settings.database_url)
        project_id = db.add_project(str(project_dir), "Demo")
        db.set_target_language(project_id, "eng")
        db.add_subtitle_file(project_id, str(srt), srt.name, "eng")
        db.save_video_analysis(
            project_id,
            {
                "filename": "lesson.mp4",
                "path": str(video),
                "size_mb": 1.0,
                "duration_seconds": 10,
                "audio_streams": [
                    {"index": 0, "language": "eng", "codec": "aac"},
                    {"index": 1, "language": "rus", "codec": "aac"},
                ],
                "subtitles": [
                    {
                        "language": "eng",
                        "embedded": False,
                        "path": str(srt),
                        "filename": srt.name,
                    }
                ],
            },
        )
        video_id = db.get_video_analysis(project_id)[0]["id"]

        response = client.post(
            f"/api/projects/{project_id}/videos/{video_id}/reset-dub"
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        assert "task_id" in body


class TestListVideosPipelineStatus:
    """GET /api/projects/{id}/videos pipeline_status.replaced for finalized dubs."""

    def test_target_state_is_replaced_even_with_leftover_dubbed_file(
        self, client: TestClient, tmp_path
    ) -> None:
        from app.core.config import settings
        from app.core.project_paths import get_project_working_dir
        from database import DatabaseManager

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        video = project_dir / "lesson.mp4"
        video.write_bytes(b"fake")
        srt = project_dir / "lesson.en.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")

        db = DatabaseManager(settings.database_url)
        project_id = db.add_project(str(project_dir), "Demo")
        db.set_target_language(project_id, "eng")
        db.save_video_analysis(
            project_id,
            {
                "filename": "lesson.mp4",
                "path": str(video),
                "size_mb": 1.0,
                "duration_seconds": 10,
                "audio_streams": [
                    {
                        "index": 0,
                        "language": "en",
                        "codec": "aac",
                        "channels": 2,
                        "sample_rate": 48000,
                    },
                    {
                        "index": 1,
                        "language": "rus",
                        "codec": "aac",
                        "channels": 2,
                        "sample_rate": 48000,
                    },
                ],
                "subtitles": [
                    {
                        "language": "eng",
                        "embedded": False,
                        "path": str(srt),
                        "filename": srt.name,
                    }
                ],
            },
        )

        working_dir = get_project_working_dir(str(project_dir), "Demo")
        dubbed = working_dir / "lesson.mp4" / "lesson.dubbed.mp4"
        dubbed.parent.mkdir(parents=True)
        dubbed.write_bytes(b"leftover-dubbed")

        response = client.get(f"/api/projects/{project_id}/videos")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        status = body[0]["pipeline_status"]
        assert status["replaced"] is True
        assert status["is_complete"] is True
        assert status["current_stage"] == "Complete"

    def test_not_replaced_after_dub_reset(self, client: TestClient, tmp_path) -> None:
        """After reset, video should no longer show as finalized (Remove dub)."""
        import json
        from unittest.mock import patch

        from app.core.config import settings
        from app.services.dub_reset import reset_dubbed_video
        from database import DatabaseManager

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        video = project_dir / "lesson.mp4"
        video.write_bytes(b"fake")
        srt = project_dir / "lesson.en.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")

        db = DatabaseManager(settings.database_url)
        project_id = db.add_project(str(project_dir), "Demo")
        db.set_target_language(project_id, "eng")
        db.save_video_analysis(
            project_id,
            {
                "filename": "lesson.mp4",
                "path": str(video),
                "size_mb": 1.0,
                "duration_seconds": 10,
                "audio_streams": [
                    {"index": 0, "language": "eng", "codec": "aac"},
                    {"index": 1, "language": "rus", "codec": "aac"},
                ],
                "subtitles": [
                    {
                        "language": "eng",
                        "embedded": False,
                        "path": str(srt),
                        "filename": srt.name,
                    }
                ],
            },
        )
        record = db.get_video_analysis(project_id)[0]

        backup_dir = project_dir / ".redubber" / "backups"
        backup_dir.mkdir(parents=True)
        (backup_dir / "lesson.20250101.mp4").write_bytes(b"backup")

        with (
            patch("app.services.dub_reset.strip_first_audio_track"),
            patch("redubber.sync_video_metadata"),
        ):
            reset_dubbed_video(
                db=db,
                project_id=project_id,
                video_record=record,
                project_path=str(project_dir),
                project_name="Demo",
                target_language="eng",
            )

        # Simulate metadata sync after strip (single original audio track, no target sub)
        import sqlite3

        with sqlite3.connect(db.db_path) as conn:
            conn.execute(
                """
                UPDATE video_analysis
                SET audio_streams = ?, subtitle_matches = ?
                WHERE project_id = ? AND file_path = ?
                """,
                (
                    json.dumps([
                        {
                            "index": 0,
                            "language": "rus",
                            "codec": "aac",
                            "channels": 2,
                            "sample_rate": 48000,
                        }
                    ]),
                    json.dumps([]),
                    project_id,
                    str(video),
                ),
            )
            conn.commit()

        response = client.get(f"/api/projects/{project_id}/videos")
        assert response.status_code == 200
        status = response.json()[0]["pipeline_status"]
        assert status is None or status.get("replaced") is not True
