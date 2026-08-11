"""Tests for project duration/size aggregate fields."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.schemas.models import ProjectResponse
from database import DatabaseManager


class TestProjectTotalsSchema:
    """ProjectResponse exposes total duration and size aggregates."""

    def test_defaults_to_zero(self) -> None:
        now = datetime.now()
        pr = ProjectResponse(
            id=1,
            path="/test",
            name="Test",
            created_at=now,
            updated_at=now,
        )
        assert pr.total_duration_seconds == 0
        assert pr.total_size_mb == 0

    def test_accepts_aggregate_values(self) -> None:
        now = datetime.now()
        pr = ProjectResponse(
            id=1,
            path="/test",
            name="Test",
            created_at=now,
            updated_at=now,
            total_duration_seconds=3661.5,
            total_size_mb=2048.25,
        )
        assert pr.total_duration_seconds == 3661.5
        assert pr.total_size_mb == 2048.25


class TestProjectTotalsDatabase:
    """DatabaseManager aggregates duration and size on count updates."""

    def test_update_project_video_counts_sets_aggregates(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        db = DatabaseManager(db_path=str(tmp_path / "totals.db"))
        project_id = db.add_project("/videos/demo", "Demo")

        db.save_video_analysis(
            project_id=project_id,
            video_data={
                "filename": "a.mp4",
                "path": "/videos/demo/a.mp4",
                "size_mb": 100.5,
                "duration_seconds": 600,
                "audio_streams": [],
                "subtitles": [],
            },
        )
        db.save_video_analysis(
            project_id=project_id,
            video_data={
                "filename": "b.mp4",
                "path": "/videos/demo/b.mp4",
                "size_mb": 200.25,
                "duration_seconds": 1200,
                "audio_streams": [],
                "subtitles": [],
            },
        )

        db.update_project_video_counts(project_id, total=2, replaced=1)
        project = db.get_project_by_id(project_id)

        assert project is not None
        assert project["total_videos"] == 2
        assert project["replaced_videos"] == 1
        assert project["total_duration_seconds"] == 1800
        assert project["total_size_mb"] == pytest.approx(300.75)

    def test_refresh_project_duration_size(self, tmp_path) -> None:
        db = DatabaseManager(db_path=str(tmp_path / "refresh.db"))
        project_id = db.add_project("/videos/r", "Refresh")
        db.save_video_analysis(
            project_id=project_id,
            video_data={
                "filename": "a.mp4",
                "path": "/videos/r/a.mp4",
                "size_mb": 10.0,
                "duration_seconds": 100,
                "audio_streams": [],
                "subtitles": [],
            },
        )
        db.refresh_project_duration_size(project_id)
        project = db.get_project_by_id(project_id)
        assert project["total_duration_seconds"] == 100
        assert project["total_size_mb"] == 10.0

    def test_clear_project_files_resets_aggregates(self, tmp_path) -> None:
        db = DatabaseManager(db_path=str(tmp_path / "clear.db"))
        project_id = db.add_project("/videos/c", "Clear")
        db.save_video_analysis(
            project_id=project_id,
            video_data={
                "filename": "a.mp4",
                "path": "/videos/c/a.mp4",
                "size_mb": 50.0,
                "duration_seconds": 200,
                "audio_streams": [],
                "subtitles": [],
            },
        )
        db.update_project_video_counts(project_id, total=1, replaced=0)
        db.clear_project_files(project_id)

        project = db.get_project_by_id(project_id)
        assert project["total_videos"] == 0
        assert project["replaced_videos"] == 0
        assert project["total_duration_seconds"] == 0
        assert project["total_size_mb"] == 0
        assert db.get_video_analysis(project_id) == []

    def test_rescan_repopulates_aggregates(self, tmp_path) -> None:
        """Simulates clear → re-analyze → update_project_video_counts (rescan)."""
        db = DatabaseManager(db_path=str(tmp_path / "rescan.db"))
        project_id = db.add_project("/videos/rs", "Rescan")

        # Stale pre-rescan state
        db.save_video_analysis(
            project_id=project_id,
            video_data={
                "filename": "old.mp4",
                "path": "/videos/rs/old.mp4",
                "size_mb": 1.0,
                "duration_seconds": 10,
                "audio_streams": [],
                "subtitles": [],
            },
        )
        db.update_project_video_counts(project_id, total=1, replaced=0)

        # Rescan: clear then write fresh analysis and recount
        db.clear_project_files(project_id)
        db.save_video_analysis(
            project_id=project_id,
            video_data={
                "filename": "new_a.mp4",
                "path": "/videos/rs/new_a.mp4",
                "size_mb": 120.0,
                "duration_seconds": 600,
                "audio_streams": [],
                "subtitles": [],
            },
        )
        db.save_video_analysis(
            project_id=project_id,
            video_data={
                "filename": "new_b.mp4",
                "path": "/videos/rs/new_b.mp4",
                "size_mb": 80.5,
                "duration_seconds": 300,
                "audio_streams": [],
                "subtitles": [],
            },
        )
        db.update_project_video_counts(project_id, total=2, replaced=0)

        project = db.get_project_by_id(project_id)
        assert project["total_videos"] == 2
        assert project["total_duration_seconds"] == 900
        assert project["total_size_mb"] == pytest.approx(200.5)

    def test_migration_adds_aggregate_columns(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Legacy DBs gain total_duration_seconds / total_size_mb on init."""
        import sqlite3

        db_path = str(tmp_path / "legacy.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_videos INTEGER DEFAULT 0,
                    replaced_videos INTEGER DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE video_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    size_mb REAL,
                    duration_seconds REAL,
                    audio_streams TEXT,
                    subtitle_matches TEXT,
                    status TEXT DEFAULT 'analyzed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "INSERT INTO projects (path, name, total_videos, replaced_videos) VALUES (?, ?, ?, ?)",
                ("/legacy", "Legacy", 1, 0),
            )
            conn.execute(
                """
                INSERT INTO video_analysis
                    (project_id, filename, file_path, size_mb, duration_seconds)
                VALUES (1, 'x.mp4', '/legacy/x.mp4', 50.0, 300.0)
                """
            )
            conn.commit()

        db = DatabaseManager(db_path=db_path)
        project = db.get_project_by_id(1)
        assert project is not None
        assert "total_duration_seconds" in project
        assert "total_size_mb" in project
        assert project["total_duration_seconds"] == 300.0
        assert project["total_size_mb"] == 50.0
