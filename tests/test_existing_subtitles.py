"""Tests for reusing existing subtitles instead of running STT."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.existing_subtitles import (
    SUBTITLES_READY_PROGRESS,
    find_reusable_subtitle,
    find_sidecar_subtitles,
    segments_from_subtitle_file,
    stage_existing_subtitle,
    stage_target_subtitles_for_videos,
    workdir_subtitle_dest,
)
from pipeline_status import get_pipeline_status

SAMPLE_SRT = """1
00:00:00,000 --> 00:00:02,500
Hello there.

2
00:00:03,000 --> 00:00:06,000
Welcome to the lesson.
"""


def _write_video_and_sub(
    folder: Path, sub_name: str, content: str = SAMPLE_SRT
) -> Path:
    video = folder / "lesson.mp4"
    video.write_bytes(b"fake-video")
    (folder / sub_name).write_text(content, encoding="utf-8")
    return video


class TestFindSidecarSubtitles:
    def test_matches_target_language_suffix(self, tmp_path: Path) -> None:
        video = _write_video_and_sub(tmp_path, "lesson.en.srt")
        (tmp_path / "lesson.ru.srt").write_text("ru", encoding="utf-8")

        found = find_sidecar_subtitles(video, "eng")

        assert [p.name for p in found] == ["lesson.en.srt"]

    def test_unsuffixed_counts_as_target(self, tmp_path: Path) -> None:
        video = _write_video_and_sub(tmp_path, "lesson.srt")

        found = find_sidecar_subtitles(video, "eng", include_unsuffixed=True)

        assert [p.name for p in found] == ["lesson.srt"]

    def test_source_language_sidecar_ignored_for_target(self, tmp_path: Path) -> None:
        video = _write_video_and_sub(tmp_path, "lesson.ru.srt")

        assert find_sidecar_subtitles(video, "eng") == []

    def test_any_language_when_unfiltered(self, tmp_path: Path) -> None:
        video = _write_video_and_sub(tmp_path, "lesson.ru.srt")

        found = find_sidecar_subtitles(video, language=None)

        assert [p.name for p in found] == ["lesson.ru.srt"]


class TestStageExistingSubtitle:
    def test_copies_sidecar_into_workdir(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = _write_video_and_sub(project, "lesson.en.srt")

        dest = stage_existing_subtitle(
            video,
            project_path=str(project),
            project_name="Demo",
            language="eng",
        )

        expected = workdir_subtitle_dest(video, str(project), "Demo")
        assert dest == expected
        assert dest.is_file()
        assert dest.read_text(encoding="utf-8") == SAMPLE_SRT

    def test_does_not_overwrite_existing_workdir_copy(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = _write_video_and_sub(project, "lesson.en.srt")
        dest = workdir_subtitle_dest(video, str(project), "Demo")
        dest.parent.mkdir(parents=True)
        dest.write_text("already staged", encoding="utf-8")

        result = stage_existing_subtitle(
            video,
            project_path=str(project),
            project_name="Demo",
            language="eng",
        )

        assert result == dest
        assert dest.read_text(encoding="utf-8") == "already staged"

    def test_scan_helper_stages_all_videos(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = _write_video_and_sub(project, "lesson.en.srt")

        staged = stage_target_subtitles_for_videos(
            [video],
            project_path=str(project),
            project_name="Demo",
            target_language="eng",
        )

        assert len(staged) == 1
        assert staged[0].is_file()


class TestSegmentsFromSubtitle:
    def test_parses_cues_into_transcription_segments(self, tmp_path: Path) -> None:
        srt = tmp_path / "lesson.en.srt"
        srt.write_text(SAMPLE_SRT, encoding="utf-8")

        segments = segments_from_subtitle_file(srt)

        assert len(segments) == 2
        assert segments[0].text == "Hello there."
        assert segments[0].start == 0.0
        assert segments[0].end == 2.5
        assert segments[1].text == "Welcome to the lesson."

    def test_empty_on_missing_file(self, tmp_path: Path) -> None:
        assert segments_from_subtitle_file(tmp_path / "missing.srt") == []


class TestPipelineStatusWithExistingSubs:
    def test_sidecar_sets_progress_to_subtitles_ready(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = _write_video_and_sub(project, "lesson.en.srt")
        working = project / ".redubber"

        status = get_pipeline_status(
            str(video), str(project), str(working), target_language="eng"
        )

        assert status.has_external_subs is True
        assert status.progress_percent == SUBTITLES_READY_PROGRESS
        assert status.current_stage == "Generate TTS"

    def test_staged_workdir_sub_sets_progress(self, tmp_path: Path) -> None:
        from app.core.project_paths import get_project_working_dir

        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"fake")
        dest = workdir_subtitle_dest(video, str(project), "Demo")
        dest.parent.mkdir(parents=True)
        dest.write_text(SAMPLE_SRT, encoding="utf-8")
        working = get_project_working_dir(str(project), "Demo")

        status = get_pipeline_status(
            str(video), str(project), str(working), target_language="eng"
        )

        assert status.subtitles_generated is True
        assert status.progress_percent == SUBTITLES_READY_PROGRESS

    def test_source_language_sidecar_does_not_skip_stt(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = _write_video_and_sub(project, "lesson.ru.srt")
        working = project / ".redubber"

        status = get_pipeline_status(
            str(video), str(project), str(working), target_language="eng"
        )

        assert status.has_external_subs is False
        assert status.progress_percent == 0
        assert (
            find_reusable_subtitle(
                video,
                project_path=str(project),
                project_name="Demo",
                language="eng",
            )
            is None
        )


class TestVoiceAnalysisLoadsExistingSubs:
    def test_load_real_segments_from_sidecar(self, tmp_path: Path) -> None:
        from app.api.routes.voice_refinement import _load_real_segments
        from app.core.config import settings
        from database import DatabaseManager

        project = tmp_path / "proj"
        project.mkdir()
        video = _write_video_and_sub(project, "lesson.en.srt")
        db = DatabaseManager(settings.database_url)
        project_id = db.add_project(str(project), "Demo")
        db.save_video_analysis(
            project_id,
            {
                "filename": "lesson.mp4",
                "path": str(video),
                "size_mb": 1.0,
                "duration_seconds": 10,
                "audio_streams": [],
                "subtitles": [
                    {
                        "language": "eng",
                        "embedded": False,
                        "path": str(project / "lesson.en.srt"),
                        "filename": "lesson.en.srt",
                    }
                ],
            },
        )

        segments = _load_real_segments(project_id, str(project))

        assert len(segments) == 2
        assert segments[0].original_text == "Hello there."
        assert segments[1].original_text == "Welcome to the lesson."
        assert segments[0].video_filename == "lesson.mp4"


class TestScanStagesSubsAndProgress:
    def test_create_project_scan_copies_sub_and_shows_progress(
        self, client, tmp_path: Path
    ) -> None:
        from app.services.existing_subtitles import SUBTITLES_READY_PROGRESS

        project_dir = tmp_path / "videos"
        project_dir.mkdir()
        _write_video_and_sub(project_dir, "lesson.en.srt")

        created = client.post(
            "/api/projects/", json={"path": str(project_dir), "name": "Demo"}
        )
        assert created.status_code == 201
        project_id = created.json()["id"]

        dest = workdir_subtitle_dest(
            project_dir / "lesson.mp4", str(project_dir), "Demo"
        )
        assert dest.is_file()

        videos = client.get(f"/api/projects/{project_id}/videos")
        assert videos.status_code == 200
        body = videos.json()
        assert body
        statuses = [v["pipeline_status"] for v in body if v.get("pipeline_status")]
        assert statuses
        assert any(s["progress"] == SUBTITLES_READY_PROGRESS for s in statuses)
        assert any(s["current_stage"] == "Generate TTS" for s in statuses)


class TestTranscriptionTaskDedupe:
    @pytest.mark.asyncio
    async def test_reuses_active_transcription_task(self, monkeypatch) -> None:
        import asyncio

        from app.infrastructure.task_queue import TaskQueueManager

        manager = TaskQueueManager()

        async def hang(task_id: str) -> None:
            await manager._update_task_status(
                task_id, stage="Transcribing", progress=20, status="running"
            )
            await asyncio.sleep(0.3)

        monkeypatch.setattr(manager, "_process_transcription_task", hang)

        first = await manager.submit_transcription_task("/videos/lesson.mp4", 1)
        second = await manager.submit_transcription_task("/videos/lesson.mp4", 1)

        assert first == second
        assert len(manager._tasks) == 1

    @pytest.mark.asyncio
    async def test_skips_stt_when_subtitle_exists(self, tmp_path: Path) -> None:
        import asyncio
        from unittest.mock import patch

        from app.core.config import settings
        from app.infrastructure.task_queue import TaskQueueManager
        from database import DatabaseManager

        project = tmp_path / "proj"
        project.mkdir()
        video = _write_video_and_sub(project, "lesson.en.srt")
        db = DatabaseManager(settings.database_url)
        project_id = db.add_project(str(project), "Demo")

        manager = TaskQueueManager()
        with patch("redubber.Redubber.get_text_and_segments") as mock_stt:
            task_id = await manager.submit_transcription_task(str(video), project_id)
            status = None
            for _ in range(80):
                status = await manager.get_status(task_id)
                if status and status.status in ("completed", "failed"):
                    break
                await asyncio.sleep(0.05)

        assert status is not None
        assert status.status == "completed", status.error
        assert status.progress == 100
        mock_stt.assert_not_called()

    @pytest.mark.asyncio
    async def test_source_language_sub_does_not_stage_as_target(
        self, tmp_path: Path
    ) -> None:
        import asyncio
        from unittest.mock import patch

        from app.core.config import settings
        from app.infrastructure.task_queue import TaskQueueManager
        from database import DatabaseManager

        project = tmp_path / "proj"
        project.mkdir()
        video = _write_video_and_sub(project, "lesson.ru.srt")
        db = DatabaseManager(settings.database_url)
        project_id = db.add_project(str(project), "Demo")

        manager = TaskQueueManager()
        with patch("redubber.Redubber.get_text_and_segments") as mock_stt:
            task_id = await manager.submit_transcription_task(str(video), project_id)
            status = None
            for _ in range(80):
                status = await manager.get_status(task_id)
                if status and status.status in ("completed", "failed"):
                    break
                await asyncio.sleep(0.05)

        assert status is not None
        assert status.status == "completed", status.error
        mock_stt.assert_not_called()
        dest = workdir_subtitle_dest(video, str(project), "Demo")
        assert not dest.exists()
