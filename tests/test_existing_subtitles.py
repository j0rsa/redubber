"""Tests for reusing existing subtitles instead of running STT."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.existing_subtitles import (
    SUBTITLES_READY_PROGRESS,
    external_subtitle_records,
    find_reusable_subtitle,
    find_sidecar_subtitles,
    iter_sidecar_subtitles,
    segments_from_subtitle_file,
    stage_existing_subtitle,
    stage_target_subtitles_for_videos,
    subtitle_belongs_to_video,
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

    def test_numeric_stem_matches_only_that_lesson(self, tmp_path: Path) -> None:
        video = tmp_path / "01.mp4"
        video.write_bytes(b"fake-video")
        (tmp_path / "01.eng.srt").write_text("eng", encoding="utf-8")
        (tmp_path / "01.kor.srt").write_text("kor", encoding="utf-8")
        (tmp_path / "02.eng.srt").write_text("no", encoding="utf-8")
        (tmp_path / "010.eng.srt").write_text("no", encoding="utf-8")

        found = iter_sidecar_subtitles(video)

        assert [p.name for p in found] == ["01.eng.srt", "01.kor.srt"]

    def test_ignores_same_named_subs_in_other_folders(self, tmp_path: Path) -> None:
        section = tmp_path / "SECTION 01. What Is Deformation"
        extras = tmp_path / "extras"
        section.mkdir()
        extras.mkdir()
        video = section / "01.mp4"
        video.write_bytes(b"fake-video")
        (section / "01.eng.srt").write_text("eng", encoding="utf-8")
        (section / "01.kor.srt").write_text("kor", encoding="utf-8")
        (extras / "01.eng.srt").write_text("dup", encoding="utf-8")
        (extras / "01.en.srt").write_text("en", encoding="utf-8")
        (extras / "01.english.srt").write_text("english", encoding="utf-8")

        found = iter_sidecar_subtitles(video)
        records = external_subtitle_records(video)

        assert [p.name for p in found] == ["01.eng.srt", "01.kor.srt"]
        assert [r["language"] for r in records] == ["eng", "kor"]
        assert not subtitle_belongs_to_video(extras / "01.english.srt", video)


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

    def test_scan_attaches_only_same_folder_sidecars(self, client, tmp_path: Path) -> None:
        project_dir = tmp_path / "course"
        section = project_dir / "SECTION 01. What Is Deformation"
        extras = project_dir / "extras"
        section.mkdir(parents=True)
        extras.mkdir()
        (section / "01.mp4").write_bytes(b"fake-video")
        (section / "01.eng.srt").write_text("eng", encoding="utf-8")
        (section / "01.kor.srt").write_text("kor", encoding="utf-8")
        (extras / "01.eng.srt").write_text("dup", encoding="utf-8")
        (extras / "01.en.srt").write_text("en", encoding="utf-8")
        (extras / "01.english.srt").write_text("english", encoding="utf-8")
        (extras / "02.eng.srt").write_text("other", encoding="utf-8")

        created = client.post(
            "/api/projects/", json={"path": str(project_dir), "name": "Course"}
        )
        assert created.status_code == 201
        project_id = created.json()["id"]

        videos = client.get(f"/api/projects/{project_id}/videos")
        assert videos.status_code == 200
        body = videos.json()
        assert len(body) == 1
        names = sorted(Path(sub["path"]).name for sub in body[0]["subtitles"])
        langs = sorted(sub["language"] for sub in body[0]["subtitles"])
        assert names == ["01.eng.srt", "01.kor.srt"]
        assert langs == ["eng", "kor"]

        rescan = client.post(f"/api/projects/{project_id}/scan")
        assert rescan.status_code == 200
        videos = client.get(f"/api/projects/{project_id}/videos")
        body = videos.json()
        names = sorted(Path(sub["path"]).name for sub in body[0]["subtitles"])
        assert names == ["01.eng.srt", "01.kor.srt"]

    def test_scan_caches_hallucination_quality(self, client, tmp_path: Path) -> None:
        from unittest.mock import patch

        from app.core.config import settings
        from app.services.video_subtitle_quality import load_quality_cache
        from database import DatabaseManager

        project_dir = tmp_path / "videos"
        project_dir.mkdir()
        video = project_dir / "lesson.mp4"
        video.write_bytes(b"fake-video")
        srt = project_dir / "lesson.en.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:05,000\nThank you for watching this video.\n",
            encoding="utf-8",
        )

        created = client.post(
            "/api/projects/", json={"path": str(project_dir), "name": "Demo"}
        )
        assert created.status_code == 201
        project_id = created.json()["id"]

        db = DatabaseManager(settings.database_url)
        cached = load_quality_cache(db, project_id)
        hit = cached[str(srt.resolve())]
        assert hit.issue_count >= 1
        assert any(
            issue.rule_id == "known_hallucination_phrase" for issue in hit.issues
        )
        assert hit.video_path == str(video.resolve())

        with patch(
            "app.services.video_subtitle_quality.analyze_subtitle_file",
            side_effect=AssertionError("listing must not re-analyze after scan"),
        ):
            videos = client.get(f"/api/projects/{project_id}/videos")

        assert videos.status_code == 200
        eng = next(
            sub
            for sub in videos.json()[0]["subtitles"]
            if str(sub["path"]).endswith("lesson.en.srt")
        )
        assert eng["quality_issue_count"] >= 1


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


class TestSubtitleFilesForVideoQuery:
    def test_does_not_match_other_numeric_prefixes(self, tmp_path: Path) -> None:
        from app.core.config import settings
        from database import DatabaseManager

        db = DatabaseManager(settings.database_url)
        project_id = db.add_project(str(tmp_path), "Demo")
        video = tmp_path / "01.mp4"
        video.write_bytes(b"x")
        db.add_subtitle_file(project_id, str(tmp_path / "01.eng.srt"), "01.eng.srt", "eng")
        db.add_subtitle_file(project_id, str(tmp_path / "010.eng.srt"), "010.eng.srt", "eng")
        db.add_subtitle_file(project_id, str(tmp_path / "02.eng.srt"), "02.eng.srt", "eng")

        matched = db.get_subtitle_files_for_video(project_id, "01.mp4", str(video))
        assert [row["filename"] for row in matched] == ["01.eng.srt"]

    def test_requires_same_directory_when_path_given(self, tmp_path: Path) -> None:
        from app.core.config import settings
        from database import DatabaseManager

        db = DatabaseManager(settings.database_url)
        project_id = db.add_project(str(tmp_path), "Demo")
        section = tmp_path / "section"
        extras = tmp_path / "extras"
        section.mkdir()
        extras.mkdir()
        video = section / "01.mp4"
        video.write_bytes(b"x")
        db.add_subtitle_file(
            project_id, str(section / "01.eng.srt"), "01.eng.srt", "eng"
        )
        db.add_subtitle_file(
            project_id, str(extras / "01.english.srt"), "01.english.srt", "eng"
        )

        matched = db.get_subtitle_files_for_video(project_id, "01.mp4", str(video))
        assert [row["filename"] for row in matched] == ["01.eng.srt"]
