"""Tests for generated-subtitle review mapping."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services.subtitle_review import (
    SubtitleReviewError,
    analyze_srt_hallucinations,
    build_subtitle_review,
    chunk_for_time,
    find_review_srt,
    is_safe_chunk_name,
    list_review_srts,
    parse_srt,
    resolve_review_srt,
)
from database import DatabaseManager

SAMPLE_SRT = """1
00:00:00,000 --> 00:00:03,500
Hello there.

2
00:00:03,800 --> 00:00:10,000
This is a longer line of narration.

3
00:00:15,000 --> 00:00:16,200
Short.
"""


class TestParseSrt:
    def test_parses_cues(self) -> None:
        cues = parse_srt(SAMPLE_SRT)
        assert len(cues) == 3
        assert cues[0] == (0.0, 3.5, "Hello there.")
        assert cues[1][2] == "This is a longer line of narration."
        assert cues[2][0] == 15.0

    def test_skips_malformed_blocks(self) -> None:
        assert parse_srt("not a subtitle") == []


class TestChunkForTime:
    def test_picks_covering_chunk(self, tmp_path: Path) -> None:
        a = tmp_path / "a.m4a"
        b = tmp_path / "b.m4a"
        a.write_bytes(b"x")
        b.write_bytes(b"x")
        timeline = [(a, 0.0, 900.0), (b, 900.0, 1800.0)]
        hit = chunk_for_time(timeline, 901.0)
        assert hit is not None
        assert hit[0] == b
        assert chunk_for_time(timeline, 0.0)[0] == a


class TestSafeChunkName:
    def test_accepts_simple_names(self) -> None:
        assert is_safe_chunk_name("lesson_001.m4a")
        assert is_safe_chunk_name("16. Structure of the Chest Area_001.m4a")
        assert not is_safe_chunk_name("../secret.m4a")
        assert not is_safe_chunk_name("a/b.m4a")


class TestFindReviewSrt:
    def test_prefers_working_dir_copy(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        working = project / ".redubber" / "lesson.mp4" / "03_subtitles"
        working.mkdir(parents=True)
        generated = working / "lesson.en.srt"
        generated.write_text(SAMPLE_SRT)
        sidecar = project / "lesson.ru.srt"
        sidecar.write_text("orig")

        found = find_review_srt(video, str(project), "proj", "eng")
        assert found == generated

    def test_falls_back_to_target_language_sidecar(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        sidecar = project / "lesson.en.srt"
        sidecar.write_text(SAMPLE_SRT)
        (project / "lesson.ru.srt").write_text("orig")

        found = find_review_srt(video, str(project), "proj", "eng")
        assert found == sidecar


class TestListReviewSrts:
    def test_lists_multiple_sidecars(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        generated = project / ".redubber" / "lesson.mp4" / "03_subtitles"
        generated.mkdir(parents=True)
        (generated / "lesson.en.srt").write_text(SAMPLE_SRT)
        (project / "lesson.ru.srt").write_text("orig")

        options = list_review_srts(video, str(project), "proj", "eng")
        labels = [option.label for option in options]
        assert "lesson.en.srt" in labels
        assert "lesson.ru.srt" in labels

    def test_resolve_rejects_unknown_path(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        (project / "lesson.en.srt").write_text(SAMPLE_SRT)

        with pytest.raises(SubtitleReviewError):
            resolve_review_srt(
                video,
                str(project),
                "proj",
                "eng",
                srt_path=str(tmp_path / "other.srt"),
            )


class TestAnalyzeSrtHallucinations:
    def test_detects_known_phrase(self) -> None:
        cues = [(0.0, 5.0, "Thank you for watching this video.")]
        warnings = analyze_srt_hallucinations(cues, source_label="test.srt")
        assert any(w.code == "known_hallucination_phrase" for w in warnings)
        assert warnings[0].segment_index == 0

    def test_marks_all_consecutive_duplicates(self) -> None:
        cues = [
            (0.0, 2.0, "Thank you, everyone."),
            (2.0, 4.0, "Thank you, everyone."),
            (4.0, 6.0, "Thank you, everyone."),
        ]
        warnings = analyze_srt_hallucinations(cues, source_label="test.srt")
        duplicate = [
            w for w in warnings if w.code == "consecutive_duplicate_segments"
        ]
        assert {w.segment_index for w in duplicate} == {0, 1, 2}

    def test_marks_in_cue_phrase_loops(self) -> None:
        text = "Thank you " * 6
        cues = [(0.0, 10.0, text.strip())]
        warnings = analyze_srt_hallucinations(cues, source_label="test.srt")
        assert any(
            w.code == "repeated_phrase_loop" and w.segment_index == 0
            for w in warnings
        )


class TestBuildSubtitleReview:
    def test_maps_tts_and_chunks(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        srt = project / "lesson.en.srt"
        srt.write_text(SAMPLE_SRT)
        chunks = project / ".redubber" / "lesson.mp4" / "01_source_audio_chunks"
        tts = project / ".redubber" / "lesson.mp4" / "04_tts"
        chunks.mkdir(parents=True)
        tts.mkdir(parents=True)
        chunk = chunks / "lesson_001.m4a"
        chunk.write_bytes(b"audio")
        (tts / "000.en.m4a").write_bytes(b"tts0")
        (tts / "001.en.m4a").write_bytes(b"tts1")

        with patch("app.services.subtitle_review.probe_duration", return_value=60.0):
            result = build_subtitle_review(
                project_id=1,
                video_id=2,
                video_path=video,
                filename="lesson.mp4",
                project_path=str(project),
                project_name="proj",
                target_language="eng",
            )

        assert result.total == 3
        assert result.has_chunks is True
        assert result.has_tts is True
        assert len(result.available_files) >= 1
        assert result.hallucination_warnings == []
        first = result.segments[0]
        assert first.original is not None
        assert first.original.chunk_name == "lesson_001.m4a"
        assert first.original.seek_start == 0.0
        assert first.tts_url is not None
        assert result.segments[2].tts_url is None

    def test_duration_filter(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        (project / "lesson.en.srt").write_text(SAMPLE_SRT)

        result = build_subtitle_review(
            project_id=1,
            video_id=2,
            video_path=video,
            filename="lesson.mp4",
            project_path=str(project),
            project_name="proj",
            target_language="eng",
            min_duration=5.0,
        )
        assert result.returned == 1
        assert result.segments[0].index == 1

    def test_missing_srt_raises(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        with pytest.raises(SubtitleReviewError):
            build_subtitle_review(
                project_id=1,
                video_id=2,
                video_path=video,
                filename="lesson.mp4",
                project_path=str(project),
                project_name="proj",
                target_language="eng",
            )


class TestSubtitleReviewAPI:
    def test_not_found(self, client: TestClient) -> None:
        response = client.get("/api/projects/99999/videos/1/subtitle-review")
        assert response.status_code == 404

    def test_returns_script(self, client: TestClient, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        video = project_dir / "lesson.mp4"
        video.write_bytes(b"x")
        (project_dir / "lesson.en.srt").write_text(SAMPLE_SRT)

        db = DatabaseManager(settings.database_url)
        project_id = db.add_project(str(project_dir), "Demo")
        db.save_video_analysis(
            project_id,
            {
                "filename": "lesson.mp4",
                "path": str(video),
                "size_mb": 1.0,
                "duration_seconds": 20,
                "audio_streams": [],
                "subtitles": [],
            },
        )
        video_id = db.get_video_analysis(project_id)[0]["id"]

        response = client.get(
            f"/api/projects/{project_id}/videos/{video_id}/subtitle-review"
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 3
        assert body["available_files"]
        assert body["hallucination_warnings"] == []
        assert body["segments"][0]["text"] == "Hello there."
        assert body["segments"][1]["duration"] == pytest.approx(6.2)

    def test_selects_specific_srt(self, client: TestClient, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        video = project_dir / "lesson.mp4"
        video.write_bytes(b"x")
        alt = project_dir / "lesson.ru.srt"
        alt.write_text("""1
00:00:00,000 --> 00:00:02,000
Alternate.
""")
        (project_dir / "lesson.en.srt").write_text(SAMPLE_SRT)

        db = DatabaseManager(settings.database_url)
        project_id = db.add_project(str(project_dir), "Demo")
        db.save_video_analysis(
            project_id,
            {
                "filename": "lesson.mp4",
                "path": str(video),
                "size_mb": 1.0,
                "duration_seconds": 20,
                "audio_streams": [],
                "subtitles": [],
            },
        )
        video_id = db.get_video_analysis(project_id)[0]["id"]

        response = client.get(
            f"/api/projects/{project_id}/videos/{video_id}/subtitle-review",
            params={"srt_path": str(alt.resolve())},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["segments"][0]["text"] == "Alternate."
        assert len(body["available_files"]) >= 2
