"""Tests for generated-subtitle review mapping."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services.subtitle_quality_rules import analyze_subtitle_quality
from app.services.subtitle_review import (
    SubtitleReviewError,
    build_subtitle_review,
    chunk_for_time,
    delete_subtitle_cue,
    find_review_srt,
    is_safe_chunk_name,
    list_review_srts,
    parse_srt,
    resolve_review_srt,
    seconds_to_srt_timestamp,
    update_subtitle_cue_text,
    write_srt_cues,
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

    def test_omits_staged_copy_identical_to_sidecar(self, tmp_path: Path) -> None:
        from app.services.existing_subtitles import workdir_subtitle_dest

        project = tmp_path / "proj"
        section = project / "SECTION 01. What Is Deformation"
        section.mkdir(parents=True)
        video = section / "01.mp4"
        video.write_bytes(b"x")
        (section / "01.eng.srt").write_text(SAMPLE_SRT)
        (section / "01.kor.srt").write_text("kor cues")
        dest = workdir_subtitle_dest(video, str(project), "proj")
        dest.parent.mkdir(parents=True)
        dest.write_text(SAMPLE_SRT)

        options = list_review_srts(video, str(project), "proj", "eng")
        labels = [option.label for option in options]
        sources = [option.source for option in options]

        assert labels == ["01.eng.srt", "01.kor.srt"]
        assert sources == ["sidecar", "sidecar"]
        found = find_review_srt(video, str(project), "proj", "eng")
        assert found is not None
        assert found.resolve() == (section / "01.eng.srt").resolve()

    def test_keeps_generated_file_when_content_differs(self, tmp_path: Path) -> None:
        from app.services.existing_subtitles import workdir_subtitle_dest

        project = tmp_path / "proj"
        project.mkdir()
        video = project / "01.mp4"
        video.write_bytes(b"x")
        (project / "01.eng.srt").write_text("original sidecar")
        dest = workdir_subtitle_dest(video, str(project), "proj")
        dest.parent.mkdir(parents=True)
        dest.write_text(SAMPLE_SRT)

        options = list_review_srts(video, str(project), "proj", "eng")
        labels = [option.label for option in options]

        assert labels[0] == "01.en.srt"
        assert "01.eng.srt" in labels

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


class TestAnalyzeSubtitleQuality:
    def test_detects_known_phrase(self) -> None:
        cues = [(0.0, 5.0, "Thank you for watching this video.")]
        analysis = analyze_subtitle_quality(cues)
        assert any(b.rule_id == "known_hallucination_phrase" for b in analysis.breaches)
        assert analysis.breaches[0].segment_index == 0
        assert len(analysis.rules) >= 10

    def test_marks_all_consecutive_duplicates(self) -> None:
        cues = [
            (0.0, 2.0, "Thank you, everyone."),
            (2.0, 4.0, "Thank you, everyone."),
            (4.0, 6.0, "Thank you, everyone."),
        ]
        analysis = analyze_subtitle_quality(cues)
        duplicate = [
            b
            for b in analysis.breaches
            if b.rule_id == "consecutive_duplicate_segments"
        ]
        assert {b.segment_index for b in duplicate} == {0, 1, 2}

    def test_marks_in_cue_phrase_loops(self) -> None:
        text = "Thank you " * 6
        cues = [(0.0, 10.0, text.strip())]
        analysis = analyze_subtitle_quality(cues)
        assert any(
            b.rule_id == "repeated_phrase_loop" and b.segment_index == 0
            for b in analysis.breaches
        )

    def test_marks_numbered_enumeration_hallucination(self) -> None:
        cues = [
            (249.0, 254.0, "4. Not with the water like a fool."),
            (254.0, 259.0, "5. Not with the water as a fool."),
            (259.0, 264.0, "6. Stop with the water."),
            (264.0, 269.0, "7. Not with the water as a fill."),
        ]
        analysis = analyze_subtitle_quality(cues)
        numbered = [
            b for b in analysis.breaches if b.rule_id == "numbered_enumeration_loop"
        ]
        assert len(numbered) == 4
        assert {b.segment_index for b in numbered} == {0, 1, 2, 3}

    def test_marks_shared_phrase_template_loop(self) -> None:
        cues = [
            (415.0, 425.0, "Not with the water like a fool."),
            (425.0, 435.0, "Stop with the water like a fool."),
            (435.0, 445.0, "Stop with the water like a fool."),
            (445.0, 455.0, "Stop with the water like a fool."),
            (455.0, 465.0, "Stop with the water like a fool."),
            (465.0, 475.0, "Not with the water as a fool."),
        ]
        analysis = analyze_subtitle_quality(cues)
        shared = [b for b in analysis.breaches if b.rule_id == "shared_phrase_run"]
        assert len(shared) == 6
        assert {b.segment_index for b in shared} == {0, 1, 2, 3, 4, 5}

    def test_marks_intra_cue_menu_hallucination(self) -> None:
        text = (
            "145. Ginger Stir-Fried Pork 146. Ginger Stir-Fried Pork "
            "147. Ginger Stir-Fried Pork 148. Ginger Stir-Fried Pork"
        )
        analysis = analyze_subtitle_quality([(250.0, 260.0, text)])
        assert any(
            b.rule_id == "intra_cue_numbered_list" and b.segment_index == 0
            for b in analysis.breaches
        )

    def test_marks_insert_belly_spam(self) -> None:
        text = "Insert belly " * 8
        analysis = analyze_subtitle_quality([(215.0, 225.0, text.strip())])
        assert any(
            b.rule_id == "phrase_spam_in_cue" and b.segment_index == 0
            for b in analysis.breaches
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
        assert len(result.quality_rules) >= 10
        assert result.quality_breaches == []
        assert all(segment.breached_rule_count == 0 for segment in result.segments)
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
        assert len(body["quality_rules"]) >= 10
        assert body["segments"][0]["breached_rule_count"] == 0
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

    def test_patch_updates_cue_text(self, client: TestClient, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        video = project_dir / "lesson.mp4"
        video.write_bytes(b"x")
        srt = project_dir / "lesson.en.srt"
        srt.write_text(SAMPLE_SRT)

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

        response = client.patch(
            f"/api/projects/{project_id}/videos/{video_id}/subtitle-review/cues/0",
            json={"text": "Edited hello."},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["segments"][0]["text"] == "Edited hello."
        assert body["segments"][0]["start"] == pytest.approx(0.0)
        assert body["segments"][1]["text"] == "This is a longer line of narration."
        assert "Edited hello." in srt.read_text(encoding="utf-8")

    def test_patch_refreshes_cached_quality(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        from app.services.video_subtitle_quality import load_quality_cache

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        video = project_dir / "lesson.mp4"
        video.write_bytes(b"x")
        srt = project_dir / "lesson.en.srt"
        srt.write_text(SAMPLE_SRT)

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

        dirty = client.patch(
            f"/api/projects/{project_id}/videos/{video_id}/subtitle-review/cues/0",
            json={"text": "Thank you for watching this video."},
        )
        assert dirty.status_code == 200
        cached = load_quality_cache(db, project_id)[str(srt.resolve())]
        assert cached.issue_count >= 1
        assert any(
            issue.rule_id == "known_hallucination_phrase" for issue in cached.issues
        )

        clean = client.patch(
            f"/api/projects/{project_id}/videos/{video_id}/subtitle-review/cues/0",
            json={"text": "A normal spoken line."},
        )
        assert clean.status_code == 200
        cached = load_quality_cache(db, project_id)[str(srt.resolve())]
        assert not any(
            issue.rule_id == "known_hallucination_phrase" for issue in cached.issues
        )

    def test_patch_rejects_empty_text(self, client: TestClient, tmp_path: Path) -> None:
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

        response = client.patch(
            f"/api/projects/{project_id}/videos/{video_id}/subtitle-review/cues/0",
            json={"text": "   "},
        )
        assert response.status_code == 422

    def test_delete_removes_cue(self, client: TestClient, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        video = project_dir / "lesson.mp4"
        video.write_bytes(b"x")
        srt = project_dir / "lesson.en.srt"
        srt.write_text(SAMPLE_SRT)

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

        response = client.delete(
            f"/api/projects/{project_id}/videos/{video_id}/subtitle-review/cues/1"
        )

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert [segment["index"] for segment in body["segments"]] == [0, 1]
        assert [segment["text"] for segment in body["segments"]] == [
            "Hello there.",
            "Short.",
        ]
        assert [cue[2] for cue in parse_srt(srt.read_text(encoding="utf-8"))] == [
            "Hello there.",
            "Short.",
        ]


class TestUpdateSubtitleCue:
    def test_rewrites_text_and_keeps_timings(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        srt = project / "lesson.en.srt"
        srt.write_text(SAMPLE_SRT)
        tts = project / ".redubber" / "lesson.mp4" / "04_tts"
        tts.mkdir(parents=True)
        stale = tts / "000.en.m4a"
        stale.write_bytes(b"old")
        assemble = project / ".redubber" / "lesson.mp4" / "05_target_audio_chunks"
        assemble.mkdir()
        (assemble / "chunk.m4a").write_bytes(b"mix")

        update_subtitle_cue_text(
            video_path=video,
            project_path=str(project),
            project_name="proj",
            target_language="eng",
            cue_index=0,
            text="Rewritten line",
        )

        cues = parse_srt(srt.read_text(encoding="utf-8"))
        assert cues[0][2] == "Rewritten line"
        assert cues[0][0] == pytest.approx(0.0)
        assert cues[0][1] == pytest.approx(3.5)
        assert cues[1][2] == "This is a longer line of narration."
        assert not stale.exists()
        assert not assemble.exists()

    def test_rejects_out_of_range_index(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        (project / "lesson.en.srt").write_text(SAMPLE_SRT)

        with pytest.raises(SubtitleReviewError, match="out of range"):
            update_subtitle_cue_text(
                video_path=video,
                project_path=str(project),
                project_name="proj",
                target_language="eng",
                cue_index=99,
                text="nope",
            )

    def test_srt_timestamp_roundtrip(self, tmp_path: Path) -> None:
        assert seconds_to_srt_timestamp(3.5) == "00:00:03,500"
        assert seconds_to_srt_timestamp(3723.04) == "01:02:03,040"
        path = tmp_path / "x.srt"
        write_srt_cues(path, [(0.0, 1.25, "Hi")])
        assert parse_srt(path.read_text(encoding="utf-8")) == [(0.0, 1.25, "Hi")]


class TestDeleteSubtitleCue:
    def test_rewrites_all_copies_and_invalidates_indexed_audio(
        self, tmp_path: Path
    ) -> None:
        from app.services.existing_subtitles import workdir_subtitle_dest

        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        sidecar = project / "lesson.en.srt"
        sidecar.write_text(SAMPLE_SRT)
        staged = workdir_subtitle_dest(video, str(project), "proj")
        staged.parent.mkdir(parents=True)
        staged.write_text(SAMPLE_SRT)

        root = project / ".redubber" / "lesson.mp4"
        tts = root / "04_tts"
        tts.mkdir()
        (tts / "000.en.m4a").write_bytes(b"tts0")
        (tts / "001.en.m4a").write_bytes(b"tts1")
        assemble = root / "05_target_audio_chunks"
        assemble.mkdir()
        (assemble / "chunk.m4a").write_bytes(b"mix")
        dubbed = root / "lesson.dubbed.mp4"
        dubbed.write_bytes(b"video")

        delete_subtitle_cue(
            video_path=video,
            project_path=str(project),
            project_name="proj",
            target_language="eng",
            cue_index=1,
            srt_path=str(sidecar),
        )

        expected = [(0.0, 3.5, "Hello there."), (15.0, 16.2, "Short.")]
        assert parse_srt(sidecar.read_text(encoding="utf-8")) == expected
        assert parse_srt(staged.read_text(encoding="utf-8")) == expected
        assert not tts.exists()
        assert not assemble.exists()
        assert not dubbed.exists()

    def test_can_delete_last_remaining_cue(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        srt = project / "lesson.en.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nOnly cue.\n")

        delete_subtitle_cue(
            video_path=video,
            project_path=str(project),
            project_name="proj",
            target_language="eng",
            cue_index=0,
        )

        assert srt.read_text(encoding="utf-8") == ""

    def test_rejects_out_of_range_index(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        (project / "lesson.en.srt").write_text(SAMPLE_SRT)

        with pytest.raises(SubtitleReviewError, match="out of range"):
            delete_subtitle_cue(
                video_path=video,
                project_path=str(project),
                project_name="proj",
                target_language="eng",
                cue_index=99,
            )
