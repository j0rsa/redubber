"""Tests for attaching quality-rule summaries to project file-list subtitles."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.core.config import settings
from app.schemas.models import SubtitleInfo
from app.services.subtitle_quality_rules import analyze_subtitle_file
from app.services.video_subtitle_quality import (
    CachedSubtitleQuality,
    enrich_subtitles_with_quality,
    load_quality_cache,
    refresh_project_subtitle_quality,
    refresh_subtitle_quality_for_paths,
    store_subtitle_quality,
)
from database import DatabaseManager

HALLUCINATION_SRT = """1
00:00:00,000 --> 00:00:05,000
Thank you for watching this video.
"""

CLEAN_SRT = """1
00:00:00,000 --> 00:00:02,500
Hello there.

2
00:00:03,000 --> 00:00:06,000
Welcome to the lesson.
"""


def _db() -> DatabaseManager:
    return DatabaseManager(settings.database_url)


def _project_with_video(
    tmp_path: Path, srt_text: str
) -> tuple[DatabaseManager, int, Path, Path]:
    project = tmp_path / "proj"
    project.mkdir()
    video = project / "lesson.mp4"
    video.write_bytes(b"x")
    srt = project / "lesson.en.srt"
    srt.write_text(srt_text, encoding="utf-8")

    db = _db()
    project_id = db.add_project(str(project), "proj")
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
                    "path": str(srt),
                    "filename": srt.name,
                }
            ],
        },
    )
    return db, project_id, video, srt


class TestAnalyzeSubtitleFile:
    def test_detects_known_phrase_on_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "lesson.en.srt"
        path.write_text(HALLUCINATION_SRT, encoding="utf-8")
        analysis = analyze_subtitle_file(path)
        assert any(b.rule_id == "known_hallucination_phrase" for b in analysis.breaches)

    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        analysis = analyze_subtitle_file(tmp_path / "missing.srt")
        assert analysis.breaches == ()


class TestStoreSubtitleQuality:
    def test_persists_issues_for_hallucinated_file(self, tmp_path: Path) -> None:
        db, project_id, video, srt = _project_with_video(tmp_path, HALLUCINATION_SRT)

        count, issues = store_subtitle_quality(
            db, project_id=project_id, video_path=video, subtitle_path=srt
        )

        assert count >= 1
        assert any(issue.rule_id == "known_hallucination_phrase" for issue in issues)
        cached = load_quality_cache(db, project_id)
        hit = cached[str(srt.resolve())]
        assert hit.issue_count == count
        assert hit.issues == issues
        assert hit.video_path == str(video.resolve())

    def test_overwrite_on_edit(self, tmp_path: Path) -> None:
        db, project_id, video, srt = _project_with_video(tmp_path, HALLUCINATION_SRT)
        store_subtitle_quality(
            db, project_id=project_id, video_path=video, subtitle_path=srt
        )
        assert load_quality_cache(db, project_id)[str(srt.resolve())].issue_count >= 1

        srt.write_text(CLEAN_SRT, encoding="utf-8")
        refresh_subtitle_quality_for_paths(
            db, project_id=project_id, video_path=video, paths=[srt]
        )

        hit = load_quality_cache(db, project_id)[str(srt.resolve())]
        assert hit.issue_count == 0
        assert hit.issues == []


class TestEnrichSubtitlesWithQuality:
    def test_attaches_cached_issues_to_matching_sidecar(self, tmp_path: Path) -> None:
        db, project_id, video, srt = _project_with_video(tmp_path, HALLUCINATION_SRT)
        store_subtitle_quality(
            db, project_id=project_id, video_path=video, subtitle_path=srt
        )
        cache = load_quality_cache(db, project_id)

        enriched = enrich_subtitles_with_quality(
            video_path=video,
            project_path=str(video.parent),
            project_name="proj",
            target_language="eng",
            subtitles=[
                SubtitleInfo(language="eng", embedded=False, path=str(srt)),
            ],
            cache=cache,
        )

        assert len(enriched) == 1
        assert enriched[0].quality_issue_count >= 1
        assert any(
            issue.rule_id == "known_hallucination_phrase"
            for issue in enriched[0].quality_issues
        )

    def test_clean_sidecar_has_zero_issues(self, tmp_path: Path) -> None:
        db, project_id, video, srt = _project_with_video(tmp_path, CLEAN_SRT)
        store_subtitle_quality(
            db, project_id=project_id, video_path=video, subtitle_path=srt
        )

        enriched = enrich_subtitles_with_quality(
            video_path=video,
            project_path=str(video.parent),
            project_name="proj",
            target_language="eng",
            subtitles=[
                SubtitleInfo(language="eng", embedded=False, path=str(srt)),
            ],
            cache=load_quality_cache(db, project_id),
        )

        assert enriched[0].quality_issue_count == 0
        assert enriched[0].quality_issues == []

    def test_does_not_analyze_files_when_reading_cache(self, tmp_path: Path) -> None:
        db, project_id, video, srt = _project_with_video(tmp_path, HALLUCINATION_SRT)
        store_subtitle_quality(
            db, project_id=project_id, video_path=video, subtitle_path=srt
        )
        cache = load_quality_cache(db, project_id)
        fake = CachedSubtitleQuality(
            file_path=str(srt.resolve()),
            video_path=str(video.resolve()),
            issue_count=1,
            issues=cache[str(srt.resolve())].issues,
        )

        with patch(
            "app.services.video_subtitle_quality.analyze_subtitle_file",
            side_effect=AssertionError("listing must not re-analyze"),
        ):
            enriched = enrich_subtitles_with_quality(
                video_path=video,
                project_path=str(video.parent),
                project_name="proj",
                target_language="eng",
                subtitles=[
                    SubtitleInfo(language="eng", embedded=False, path=str(srt)),
                ],
                cache={fake.file_path: fake},
            )

        assert enriched[0].quality_issue_count == 1

    def test_appends_extra_workdir_sub_from_cache(self, tmp_path: Path) -> None:
        db, project_id, video, srt = _project_with_video(tmp_path, CLEAN_SRT)
        extra = video.parent / "generated.en.srt"
        extra.write_text(HALLUCINATION_SRT, encoding="utf-8")
        store_subtitle_quality(
            db, project_id=project_id, video_path=video, subtitle_path=extra
        )

        enriched = enrich_subtitles_with_quality(
            video_path=video,
            project_path=str(video.parent),
            project_name="proj",
            target_language="eng",
            subtitles=[
                SubtitleInfo(language="eng", embedded=False, path=str(srt)),
            ],
            cache=load_quality_cache(db, project_id),
        )

        extra_rows = [
            item for item in enriched if str(item.path).endswith("generated.en.srt")
        ]
        assert len(extra_rows) == 1
        assert extra_rows[0].quality_issue_count >= 1


class TestRefreshProjectSubtitleQuality:
    def test_scan_helper_caches_every_sidecar(self, tmp_path: Path) -> None:
        db, project_id, video, srt = _project_with_video(tmp_path, HALLUCINATION_SRT)
        refresh_project_subtitle_quality(
            db,
            project_id=project_id,
            project_path=str(video.parent),
            project_name="proj",
            target_language="eng",
        )
        hit = load_quality_cache(db, project_id)[str(srt.resolve())]
        assert hit.issue_count >= 1
        assert any(
            issue.rule_id == "known_hallucination_phrase" for issue in hit.issues
        )
