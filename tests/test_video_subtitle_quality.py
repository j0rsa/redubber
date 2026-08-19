"""Tests for attaching quality-rule summaries to project file-list subtitles."""

from __future__ import annotations

from pathlib import Path

from app.schemas.models import SubtitleInfo
from app.services.subtitle_quality_rules import analyze_subtitle_file
from app.services.video_subtitle_quality import enrich_subtitles_with_quality

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


class TestAnalyzeSubtitleFile:
    def test_detects_known_phrase_on_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "lesson.en.srt"
        path.write_text(HALLUCINATION_SRT, encoding="utf-8")
        analysis = analyze_subtitle_file(path)
        assert any(b.rule_id == "known_hallucination_phrase" for b in analysis.breaches)

    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        analysis = analyze_subtitle_file(tmp_path / "missing.srt")
        assert analysis.breaches == ()


class TestEnrichSubtitlesWithQuality:
    def test_attaches_issues_to_matching_sidecar(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        srt = project / "lesson.en.srt"
        srt.write_text(HALLUCINATION_SRT, encoding="utf-8")

        enriched = enrich_subtitles_with_quality(
            video_path=video,
            project_path=str(project),
            project_name="proj",
            target_language="eng",
            subtitles=[
                SubtitleInfo(language="eng", embedded=False, path=str(srt)),
            ],
        )

        assert len(enriched) == 1
        assert enriched[0].quality_issue_count >= 1
        assert any(
            issue.rule_id == "known_hallucination_phrase"
            for issue in enriched[0].quality_issues
        )

    def test_clean_sidecar_has_zero_issues(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"x")
        srt = project / "lesson.en.srt"
        srt.write_text(CLEAN_SRT, encoding="utf-8")

        enriched = enrich_subtitles_with_quality(
            video_path=video,
            project_path=str(project),
            project_name="proj",
            target_language="eng",
            subtitles=[
                SubtitleInfo(language="eng", embedded=False, path=str(srt)),
            ],
        )

        assert enriched[0].quality_issue_count == 0
        assert enriched[0].quality_issues == []
