"""Tests for the generated-subtitle quality hold policy."""

import pytest
from pydantic import ValidationError

from app.schemas.models import TaskCreate
from app.services.subtitle_quality_gate import should_pause_for_subtitle_review


def test_generated_subtitle_warnings_pause_pipeline() -> None:
    assert should_pause_for_subtitle_review(
        generated_in_task=True,
        quality_issue_count=2,
        ignore_warnings=False,
    )


def test_existing_subtitle_warnings_remain_advisory() -> None:
    assert not should_pause_for_subtitle_review(
        generated_in_task=False,
        quality_issue_count=2,
        ignore_warnings=False,
    )


def test_explicit_ignore_allows_generated_subtitle_to_continue() -> None:
    assert not should_pause_for_subtitle_review(
        generated_in_task=True,
        quality_issue_count=2,
        ignore_warnings=True,
    )


def test_clean_generated_subtitle_continues() -> None:
    assert not should_pause_for_subtitle_review(
        generated_in_task=True,
        quality_issue_count=0,
        ignore_warnings=False,
    )


def test_ignore_requires_subtitle_resume() -> None:
    with pytest.raises(ValidationError, match="requires resume_from_subtitles"):
        TaskCreate(
            video_path="/tmp/video.mp4",
            project_id=1,
            ignore_subtitle_warnings=True,
        )
