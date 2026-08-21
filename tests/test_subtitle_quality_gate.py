"""Tests for the generated-subtitle quality hold policy."""

import pytest
from pydantic import ValidationError

from app.infrastructure.task_queue import TaskQueueManager, TaskStatus
from app.schemas.models import TaskCreate
from app.services.subtitle_quality_gate import should_pause_for_subtitle_review
from app.services.subtitle_quality_hold import (
    clear_subtitle_quality_hold,
    hold_marker_for_root,
    read_subtitle_quality_hold,
    write_subtitle_quality_hold,
)


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


@pytest.mark.asyncio
async def test_hold_status_keeps_recovery_metadata() -> None:
    manager = TaskQueueManager()
    task = TaskStatus(
        task_id="held",
        video_path="/tmp/video.mp4",
        stage="running",
        progress=35,
        status="running",
    )
    manager._tasks[task.task_id] = task

    await manager._update_task_status(
        task.task_id,
        stage="Subtitle review required",
        progress=38,
        status="awaiting_subtitle_review",
        subtitle_path="/tmp/video.en.srt",
        quality_issue_count=1,
        quality_issues=(
            {
                "rule_id": "known_hallucination_phrase",
                "label": "Known STT phrase",
                "message": "known phrase",
                "segment_index": 0,
            },
        ),
    )

    held = await manager.get_status(task.task_id)
    assert held is not None
    assert held.status == "awaiting_subtitle_review"
    assert held.subtitle_path == "/tmp/video.en.srt"
    assert held.quality_issue_count == 1
    assert held.quality_issues[0]["rule_id"] == "known_hallucination_phrase"


def test_hold_marker_survives_process_state(tmp_path) -> None:
    write_subtitle_quality_hold(
        root=tmp_path,
        subtitle_path="/tmp/video.en.srt",
        quality_issue_count=1,
        quality_issues=(
            {
                "rule_id": "known_hallucination_phrase",
                "label": "Known STT phrase",
                "message": "known phrase",
                "segment_index": 0,
            },
        ),
    )

    marker = hold_marker_for_root(tmp_path)
    persisted = read_subtitle_quality_hold(marker)
    assert persisted is not None
    assert persisted["quality_issue_count"] == 1
    assert persisted["subtitle_path"] == "/tmp/video.en.srt"

    clear_subtitle_quality_hold(tmp_path)
    assert not marker.exists()
