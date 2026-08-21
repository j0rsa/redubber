"""Policy for pausing generated subtitles before TTS."""

from __future__ import annotations


def should_pause_for_subtitle_review(
    *,
    generated_in_task: bool,
    quality_issue_count: int,
    ignore_warnings: bool,
) -> bool:
    """Return whether TTS must wait for explicit subtitle review.

    Existing sidecars are always advisory. Only subtitles generated from STT in
    the current pipeline are eligible for this hold.
    """
    return generated_in_task and quality_issue_count > 0 and not ignore_warnings
