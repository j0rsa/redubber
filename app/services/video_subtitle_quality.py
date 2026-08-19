"""Attach subtitle quality-rule summaries to project file-list rows."""

from __future__ import annotations

from pathlib import Path

from app.schemas.models import SubtitleInfo, SubtitleQualityIssue
from app.services.subtitle_quality_rules import (
    analyze_subtitle_file,
    unique_rule_ids,
)
from utils import detect_subtitle_language


def _resolved_key(path: str | Path) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def issues_for_subtitle_path(
    path: Path,
    cache: dict[str, tuple[int, list[SubtitleQualityIssue]]],
) -> tuple[int, list[SubtitleQualityIssue]]:
    """Return (distinct rule count, all breaches) for ``path``, cached per request."""
    key = _resolved_key(path)
    if key in cache:
        return cache[key]
    analysis = analyze_subtitle_file(path)
    labels = {rule.id: rule.label for rule in analysis.rules}
    issues = [
        SubtitleQualityIssue(
            rule_id=breach.rule_id,
            label=labels.get(breach.rule_id, breach.rule_id),
            message=breach.message,
            segment_index=breach.segment_index,
        )
        for breach in analysis.breaches
    ]
    result = (len(unique_rule_ids(list(analysis.breaches))), issues)
    cache[key] = result
    return result


def enrich_subtitles_with_quality(
    *,
    video_path: Path,
    project_path: str,
    project_name: str,
    target_language: str,
    subtitles: list[SubtitleInfo],
    cache: dict[str, tuple[int, list[SubtitleQualityIssue]]] | None = None,
) -> list[SubtitleInfo]:
    """Copy quality summaries onto each sidecar and any extra reviewable SRT."""
    from app.services.subtitle_review import list_review_srts

    cache = cache if cache is not None else {}
    enriched: list[SubtitleInfo] = []
    seen: set[str] = set()

    for subtitle in subtitles:
        path_str = (subtitle.path or "").strip()
        if not path_str or subtitle.embedded:
            enriched.append(subtitle)
            continue
        path = Path(path_str)
        key = _resolved_key(path)
        seen.add(key)
        count, issues = issues_for_subtitle_path(path, cache)
        enriched.append(
            subtitle.model_copy(
                update={"quality_issue_count": count, "quality_issues": issues}
            )
        )

    if not video_path.is_file():
        return enriched

    for option in list_review_srts(
        video_path, project_path, project_name, target_language
    ):
        path = Path(option.path)
        key = _resolved_key(path)
        if key in seen:
            continue
        count, issues = issues_for_subtitle_path(path, cache)
        if count == 0:
            continue
        language = detect_subtitle_language(path) or target_language or "und"
        enriched.append(
            SubtitleInfo(
                language=language,
                embedded=False,
                path=str(path),
                quality_issue_count=count,
                quality_issues=issues,
            )
        )
        seen.add(key)

    return enriched
