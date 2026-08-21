"""Attach subtitle quality-rule summaries to project file-list rows.

Quality analysis is expensive (parse SRT + run every heuristic). Results are
persisted in ``subtitle_quality`` and only recomputed on scan, subtitle
generation, or cue edits — never while listing videos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.schemas.models import SubtitleInfo, SubtitleQualityIssue
from app.services.subtitle_quality_rules import (
    analyze_subtitle_file,
    unique_rule_ids,
)
from database import DatabaseManager
from stt_hallucination import HallucinationConfig, resolve_hallucination_config
from utils import detect_subtitle_language


def _resolved_key(path: str | Path) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


@dataclass(frozen=True)
class CachedSubtitleQuality:
    """One persisted quality analysis, keyed by resolved subtitle path."""

    file_path: str
    video_path: str
    issue_count: int
    issues: list[SubtitleQualityIssue]


def _issues_from_analysis(path: Path, config: HallucinationConfig | None) -> tuple[int, list[SubtitleQualityIssue]]:
    analysis = analyze_subtitle_file(path, config=config)
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
    return len(unique_rule_ids(list(analysis.breaches))), issues


def issues_for_subtitle_path(
    path: Path,
    cache: dict[str, tuple[int, list[SubtitleQualityIssue]]],
    config: HallucinationConfig | None = None,
) -> tuple[int, list[SubtitleQualityIssue]]:
    """Return (distinct rule count, all breaches) for ``path``.

    ``cache`` is an in-process memoizer used while writing the DB cache,
    not a substitute for persisted ``subtitle_quality`` rows.
    """
    key = _resolved_key(path)
    if key in cache:
        return cache[key]
    result = _issues_from_analysis(path, config)
    cache[key] = result
    return result


def load_quality_cache(
    db: DatabaseManager, project_id: int
) -> dict[str, CachedSubtitleQuality]:
    """Load persisted quality rows for a project, keyed by resolved file path."""
    cached: dict[str, CachedSubtitleQuality] = {}
    for row in db.list_subtitle_quality(project_id):
        file_path = _resolved_key(row.get("file_path") or "")
        if not file_path:
            continue
        raw_issues = row.get("quality_issues") or []
        issues: list[SubtitleQualityIssue] = []
        if isinstance(raw_issues, list):
            for item in raw_issues:
                if not isinstance(item, dict):
                    continue
                try:
                    issues.append(SubtitleQualityIssue(**item))
                except (TypeError, ValueError):
                    continue
        cached[file_path] = CachedSubtitleQuality(
            file_path=file_path,
            video_path=_resolved_key(row.get("video_path") or ""),
            issue_count=int(row.get("quality_issue_count") or 0),
            issues=issues,
        )
    return cached


def store_subtitle_quality(
    db: DatabaseManager,
    *,
    project_id: int,
    video_path: str | Path,
    subtitle_path: str | Path,
    config: HallucinationConfig | None = None,
    memo: dict[str, tuple[int, list[SubtitleQualityIssue]]] | None = None,
) -> tuple[int, list[SubtitleQualityIssue]]:
    """Analyze one subtitle file and persist the result."""
    path = Path(subtitle_path)
    memo = memo if memo is not None else {}
    count, issues = issues_for_subtitle_path(path, memo, config)
    db.upsert_subtitle_quality(
        project_id=project_id,
        video_path=_resolved_key(video_path),
        file_path=_resolved_key(path),
        quality_issue_count=count,
        quality_issues=[issue.model_dump() for issue in issues],
    )
    return count, issues


def _sidecar_paths(subtitles: list[SubtitleInfo] | list[dict] | None) -> list[Path]:
    paths: list[Path] = []
    for subtitle in subtitles or []:
        if isinstance(subtitle, SubtitleInfo):
            if subtitle.embedded:
                continue
            path_str = (subtitle.path or "").strip()
        else:
            if subtitle.get("embedded"):
                continue
            path_str = str(subtitle.get("path") or "").strip()
        if path_str:
            paths.append(Path(path_str))
    return paths


def refresh_subtitle_quality_for_paths(
    db: DatabaseManager,
    *,
    project_id: int,
    video_path: str | Path,
    paths: list[str | Path],
    config: HallucinationConfig | None = None,
    memo: dict[str, tuple[int, list[SubtitleQualityIssue]]] | None = None,
) -> None:
    """Re-analyze and persist quality for the given subtitle paths."""
    resolved_config = resolve_hallucination_config(config)
    memo = memo if memo is not None else {}
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw)
        key = _resolved_key(path)
        if not key or key in seen:
            continue
        seen.add(key)
        store_subtitle_quality(
            db,
            project_id=project_id,
            video_path=video_path,
            subtitle_path=path,
            config=resolved_config,
            memo=memo,
        )


def refresh_subtitle_quality_for_video(
    db: DatabaseManager,
    *,
    project_id: int,
    video_path: str | Path,
    project_path: str,
    project_name: str,
    target_language: str,
    subtitles: list[SubtitleInfo] | list[dict] | None = None,
    config: HallucinationConfig | None = None,
    memo: dict[str, tuple[int, list[SubtitleQualityIssue]]] | None = None,
) -> None:
    """Discover sidecar + workdir SRTs for one video and cache their analysis."""
    from app.services.subtitle_review import list_review_srts

    resolved_config = resolve_hallucination_config(config)
    memo = memo if memo is not None else {}
    video = Path(video_path)
    paths: list[Path] = _sidecar_paths(subtitles)
    if video.is_file():
        for option in list_review_srts(
            video, project_path, project_name, target_language
        ):
            paths.append(Path(option.path))
    refresh_subtitle_quality_for_paths(
        db,
        project_id=project_id,
        video_path=video,
        paths=paths,
        config=resolved_config,
        memo=memo,
    )


def refresh_project_subtitle_quality(
    db: DatabaseManager,
    *,
    project_id: int,
    project_path: str,
    project_name: str,
    target_language: str,
    video_records: list[dict] | None = None,
) -> dict[str, CachedSubtitleQuality]:
    """Recompute quality for every video in a project (used by file scan)."""
    config = resolve_hallucination_config()
    memo: dict[str, tuple[int, list[SubtitleQualityIssue]]] = {}
    records = (
        video_records
        if video_records is not None
        else db.get_video_analysis(project_id)
    )
    for record in records:
        refresh_subtitle_quality_for_video(
            db,
            project_id=project_id,
            video_path=record["file_path"],
            project_path=project_path,
            project_name=project_name,
            target_language=target_language,
            subtitles=record.get("subtitle_matches") or [],
            config=config,
            memo=memo,
        )
    return load_quality_cache(db, project_id)


def _project_has_external_subs(video_records: list[dict]) -> bool:
    for record in video_records:
        for sub in record.get("subtitle_matches") or []:
            path = str(sub.get("path") or "").strip()
            if path and not sub.get("embedded"):
                return True
    return False


def quality_cache_for_listing(
    db: DatabaseManager,
    *,
    project_id: int,
    project_path: str,
    project_name: str,
    target_language: str,
    video_records: list[dict],
) -> dict[str, CachedSubtitleQuality]:
    """Return the DB cache, backfilling once if this project has never been analyzed."""
    cached = load_quality_cache(db, project_id)
    if cached or not _project_has_external_subs(video_records):
        return cached
    return refresh_project_subtitle_quality(
        db,
        project_id=project_id,
        project_path=project_path,
        project_name=project_name,
        target_language=target_language,
        video_records=video_records,
    )


def enrich_subtitles_with_quality(
    *,
    video_path: Path,
    project_path: str,
    project_name: str,
    target_language: str,
    subtitles: list[SubtitleInfo],
    cache: dict[str, CachedSubtitleQuality] | None = None,
) -> list[SubtitleInfo]:
    """Copy cached quality summaries onto each sidecar and extra reviewable SRT.

    Does not read subtitle files or run heuristics. ``project_path`` and
    ``project_name`` are kept for call-site compatibility.
    """
    del project_path, project_name
    cached = cache if cache is not None else {}
    enriched: list[SubtitleInfo] = []
    seen: set[str] = set()
    video_key = _resolved_key(video_path)

    for subtitle in subtitles:
        path_str = (subtitle.path or "").strip()
        if not path_str or subtitle.embedded:
            enriched.append(subtitle)
            continue
        key = _resolved_key(path_str)
        seen.add(key)
        hit = cached.get(key)
        if hit is None:
            enriched.append(subtitle)
            continue
        enriched.append(
            subtitle.model_copy(
                update={
                    "quality_issue_count": hit.issue_count,
                    "quality_issues": hit.issues,
                }
            )
        )

    for hit in cached.values():
        if hit.file_path in seen:
            continue
        if hit.video_path != video_key:
            continue
        if hit.issue_count == 0:
            continue
        path = Path(hit.file_path)
        language = detect_subtitle_language(path) or target_language or "und"
        enriched.append(
            SubtitleInfo(
                language=language,
                embedded=False,
                path=hit.file_path,
                quality_issue_count=hit.issue_count,
                quality_issues=hit.issues,
            )
        )
        seen.add(hit.file_path)

    return enriched
