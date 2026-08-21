"""Persist generated-subtitle quality holds beside pipeline artefacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.core.project_paths import get_project_working_dir

HOLD_MARKER_NAME = "subtitle_quality_hold.json"


def hold_marker_for_root(root: str | Path) -> Path:
    return Path(root) / HOLD_MARKER_NAME


def hold_marker_for_video(
    video_path: str | Path,
    project_path: str,
    project_name: str,
) -> Path:
    video = Path(video_path)
    working_root = get_project_working_dir(project_path, project_name)
    relative = os.path.relpath(str(video), project_path)
    return hold_marker_for_root(working_root / relative)


def write_subtitle_quality_hold(
    *,
    root: str | Path,
    subtitle_path: str,
    quality_issue_count: int,
    quality_issues: tuple[dict[str, object], ...],
) -> None:
    marker = hold_marker_for_root(root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "subtitle_path": subtitle_path,
                "quality_issue_count": quality_issue_count,
                "quality_issues": list(quality_issues),
            }
        ),
        encoding="utf-8",
    )


def read_subtitle_quality_hold(marker: Path) -> dict | None:
    if not marker.is_file():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def clear_subtitle_quality_hold(root: str | Path) -> None:
    hold_marker_for_root(root).unlink(missing_ok=True)


def clear_subtitle_quality_hold_for_video(
    video_path: str | Path,
    project_path: str,
    project_name: str,
) -> None:
    hold_marker_for_video(video_path, project_path, project_name).unlink(
        missing_ok=True
    )
