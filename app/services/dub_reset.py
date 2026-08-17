"""Reset a finalized redub: drop the dubbed audio track and generated subtitles."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from app.core.project_paths import get_project_working_dir
from database import DatabaseManager
from utils import (
    count_videos_in_target_state,
    detect_subtitle_language,
    is_video_in_target_state,
    normalize_lang_code,
)

log = logging.getLogger(__name__)

SUBTITLE_EXTS = {".srt", ".vtt", ".ass", ".ssa", ".sub", ".sbv"}


class DubResetError(Exception):
    """Raised when a dub reset cannot be performed."""


def find_generated_subtitle_paths(video_path: Path, target_language: str) -> list[Path]:
    """Return sidecar subtitle files next to the video that match the target language.

    Matches ``video.en.srt`` / ``video.eng.srt`` style names produced by
    finalization. Leaves original-language sidecars (e.g. ``video.ru.srt``)
    and unsuffixed ``video.srt`` files alone.
    """
    target = normalize_lang_code(target_language)
    if not target or not video_path.exists():
        return []

    parent = video_path.parent
    stem = video_path.stem
    found: list[Path] = []
    if not parent.is_dir():
        return found

    for candidate in sorted(parent.iterdir()):
        if not candidate.is_file() or candidate.suffix.lower() not in SUBTITLE_EXTS:
            continue
        if candidate.stem != stem and not candidate.stem.startswith(stem + "."):
            continue
        detected = detect_subtitle_language(candidate)
        if not detected:
            continue
        if normalize_lang_code(detected) == target:
            found.append(candidate)
    return found


def clear_finalization_artifacts(
    video_path: Path, project_path: str, project_name: str
) -> list[str]:
    """Remove backup and mixed-output files so pipeline status no longer shows replaced.

    After stripping the dubbed track the video file is no longer in the target
    state, but leftover ``.dubbed`` files or finalize backups would still make
    ``get_pipeline_status`` report ``replaced=True``.
    """
    removed: list[str] = []
    working_root = get_project_working_dir(project_path, project_name)
    video_stem = video_path.stem
    video_ext = video_path.suffix

    rel = os.path.relpath(str(video_path), project_path)
    per_video_dir = working_root / rel
    dubbed = per_video_dir / f"{video_stem}.dubbed{video_ext}"
    if dubbed.is_file():
        dubbed.unlink()
        removed.append(str(dubbed))

    backup_dir = working_root / "backups"
    if backup_dir.is_dir():
        for candidate in sorted(backup_dir.iterdir()):
            if candidate.is_file() and candidate.name.startswith(video_stem + "."):
                candidate.unlink()
                removed.append(str(candidate))
    return removed


def working_dir_subtitle_path(
    video_path: Path, project_path: str, project_name: str
) -> Path:
    """Path to the pipeline-generated SRT under ``03_subtitles/`` (may not exist)."""
    working_dir = get_project_working_dir(project_path, project_name)
    rel = os.path.relpath(str(video_path), project_path)
    return working_dir / rel / "03_subtitles" / f"{video_path.stem}.en.srt"


def strip_first_audio_track(video_path: Path) -> None:
    """Remux ``video_path`` without its first audio stream (the dubbed track).

    Writes to a sibling temp file, then atomically replaces the original.
    Video, remaining audio, and any other streams are copied without re-encoding.
    """
    tmp_path = video_path.with_name(f".{video_path.stem}.undub{video_path.suffix}")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0",
        "-map",
        "-0:a:0",
        "-c",
        "copy",
        str(tmp_path),
    ]
    log.info("Stripping first audio track: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise DubResetError(
            f"ffmpeg failed to strip dubbed audio track: {result.stderr[-2000:]}"
        )
    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise DubResetError("ffmpeg produced an empty output while stripping audio")
    os.replace(tmp_path, video_path)


_RESET_REJECTED_MSG = (
    "Video is not in the final redubbed state. "
    "Reset is only allowed when the file has a dubbed audio track "
    "and a generated subtitle in the project target language."
)


def reconcile_video_with_disk(
    db: DatabaseManager,
    project_id: int,
    video_record: dict,
    project_path: str,
    project_name: str,
    target_language: str,
) -> dict:
    """Align DB metadata and working-dir artifacts with the on-disk video file.

    When the video file is no longer in the final redubbed state but leftover
    backup/dubbed files or stale subtitle rows still imply completion, remove
    those artifacts and refresh analysis from disk.
    """
    from pipeline_status import get_pipeline_status
    from redubber import sync_video_metadata

    video_path = Path(video_record["file_path"])
    if not video_path.exists():
        return {"reconciled": False, "fixes": []}

    fixes: list[str] = []
    working_root = str(get_project_working_dir(project_path, project_name))
    target = normalize_lang_code(target_language)

    sync_video_metadata(db, project_id, str(video_path))
    fixes.append("refreshed video metadata from disk")

    records = db.get_video_analysis(project_id)
    record = next((r for r in records if r["id"] == video_record["id"]), video_record)
    audio_streams = record.get("audio_streams") or []
    subtitles = record.get("subtitle_matches") or []

    if is_video_in_target_state(audio_streams, subtitles, target_language):
        replaced = count_videos_in_target_state(records, target_language)
        db.update_project_video_counts(project_id, len(records), replaced)
        return {"reconciled": bool(fixes), "fixes": fixes}

    pipeline = get_pipeline_status(str(video_path), project_path, working_root)
    if pipeline.replaced or pipeline.final_file_exists:
        removed = clear_finalization_artifacts(video_path, project_path, project_name)
        fixes.extend(removed)

    for sub in db.get_subtitle_files_for_video(project_id, video_path.name):
        sub_path = sub.get("file_path") or ""
        sub_lang = normalize_lang_code(sub.get("language") or "")
        if target and sub_lang == target and not Path(sub_path).exists():
            db.delete_subtitle_file(project_id, sub_path)
            fixes.append(f"removed stale subtitle record: {sub_path}")

    records = db.get_video_analysis(project_id)
    replaced = count_videos_in_target_state(records, target_language)
    db.update_project_video_counts(project_id, len(records), replaced)

    return {"reconciled": bool(fixes), "fixes": fixes}


def reset_dubbed_video(
    db: DatabaseManager,
    project_id: int,
    video_record: dict,
    project_path: str,
    project_name: str,
    target_language: str,
) -> dict:
    """Delete generated target-language subtitles and strip the first audio track.

    Only valid for videos already in the target (final) redubbed state.

    Returns:
        Dict with ``video_path``, ``removed_audio_track``, and ``deleted_subtitles``.
    """
    video_path = Path(video_record["file_path"])
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    audio_streams = video_record.get("audio_streams") or []
    subtitles = video_record.get("subtitle_matches") or []
    if not is_video_in_target_state(audio_streams, subtitles, target_language):
        reconcile_video_with_disk(
            db,
            project_id,
            video_record,
            project_path,
            project_name,
            target_language,
        )
        raise DubResetError(_RESET_REJECTED_MSG)

    if len(audio_streams) < 2:
        raise DubResetError(
            "Video does not have a separate dubbed audio track to remove"
        )

    deleted: list[str] = []
    for sub_path in find_generated_subtitle_paths(video_path, target_language):
        try:
            sub_path.unlink()
            deleted.append(str(sub_path))
            db.delete_subtitle_file(project_id, str(sub_path))
        except OSError as exc:
            log.warning("Could not delete subtitle %s: %s", sub_path, exc)

    working_srt = working_dir_subtitle_path(video_path, project_path, project_name)
    if working_srt.exists():
        try:
            working_srt.unlink()
            deleted.append(str(working_srt))
        except OSError as exc:
            log.warning(
                "Could not delete working-dir subtitle %s: %s", working_srt, exc
            )

    strip_first_audio_track(video_path)
    clear_finalization_artifacts(video_path, project_path, project_name)

    from redubber import sync_video_metadata

    sync_video_metadata(db, project_id, str(video_path))

    records = db.get_video_analysis(project_id)
    replaced = count_videos_in_target_state(records, target_language)
    db.update_project_video_counts(project_id, len(records), replaced)

    return {
        "status": "reset",
        "path": str(video_path),
        "removed_audio_track": True,
        "deleted_subtitles": deleted,
    }
