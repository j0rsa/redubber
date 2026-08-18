"""Reset a finalized redub: drop the dubbed audio track and generated subtitles."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from app.core.project_paths import get_project_working_dir
from app.services.video_metadata import sync_video_metadata
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
    """Return sidecar subtitle files next to the video that match the target language."""
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
    """Remove backup and mixed-output files so pipeline status no longer shows replaced."""
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


def probe_audio_streams(video_path: Path) -> list[dict]:
    """Return audio streams from the file on disk (ffprobe, not DB)."""
    from video_analyzer import get_video_info_with_duration

    info = get_video_info_with_duration(video_path)
    return list(info.get("audio_streams") or [])


def identify_dubbed_stream_index(
    audio_streams: list[dict],
    target_language: str,
    source_language: str | None = None,
) -> int:
    """Return the global ffprobe stream index of the dubbed (target-language) track.

    Raises DubResetError when the dubbed track cannot be identified unambiguously.
    """
    target = normalize_lang_code(target_language)
    if not target:
        raise DubResetError("Project target language is not configured.")

    if len(audio_streams) < 2:
        raise DubResetError(
            "Video must have at least two audio tracks to remove a dub safely."
        )

    unknown_langs = [
        s for s in audio_streams
        if normalize_lang_code(s.get("language")) in ("", "unknown", "und")
    ]
    if unknown_langs:
        raise DubResetError(
            "Audio tracks are missing language tags — cannot safely identify "
            "which track is the dub. Restore from backup or re-tag the file manually."
        )

    target_matches = [
        s for s in audio_streams
        if normalize_lang_code(s.get("language")) == target
    ]
    if not target_matches:
        raise DubResetError(
            f"No audio track is tagged with the project target language ({target_language}). "
            "Refusing to remove a track by position alone."
        )
    if len(target_matches) > 1:
        raise DubResetError(
            f"Multiple audio tracks are tagged with target language {target_language}. "
            "Manual cleanup is required."
        )

    dubbed_stream = target_matches[0]
    dubbed_index = int(dubbed_stream["index"])

    if source_language:
        source = normalize_lang_code(source_language)
        if source:
            source_matches = [
                s for s in audio_streams
                if normalize_lang_code(s.get("language")) == source
            ]
            if not source_matches:
                raise DubResetError(
                    f"No audio track is tagged with the source language ({source_language}). "
                    "Refusing to strip a track until the original can be verified."
                )
            if all(int(s["index"]) == dubbed_index for s in source_matches):
                raise DubResetError(
                    "The only source-language track matches the dub track — refusing to strip."
                )
            remaining_source = [
                s for s in source_matches if int(s["index"]) != dubbed_index
            ]
            if not remaining_source:
                raise DubResetError(
                    "Removing the dubbed track would leave no source-language audio."
                )

    return dubbed_index


def backup_video_before_reset(
    video_path: Path, project_path: str, project_name: str
) -> Path:
    """Copy the video to the project working-dir backups folder before mutating it."""
    backup_dir = get_project_working_dir(project_path, project_name) / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"{video_path.stem}.pre-undub.{stamp}{video_path.suffix}"
    shutil.copy2(video_path, dest)
    log.info("Created pre-undub backup at %s", dest)
    return dest


def strip_dubbed_audio_track(
    video_path: Path,
    target_language: str,
    source_language: str | None = None,
) -> int:
    """Remux ``video_path`` without the target-language (dubbed) audio stream.

    Returns:
        The global stream index that was removed.
    """
    streams = probe_audio_streams(video_path)
    stream_index = identify_dubbed_stream_index(
        streams, target_language, source_language
    )

    tmp_path = video_path.with_name(f".{video_path.stem}.undub{video_path.suffix}")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0",
        "-map",
        f"-0:{stream_index}",
        "-c",
        "copy",
        str(tmp_path),
    ]
    log.info("Stripping dubbed audio stream %s: %s", stream_index, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise DubResetError(
            f"ffmpeg failed to strip dubbed audio track: {result.stderr[-2000:]}"
        )
    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise DubResetError("ffmpeg produced an empty output while stripping audio")

    remaining = probe_audio_streams(tmp_path)
    if len(remaining) < 1:
        tmp_path.unlink(missing_ok=True)
        raise DubResetError(
            "Stripping the dubbed track would leave the video with no audio."
        )
    if normalize_lang_code(target_language) in {
        normalize_lang_code(s.get("language")) for s in remaining
    } and len(remaining) == 1:
        tmp_path.unlink(missing_ok=True)
        raise DubResetError(
            "Output still contains only target-language audio — aborting."
        )

    os.replace(tmp_path, video_path)
    return stream_index


# Backward-compatible alias used in older tests/callers.
def strip_first_audio_track(video_path: Path) -> None:
    """Deprecated: strips by position only. Prefer ``strip_dubbed_audio_track``."""
    raise DubResetError(
        "Unsafe track removal by position is disabled. "
        "Use language-verified dub removal instead."
    )


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
    """Align DB metadata and working-dir artifacts with the on-disk video file."""
    from pipeline_status import get_pipeline_status

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

    pipeline = get_pipeline_status(
        str(video_path),
        project_path,
        working_root,
        target_language=target_language,
    )
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
    source_language: str | None = None,
) -> dict:
    """Delete generated target-language subtitles and strip the dubbed audio track."""
    video_path = Path(video_record["file_path"])
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Always refresh DB from disk before making safety decisions.
    sync_video_metadata(db, project_id, str(video_path))
    records = db.get_video_analysis(project_id)
    record = next((r for r in records if r["id"] == video_record["id"]), video_record)

    audio_streams = record.get("audio_streams") or []
    subtitles = record.get("subtitle_matches") or []

    disk_streams = probe_audio_streams(video_path)
    if len(disk_streams) < 2:
        reconcile_video_with_disk(
            db, project_id, record, project_path, project_name, target_language,
        )
        raise DubResetError(
            "Video file on disk does not have two audio tracks. "
            "If a previous remove-dub job failed partway, restore from "
            "`.redubber/backups/` and scan the project."
        )

    if not is_video_in_target_state(audio_streams, subtitles, target_language):
        reconcile_video_with_disk(
            db, project_id, record, project_path, project_name, target_language,
        )
        raise DubResetError(_RESET_REJECTED_MSG)

    # Validate dubbed track identity on disk before touching the file.
    identify_dubbed_stream_index(disk_streams, target_language, source_language)

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

    backup_path = backup_video_before_reset(video_path, project_path, project_name)

    try:
        removed_index = strip_dubbed_audio_track(
            video_path, target_language, source_language
        )
    except DubResetError:
        raise
    except Exception as exc:
        raise DubResetError(f"Dub removal failed: {exc}") from exc

    clear_finalization_artifacts(video_path, project_path, project_name)
    sync_video_metadata(db, project_id, str(video_path))

    records = db.get_video_analysis(project_id)
    replaced = count_videos_in_target_state(records, target_language)
    db.update_project_video_counts(project_id, len(records), replaced)

    return {
        "status": "reset",
        "path": str(video_path),
        "removed_audio_track": True,
        "removed_stream_index": removed_index,
        "backup_path": str(backup_path),
        "deleted_subtitles": deleted,
    }
