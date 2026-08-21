"""Sync on-disk video file metadata into the database."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)


def sync_video_metadata(db, project_id: int, video_path: str) -> None:
    """Re-detect and sync video metadata after redubbing or dub removal.

    Updates video language, audio streams, subtitles, and project timestamps.
    """
    from utils import detect_video_language
    from video_analyzer import get_video_info_with_duration

    log.info("Syncing metadata for video: %s", video_path)

    detected_lang = detect_video_language(Path(video_path))
    log.info("Detected language: %s", detected_lang)

    video_info = get_video_info_with_duration(Path(video_path))
    audio_streams = video_info["audio_streams"]
    duration_seconds = video_info["duration_seconds"]
    size_mb = round(Path(video_path).stat().st_size / (1024 * 1024), 2)
    log.info(
        "Audio streams: %s, duration: %.2fs, size: %s MB",
        audio_streams,
        duration_seconds,
        size_mb,
    )

    from app.services.existing_subtitles import external_subtitle_records

    subtitle_matches = external_subtitle_records(Path(video_path))
    for sub in subtitle_matches:
        db.add_subtitle_file(
            project_id=project_id,
            file_path=sub["path"],
            filename=sub["filename"],
            language=sub["language"] or None,
        )
    log.info("Subtitle files: %s", [s["filename"] for s in subtitle_matches])

    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE video_files
            SET language = ?
            WHERE project_id = ? AND file_path = ?
        """,
            (detected_lang, project_id, video_path),
        )
        cursor.execute(
            """
            UPDATE video_analysis
            SET audio_streams = ?, duration_seconds = ?, size_mb = ?, subtitle_matches = ?
            WHERE project_id = ? AND file_path = ?
        """,
            (
                json.dumps(audio_streams),
                duration_seconds,
                size_mb,
                json.dumps(subtitle_matches),
                project_id,
                video_path,
            ),
        )
        cursor.execute(
            """
            UPDATE projects
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """,
            (project_id,),
        )
        conn.commit()

    db.refresh_project_duration_size(project_id)

    project = db.get_project_by_id(project_id)
    if project:
        from app.services.video_subtitle_quality import refresh_subtitle_quality_for_video

        refresh_subtitle_quality_for_video(
            db,
            project_id=project_id,
            video_path=video_path,
            project_path=project["path"],
            project_name=project["name"],
            target_language=project.get("target_language") or "eng",
            subtitles=subtitle_matches,
        )

    log.info("Metadata sync complete")
