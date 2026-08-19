"""Scan a project folder and persist video / subtitle analysis."""

from __future__ import annotations

from pathlib import Path

from app.services.existing_subtitles import (
    external_subtitle_records,
    stage_target_subtitles_for_videos,
)
from database import DatabaseManager
from file_scanner import FileScanner
from utils import (
    count_videos_in_target_state,
    detect_subtitle_language,
    detect_video_language,
)
from video_analyzer import detect_dominant_language, get_video_info_with_duration


def scan_project_files(
    project_id: int,
    project_path: str,
    db: DatabaseManager,
    scanner: FileScanner,
    *,
    detect_source_language: bool = False,
) -> None:
    """Index videos and same-directory sidecar subtitles for a project.

    Replaces previous file rows so a rescan reflects the current filesystem.
    """
    db.clear_project_files(project_id)
    video_files, subtitle_files = scanner.scan_folder(project_path)

    all_audio_streams: list = []

    for video_file in video_files:
        language = detect_video_language(video_file)
        db.add_video_file(
            project_id=project_id,
            file_path=str(video_file),
            filename=video_file.name,
            language=language,
        )

        video_info = get_video_info_with_duration(video_file)
        audio_streams = video_info["audio_streams"]
        all_audio_streams.append(audio_streams)

        db.save_video_analysis(
            project_id=project_id,
            video_data={
                "filename": video_file.name,
                "path": str(video_file),
                "size_mb": round(video_file.stat().st_size / (1024 * 1024), 2),
                "duration_seconds": video_info["duration_seconds"],
                "audio_streams": audio_streams,
                "subtitles": external_subtitle_records(video_file),
            },
        )

    for subtitle_file in subtitle_files:
        language = detect_subtitle_language(subtitle_file)
        db.add_subtitle_file(
            project_id=project_id,
            file_path=str(subtitle_file),
            filename=subtitle_file.name,
            language=language,
        )

    if detect_source_language and all_audio_streams:
        last_streams = [[streams[-1]] for streams in all_audio_streams if streams]
        dominant_language = detect_dominant_language(last_streams)
        if dominant_language:
            db.set_source_language_override(project_id, dominant_language)

    target_lang = db.get_target_language(project_id)
    video_records = db.get_video_analysis(project_id)
    replaced = count_videos_in_target_state(video_records, target_lang)
    db.update_project_video_counts(project_id, len(video_files), replaced)

    project = db.get_project_by_id(project_id)
    if project:
        stage_target_subtitles_for_videos(
            video_files,
            project_path=project_path,
            project_name=project["name"],
            target_language=target_lang,
        )
