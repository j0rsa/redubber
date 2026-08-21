"""API endpoints for reviewing generated subtitles against source chunks and TTS."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from app.core.dependencies import get_db
from app.schemas.subtitle_review import SubtitleCueUpdateRequest, SubtitleReviewResponse
from app.services.subtitle_review import (
    SubtitleReviewError,
    artefact_dirs,
    build_subtitle_review,
    delete_subtitle_cue,
    is_safe_chunk_name,
    tts_file_for_index,
    update_subtitle_cue_text,
)
from app.services.video_subtitle_quality import refresh_subtitle_quality_for_video
from database import DatabaseManager

router = APIRouter()


def _video_context(
    project_id: int, video_id: int, db: DatabaseManager
) -> tuple[dict, dict]:
    project = db.get_project_by_id(project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    record = next(
        (row for row in db.get_video_analysis(project_id) if row["id"] == video_id),
        None,
    )
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id} not found",
        )
    return project, record


@router.get(
    "/projects/{project_id}/videos/{video_id}/subtitle-review",
    response_model=SubtitleReviewResponse,
)
async def get_subtitle_review(
    project_id: int,
    video_id: int,
    db: Annotated[DatabaseManager, Depends(get_db)],
    min_duration: Annotated[float, Query(ge=0.0)] = 0.0,
    max_duration: Annotated[float, Query(ge=0.0)] = 0.0,
    srt_path: Annotated[str | None, Query()] = None,
) -> SubtitleReviewResponse:
    """Return the generated subtitle script with original-chunk and TTS playback URLs."""
    if min_duration and max_duration and min_duration > max_duration:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="min_duration must not exceed max_duration",
        )

    project, record = _video_context(project_id, video_id, db)
    try:
        return build_subtitle_review(
            project_id=project_id,
            video_id=video_id,
            video_path=Path(record["file_path"]),
            filename=record["filename"],
            project_path=project["path"],
            project_name=project["name"],
            target_language=project.get("target_language") or "eng",
            min_duration=min_duration,
            max_duration=max_duration,
            srt_path=srt_path,
        )
    except SubtitleReviewError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.patch(
    "/projects/{project_id}/videos/{video_id}/subtitle-review/cues/{cue_index}",
    response_model=SubtitleReviewResponse,
)
async def patch_subtitle_cue(
    project_id: int,
    video_id: int,
    cue_index: int,
    payload: SubtitleCueUpdateRequest,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> SubtitleReviewResponse:
    """Update one cue's text. Timings and the number of cues are unchanged."""
    project, record = _video_context(project_id, video_id, db)
    try:
        update_subtitle_cue_text(
            video_path=Path(record["file_path"]),
            project_path=project["path"],
            project_name=project["name"],
            target_language=project.get("target_language") or "eng",
            cue_index=cue_index,
            text=payload.text,
            srt_path=payload.srt_path,
        )
        refresh_subtitle_quality_for_video(
            db,
            project_id=project_id,
            video_path=record["file_path"],
            project_path=project["path"],
            project_name=project["name"],
            target_language=project.get("target_language") or "eng",
            subtitles=record.get("subtitle_matches") or [],
        )
        return build_subtitle_review(
            project_id=project_id,
            video_id=video_id,
            video_path=Path(record["file_path"]),
            filename=record["filename"],
            project_path=project["path"],
            project_name=project["name"],
            target_language=project.get("target_language") or "eng",
            srt_path=payload.srt_path,
        )
    except SubtitleReviewError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.delete(
    "/projects/{project_id}/videos/{video_id}/subtitle-review/cues/{cue_index}",
    response_model=SubtitleReviewResponse,
)
async def delete_subtitle_review_cue(
    project_id: int,
    video_id: int,
    cue_index: int,
    db: Annotated[DatabaseManager, Depends(get_db)],
    srt_path: Annotated[str | None, Query()] = None,
) -> SubtitleReviewResponse:
    """Delete one subtitle cue and return the updated review."""
    project, record = _video_context(project_id, video_id, db)
    try:
        delete_subtitle_cue(
            video_path=Path(record["file_path"]),
            project_path=project["path"],
            project_name=project["name"],
            target_language=project.get("target_language") or "eng",
            cue_index=cue_index,
            srt_path=srt_path,
        )
        refresh_subtitle_quality_for_video(
            db,
            project_id=project_id,
            video_path=record["file_path"],
            project_path=project["path"],
            project_name=project["name"],
            target_language=project.get("target_language") or "eng",
            subtitles=record.get("subtitle_matches") or [],
        )
        return build_subtitle_review(
            project_id=project_id,
            video_id=video_id,
            video_path=Path(record["file_path"]),
            filename=record["filename"],
            project_path=project["path"],
            project_name=project["name"],
            target_language=project.get("target_language") or "eng",
            srt_path=srt_path,
        )
    except SubtitleReviewError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.get(
    "/projects/{project_id}/videos/{video_id}/subtitle-review/original/{chunk_name}"
)
async def stream_original_chunk(
    project_id: int,
    video_id: int,
    chunk_name: str,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> FileResponse:
    """Stream a source audio chunk so the UI can seek to a cue."""
    if not is_safe_chunk_name(chunk_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chunk name",
        )
    project, record = _video_context(project_id, video_id, db)
    dirs = artefact_dirs(Path(record["file_path"]), project["path"], project["name"])
    path = dirs["chunks"] / chunk_name
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio chunk not found: {chunk_name}",
        )
    media = "audio/mp4" if path.suffix.lower() in {".m4a", ".aac"} else "audio/mpeg"
    return FileResponse(path=str(path), media_type=media)


@router.get("/projects/{project_id}/videos/{video_id}/subtitle-review/tts/{index}")
async def stream_tts_segment(
    project_id: int,
    video_id: int,
    index: int,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> FileResponse:
    """Stream the TTS file that corresponds to subtitle cue ``index``."""
    if index < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="index must be >= 0",
        )
    project, record = _video_context(project_id, video_id, db)
    dirs = artefact_dirs(Path(record["file_path"]), project["path"], project["name"])
    path = tts_file_for_index(dirs["tts"], index)
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"TTS segment {index} not found",
        )
    media = "audio/mp4" if path.suffix.lower() in {".m4a", ".aac"} else "audio/mpeg"
    return FileResponse(path=str(path), media_type=media)
