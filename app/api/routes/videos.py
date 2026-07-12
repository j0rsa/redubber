"""Video file analysis and scanning API endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from app.core.dependencies import get_db, get_scanner
from app.schemas.models import (
    AudioStream,
    PipelineStatusResponse,
    ScanTriggerResponse,
    SubtitleInfo,
    VideoAnalysis,
)
from database import DatabaseManager
from file_scanner import FileScanner
from pipeline_status import get_pipeline_status
from utils import detect_subtitle_language, detect_video_language
from video_analyzer import get_video_info_with_duration

router = APIRouter()

# Track running scans to prevent concurrent scans on same project
_running_scans: set[int] = set()


async def _scan_project_files(
    project_id: int, project_path: str, db: DatabaseManager, scanner: FileScanner
) -> None:
    """Background task to scan project directory and populate database.

    Args:
        project_id: ID of project to scan.
        project_path: Absolute path to project directory.
        db: DatabaseManager instance.
        scanner: FileScanner instance.
    """
    try:
        db.clear_project_files(project_id)
        video_files, subtitle_files = scanner.scan_folder(project_path)

        # Add video files
        for video_file in video_files:
            language = detect_video_language(video_file)
            db.add_video_file(
                project_id=project_id,
                file_path=str(video_file),
                filename=video_file.name,
                language=language,
            )

            # Find subtitle files that belong to this video (same stem, any sub extension)
            video_stem = video_file.stem
            matched_subs = [
                s for s in subtitle_files
                if s.stem == video_stem or s.stem.startswith(video_stem + ".")
            ]
            subtitle_info = []
            for sub in matched_subs:
                sub_lang = detect_subtitle_language(sub)
                subtitle_info.append({
                    "language": sub_lang or "",
                    "embedded": False,
                    "path": str(sub),
                    "filename": sub.name,
                })

            # Analyze video and store results
            video_info = get_video_info_with_duration(video_file)
            db.save_video_analysis(
                project_id=project_id,
                video_data={
                    "filename": video_file.name,
                    "path": str(video_file),
                    "size_mb": round(video_file.stat().st_size / (1024 * 1024), 2),
                    "duration_seconds": video_info["duration_seconds"],
                    "audio_streams": video_info["audio_streams"],
                    "subtitles": subtitle_info,
                },
            )

        # Register all subtitle files in the subtitle_files table
        for subtitle_file in subtitle_files:
            language = detect_subtitle_language(subtitle_file)
            db.add_subtitle_file(
                project_id=project_id,
                file_path=str(subtitle_file),
                filename=subtitle_file.name,
                language=language,
            )
    finally:
        # Remove from running scans tracking
        _running_scans.discard(project_id)


@router.post("/projects/{project_id}/scan", response_model=ScanTriggerResponse)
async def trigger_scan(
    project_id: int,
    background_tasks: BackgroundTasks,
    db: Annotated[DatabaseManager, Depends(get_db)],
    scanner: Annotated[FileScanner, Depends(get_scanner)],
) -> ScanTriggerResponse:
    """Trigger asynchronous file scan for a project.

    Initiates background scanning of the project directory to detect
    and analyze video and subtitle files. Returns immediately while
    scan runs in background.

    Args:
        project_id: Project to scan.
        background_tasks: FastAPI background tasks manager.
        db: DatabaseManager dependency.
        scanner: FileScanner dependency.

    Returns:
        Scan trigger confirmation with status.

    Raises:
        HTTPException: 404 if project not found.
        HTTPException: 409 if scan is already running for this project.
    """
    # Verify project exists
    projects = db.get_all_projects()
    project_path: str | None = None

    for project in projects:
        if project["id"] == project_id:
            project_path = project["path"]
            break

    if not project_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Check if scan is already running
    if project_id in _running_scans:
        return ScanTriggerResponse(
            project_id=project_id,
            status="already_running",
            message=f"Scan is already running for project {project_id}",
        )

    # Mark as running and trigger background scan
    _running_scans.add(project_id)
    background_tasks.add_task(
        _scan_project_files, project_id, project_path, db, scanner
    )

    return ScanTriggerResponse(
        project_id=project_id,
        status="started",
        message=f"Background scan started for project {project_id}",
    )


@router.get("/projects/{project_id}/videos", response_model=list[VideoAnalysis])
async def list_videos(
    project_id: int,
    request: Request,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> list[VideoAnalysis]:
    """List all analyzed videos for a project with pipeline status.

    Returns comprehensive video metadata including audio streams,
    subtitles, and current pipeline processing status from the
    redubber_tmp directory.

    Args:
        project_id: Project identifier.
        db: DatabaseManager dependency.

    Returns:
        List of video analysis records with pipeline status.

    Raises:
        HTTPException: 404 if project not found.
    """
    # Verify project exists and get project path
    project_record = db.get_project_by_id(project_id)

    if not project_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    project_path = project_record["path"]

    from app.core.project_paths import get_project_working_dir
    working_dir = str(get_project_working_dir(project_path, project_record["name"]))

    # Build a map of video_path → most-recent failed task so we can surface errors
    failed_tasks: dict[str, str] = {}  # video_path → error message
    try:
        from app.infrastructure.task_queue import TaskQueueManager
        task_manager: TaskQueueManager = request.app.state.task_manager
        all_tasks = await task_manager.list_tasks()
        for t in all_tasks:
            if t.status == "failed" and t.video_path and t.error:
                # Keep only the most-recent failure per video path
                if t.video_path not in failed_tasks:
                    failed_tasks[t.video_path] = t.error
    except Exception:
        pass

    # Get video analysis records
    video_records = db.get_video_analysis(project_id)

    results: list[VideoAnalysis] = []
    for record in video_records:
        # Parse audio streams
        audio_streams = [AudioStream(**stream) for stream in record["audio_streams"]]

        # Parse subtitles
        subtitles = [SubtitleInfo(**sub) for sub in record.get("subtitle_matches", [])]

        # Get pipeline status
        pipeline_status_obj = get_pipeline_status(
            video_path=record["file_path"],
            project_path=project_path,
            tmp_root=working_dir,
        )

        task_error = failed_tasks.get(record["file_path"], "")

        # Detect pre-redubbed files: ≥2 audio tracks where one matches the project target
        # language, AND a subtitle in the target language is present.
        # Covers videos imported from a previously redubbed project with no working dir.
        target_lang = project_record.get("target_language") or ""
        audio_langs = {s.language for s in audio_streams if s.language}
        sub_langs = {s.language for s in subtitles if s.language}
        pre_redubbed = (
            bool(target_lang)
            and len(audio_streams) >= 2
            and target_lang in audio_langs
            and target_lang in sub_langs
        )

        pipeline_status: PipelineStatusResponse | None = None
        if pre_redubbed and not pipeline_status_obj.final_file_exists:
            pipeline_status = PipelineStatusResponse(
                progress=100,
                current_stage="Complete",
                is_complete=True,
                replaced=True,
                failed=False,
                error="",
            )
        else:
            # Only show pipeline status when actual pipeline work has started or completed.
            actual_work_done = (
                pipeline_status_obj.has_audio_chunks
                or pipeline_status_obj.has_transcripts
                or pipeline_status_obj.subtitles_generated
                or pipeline_status_obj.has_tts
                or pipeline_status_obj.has_target_audio
                or pipeline_status_obj.final_file_exists
            )
            if actual_work_done or pipeline_status_obj.is_complete or task_error:
                pipeline_status = PipelineStatusResponse(
                    progress=pipeline_status_obj.progress_percent,
                    current_stage=pipeline_status_obj.current_stage,
                    is_complete=pipeline_status_obj.is_complete,
                    replaced=pipeline_status_obj.replaced,
                    failed=bool(task_error),
                    error=task_error,
                )

        results.append(
            VideoAnalysis(
                id=record["id"],
                filename=record["filename"],
                path=record["file_path"],
                size_mb=record["size_mb"],
                duration_seconds=record["duration_seconds"],
                audio_streams=audio_streams,
                subtitles=subtitles,
                pipeline_status=pipeline_status,
            )
        )

    return results


@router.post("/projects/{project_id}/videos/{video_id}/finalize", status_code=status.HTTP_200_OK)
async def finalize_video(
    project_id: int,
    video_id: int,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> dict:
    """Validate the dubbed file and replace the original with a backup.

    Runs in the calling request (blocking, but typically fast — mostly ffprobe + file rename).
    Steps: validate dubbed file (streams, duration, 2 audio tracks) → backup original → replace.

    Raises:
        HTTPException: 404 if project or video not found.
        HTTPException: 422 if dubbed file not found or validation fails.
        HTTPException: 500 if replacement fails.
    """
    from pathlib import Path as _Path

    from app.core.project_paths import get_project_working_dir
    from redubber import finalize_redubbing, validate_video_file

    project = db.get_project_by_id(project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")

    # Find the video record
    video_records = db.get_video_analysis(project_id)
    record = next((r for r in video_records if r["id"] == video_id), None)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Video {video_id} not found")

    video_path = record["file_path"]
    working_dir = get_project_working_dir(project["path"], project["name"])
    stem = _Path(video_path).stem
    ext = _Path(video_path).suffix
    video_filename = _Path(video_path).name  # e.g. "16. Structure of the Chest Area.mp4"
    # Reproj puts per-video artefacts under <working_dir>/<video_filename>/
    dubbed_path = str(working_dir / video_filename / f"{stem}.dubbed{ext}")

    if not _Path(dubbed_path).exists():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dubbed file not found: {dubbed_path}. Complete the redub pipeline first.",
        )

    # Validate
    if not validate_video_file(dubbed_path, reference=video_path):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Dubbed file failed validation. It may be corrupted, have the wrong duration, or be missing audio tracks.",
        )

    # Replace original
    from reproj import Reproj
    reproj = Reproj(
        source=str(_Path(video_path).parent),
        file_path=video_path,
        root=str(working_dir),
    )
    _target_lang = "eng"
    try:
        _target_lang = project.get("target_language") or "eng"
    except Exception:
        pass

    try:
        result_path = finalize_redubbing(
            db=db,
            reproj=reproj,
            final_video_path=dubbed_path,
            project_id=project_id,
            replace_original=True,
            target_language=_target_lang,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Replacement failed: {e}",
        )

    return {"status": "replaced", "path": result_path}


@router.post("/projects/{project_id}/videos/{video_id}/generate-subtitles", status_code=status.HTTP_200_OK)
async def generate_subtitles_for_video(
    project_id: int,
    video_id: int,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> dict:
    """Regenerate subtitles from existing transcription segments (.seg files).

    Reads the already-transcribed .seg files and writes the .srt to 03_subtitles/.
    Safe to call on a pipeline that completed before subtitle generation was added.

    Raises:
        HTTPException: 404 if project or video not found.
        HTTPException: 422 if no .seg files exist yet.
    """
    from pathlib import Path as _Path

    from app.core.project_paths import get_project_working_dir
    from reproj import Reproj
    from redubber import Redubber
    from openai.types.audio.transcription_segment import TranscriptionSegment

    project = db.get_project_by_id(project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")

    video_records = db.get_video_analysis(project_id)
    record = next((r for r in video_records if r["id"] == video_id), None)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Video {video_id} not found")

    video_path = record["file_path"]
    working_dir = get_project_working_dir(project["path"], project["name"])
    _Path(video_path).name

    reproj = Reproj(
        source=str(_Path(video_path).parent),
        file_path=video_path,
        root=str(working_dir),
    )

    # Load segments from .seg files in 02_stt/
    stt_dir = _Path(reproj.get_file_working_dir(Reproj.Section.STT))
    seg_files = sorted(stt_dir.glob("*.seg")) if stt_dir.exists() else []
    if not seg_files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No transcription segments found. Run the transcription step first.",
        )

    from pydantic import TypeAdapter
    from typing import List
    ta = TypeAdapter(List[TranscriptionSegment])

    all_segments: list = []
    for seg_file in seg_files:
        try:
            segments = ta.validate_json(seg_file.read_text())
            all_segments.extend(segments)
        except Exception:
            continue

    if not all_segments:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Segment files found but could not be parsed.",
        )

    all_segments.sort(key=lambda s: s.start)

    r = Redubber(openai_token="x", interactive=False)  # no API calls needed
    srt_path = r.generate_subtitles(reproj, all_segments)

    return {"status": "generated", "path": srt_path}
