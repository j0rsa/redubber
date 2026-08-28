"""Video file analysis and scanning API endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)

from app.core.dependencies import get_db, get_scanner
from app.infrastructure.task_queue import TaskQueueManager
from app.schemas.models import (
    AudioStream,
    PipelineStatusResponse,
    ScanStatusResponse,
    ScanTriggerResponse,
    SubtitleInfo,
    VideoAnalysis,
)
from app.services.project_scan import scan_project_files
from app.services.dub_reset import RESET_TO_STAGES
from app.services.video_subtitle_quality import (
    enrich_subtitles_with_quality,
    quality_cache_for_listing,
    refresh_subtitle_quality_for_paths,
)
from database import DatabaseManager
from file_scanner import FileScanner
from pipeline_status import get_pipeline_status
from utils import is_video_in_target_state

router = APIRouter()

# Track running scans to prevent concurrent scans on same project
_running_scans: set[int] = set()


async def _scan_project_files(
    project_id: int, project_path: str, db: DatabaseManager, scanner: FileScanner
) -> None:
    """Background task to scan project directory and populate database."""
    try:
        scan_project_files(project_id, project_path, db, scanner)
    finally:
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


@router.get("/projects/{project_id}/scan", response_model=ScanStatusResponse)
async def get_scan_status(
    project_id: int,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> ScanStatusResponse:
    """Return whether a background scan is currently running for the project."""
    project = db.get_project_by_id(project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    return ScanStatusResponse(
        project_id=project_id,
        status="running" if project_id in _running_scans else "idle",
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
    target_lang = project_record.get("target_language") or ""

    # Surface the most recent failed or subtitle-review-held task on each video.
    failed_tasks: dict[str, str] = {}  # video_path → error message
    held_tasks: dict[str, dict] = {}
    try:
        task_manager: TaskQueueManager = request.app.state.task_manager
        all_tasks = await task_manager.list_tasks()
        for t in sorted(all_tasks, key=lambda x: x.created_at, reverse=True):
            if t.status == "failed" and t.video_path and t.error:
                if t.video_path not in failed_tasks:
                    failed_tasks[t.video_path] = t.error
            if (
                t.status == "awaiting_subtitle_review"
                and t.video_path
                and t.video_path not in held_tasks
            ):
                held_tasks[t.video_path] = {
                    "quality_issue_count": t.quality_issue_count,
                    "subtitle_path": t.subtitle_path,
                    "quality_issues": list(t.quality_issues),
                }
    except Exception:
        pass

    # Get video analysis records
    video_records = db.get_video_analysis(project_id)
    quality_cache = quality_cache_for_listing(
        db,
        project_id=project_id,
        project_path=project_path,
        project_name=project_record["name"],
        target_language=target_lang or "eng",
        video_records=video_records,
    )

    results: list[VideoAnalysis] = []
    for record in video_records:
        # Parse audio streams
        audio_streams = [AudioStream(**stream) for stream in record["audio_streams"]]

        # Parse subtitles
        subtitles = [SubtitleInfo(**sub) for sub in record.get("subtitle_matches", [])]
        subtitles = enrich_subtitles_with_quality(
            video_path=Path(record["file_path"]),
            project_path=project_path,
            project_name=project_record["name"],
            target_language=target_lang or "eng",
            subtitles=subtitles,
            cache=quality_cache,
        )

        # Get pipeline status
        pipeline_status_obj = get_pipeline_status(
            video_path=record["file_path"],
            project_path=project_path,
            tmp_root=working_dir,
            target_language=project_record.get("target_language") or "eng",
        )

        task_error = failed_tasks.get(record["file_path"], "")
        held_data = held_tasks.get(record["file_path"])
        if held_data is None:
            from app.services.subtitle_quality_hold import (
                hold_marker_for_video,
                read_subtitle_quality_hold,
            )

            held_data = read_subtitle_quality_hold(
                hold_marker_for_video(
                    record["file_path"],
                    project_path,
                    project_record["name"],
                )
            )
        held_issue_count = int((held_data or {}).get("quality_issue_count") or 0)

        # Detect pre-redubbed files: ≥2 audio tracks where one matches the project target
        # language, AND a subtitle in the target language is present.
        # Covers videos imported from a previously redubbed project with no working dir.
        pre_redubbed = is_video_in_target_state(audio_streams, subtitles, target_lang)

        pipeline_status: PipelineStatusResponse | None = None
        if pre_redubbed:
            # File is already in the final dubbed state (2+ audio tracks +
            # target-language sub). Show as replaced even when a leftover
            # `.dubbed` file or working-dir artifacts would otherwise make
            # pipeline_status look like "awaiting Replace Original".
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
                or pipeline_status_obj.has_external_subs
                or pipeline_status_obj.has_tts
                or pipeline_status_obj.has_target_audio
                or pipeline_status_obj.final_file_exists
            )
            if (
                actual_work_done
                or pipeline_status_obj.is_complete
                or task_error
                or held_issue_count
            ):
                pipeline_status = PipelineStatusResponse(
                    progress=pipeline_status_obj.progress_percent,
                    current_stage=(
                        "Subtitle review required"
                        if held_issue_count
                        else pipeline_status_obj.current_stage
                    ),
                    is_complete=pipeline_status_obj.is_complete,
                    replaced=pipeline_status_obj.replaced,
                    failed=bool(task_error),
                    error=task_error,
                    awaiting_subtitle_review=bool(held_issue_count),
                    quality_issue_count=held_issue_count,
                    subtitle_path=(held_data or {}).get("subtitle_path"),
                    quality_issues=(held_data or {}).get("quality_issues") or [],
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


@router.post(
    "/projects/{project_id}/videos/{video_id}/finalize", status_code=status.HTTP_200_OK
)
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Find the video record
    video_records = db.get_video_analysis(project_id)
    record = next((r for r in video_records if r["id"] == video_id), None)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Video {video_id} not found"
        )

    video_path = record["file_path"]
    working_dir = get_project_working_dir(project["path"], project["name"])
    stem = _Path(video_path).stem
    ext = _Path(video_path).suffix
    video_filename = _Path(
        video_path
    ).name  # e.g. "16. Structure of the Chest Area.mp4"
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


@router.post(
    "/projects/{project_id}/videos/{video_id}/generate-subtitles",
    status_code=status.HTTP_200_OK,
)
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    video_records = db.get_video_analysis(project_id)
    record = next((r for r in video_records if r["id"] == video_id), None)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Video {video_id} not found"
        )

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

    from stt_hallucination import STTHallucinationError, assert_segments_acceptable

    span = max((s.end for s in all_segments), default=0.0) - min(
        (s.start for s in all_segments), default=0.0
    )
    try:
        assert_segments_acceptable(
            all_segments,
            audio_duration=span,
            source_label=video_path,
        )
    except STTHallucinationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.report.summary(),
        ) from exc

    r = Redubber(openai_token="x", interactive=False)  # no API calls needed
    srt_path = r.generate_subtitles(reproj, all_segments)
    refresh_subtitle_quality_for_paths(
        db,
        project_id=project_id,
        video_path=video_path,
        paths=[srt_path],
    )

    return {"status": "generated", "path": srt_path}


@router.post(
    "/projects/{project_id}/videos/{video_id}/reset-dub",
    status_code=status.HTTP_202_ACCEPTED,
)
async def reset_dubbed_video(
    project_id: int,
    video_id: int,
    request: Request,
    db: Annotated[DatabaseManager, Depends(get_db)],
    reset_to: Annotated[
        Literal[
            "start",
            "audio",
            "stt",
            "subtitles",
            "tts",
            "assemble",
            "mix",
        ],
        Query(
            description="Last pipeline stage to keep. Subtitles are deleted only at start."
        ),
    ] = "start",
    keep_subtitles: Annotated[
        bool,
        Query(description="When reset_to=start, keep the generated subtitle file instead of deleting it."),
    ] = False,
) -> dict[str, str]:
    """Queue a job to revert the dubbed video and prune later pipeline stages.

    The dubbed track is always stripped (identified by language tag). Generated
    subtitles are deleted only when ``reset_to=start``. A pre-undub backup is
    written before the file is modified. Only allowed for videos already in the
    final redubbed state. Poll ``GET /api/tasks/{task_id}`` for progress.

    Raises:
        HTTPException: 404 if project or video not found.
        HTTPException: 409 if another job for this video is already queued/running.
        HTTPException: 422 if the video is not in the final state.
    """
    project = db.get_project_by_id(project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    video_records = db.get_video_analysis(project_id)
    record = next((r for r in video_records if r["id"] == video_id), None)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id} not found",
        )

    target_language = project.get("target_language") or "eng"
    audio_streams = record.get("audio_streams") or []
    subtitles = record.get("subtitle_matches") or []
    if not is_video_in_target_state(audio_streams, subtitles, target_language):
        from app.services.dub_reset import (
            _RESET_REJECTED_MSG,
            reconcile_video_with_disk,
        )

        reconcile_video_with_disk(
            db,
            project_id,
            record,
            project["path"],
            project["name"],
            target_language,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_RESET_REJECTED_MSG,
        )

    video_path = record["file_path"]
    task_manager: TaskQueueManager = request.app.state.task_manager
    for t in await task_manager.list_tasks():
        if t.video_path == video_path and t.status in ("queued", "running"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"A job for this video is already {t.status} (task_id={t.task_id})"
                ),
            )

    if reset_to not in RESET_TO_STAGES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown reset stage {reset_to!r}",
        )

    task_id = await task_manager.submit_reset_dub_task(
        video_path=video_path,
        project_id=project_id,
        video_id=video_id,
        reset_to=reset_to,
        keep_subtitles=keep_subtitles,
    )
    return {"task_id": task_id, "status": "queued"}
