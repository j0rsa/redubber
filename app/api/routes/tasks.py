"""Redubbing task management API endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.infrastructure.task_queue import TaskQueueManager
from app.schemas.models import (
    TaskCancelResponse,
    TaskCreate,
    TaskStatusResponse,
)

router = APIRouter()


def _task_to_response(t) -> TaskStatusResponse:
    """Convert a TaskStatus dataclass to TaskStatusResponse, including counters."""
    return TaskStatusResponse(
        task_id=t.task_id,
        video_path=t.video_path,
        status=t.status,
        stage=t.stage,
        progress=t.progress,
        error=t.error,
        created_at=t.created_at,
        audio_chunks=t.audio_chunks,
        transcripts=t.transcripts,
        tts_segments=t.tts_segments,
        tts_total=t.tts_total,
        subtitles=t.subtitles,
        audio_assembled=t.audio_assembled,
        audio_assembled_total=t.audio_assembled_total,
        video_mixed=t.video_mixed,
    )


def get_task_manager(request: Request) -> TaskQueueManager:
    """Provide task queue manager from app state.

    Args:
        request: FastAPI request object with app state.

    Returns:
        TaskQueueManager instance from application state.
    """
    return request.app.state.task_manager


@router.post(
    "/redub", status_code=status.HTTP_202_ACCEPTED, response_model=dict[str, str]
)
async def submit_redub_task(
    task: TaskCreate,
    task_manager: Annotated[TaskQueueManager, Depends(get_task_manager)],
) -> dict[str, str]:
    """Submit a video redubbing job to the task queue.

    Validates the video file exists and is accessible, then enqueues
    a redubbing task for asynchronous processing. Returns immediately
    with a task ID for status polling.

    Args:
        task: Task creation request with video path and project ID.
        task_manager: TaskQueueManager injected from app state.

    Returns:
        Dictionary with task_id and initial status "queued".

    Raises:
        HTTPException: 400 if video path is invalid.
        HTTPException: 503 if task queue is full.
    """
    # Validate video file exists
    video_path = Path(task.video_path)
    if not video_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video file not found: {task.video_path}",
        )

    if not video_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path is not a file: {task.video_path}",
        )

    # Submit task to queue
    try:
        task_id = await task_manager.submit_task(
            video_path=task.video_path,
            project_id=str(task.project_id),
        )
    except Exception as e:
        # Queue full or other error
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to submit task: {e}",
        )

    return {"task_id": task_id, "status": "queued"}


@router.post(
    "/projects/{project_id}/transcribe",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=dict[str, str],
)
async def submit_transcription_task(
    project_id: int,
    task: TaskCreate,
    task_manager: Annotated[TaskQueueManager, Depends(get_task_manager)],
) -> dict[str, str]:
    """Run extract-audio + STT only — no TTS, no cost beyond Whisper.

    Use this before starting voice refinement so that real transcription
    segments are available for preview selection.

    Args:
        project_id: Project the video belongs to.
        task: Request body with video_path.
        task_manager: TaskQueueManager injected from app state.

    Returns:
        Dictionary with task_id and status "queued".

    Raises:
        HTTPException: 400 if video path is invalid.
    """
    video_path = Path(task.video_path)
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video file not found: {task.video_path}",
        )

    task_id = await task_manager.submit_transcription_task(
        video_path=task.video_path,
        project_id=project_id,
    )
    return {"task_id": task_id, "status": "queued"}


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    task_manager: Annotated[TaskQueueManager, Depends(get_task_manager)],
) -> TaskStatusResponse:
    """Get current status of a redubbing task.

    Frontend should poll this endpoint every 2-5 seconds to track
    task progress. Returns detailed stage information and progress
    percentage.

    Args:
        task_id: Unique task identifier returned from submission.
        task_manager: TaskQueueManager injected from app state.

    Returns:
        Complete task status with progress and current stage.

    Raises:
        HTTPException: 404 if task_id not found.
    """
    task_status = await task_manager.get_status(task_id)

    if task_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}",
        )

    return _task_to_response(task_status)


@router.post("/tasks/{task_id}/cancel", response_model=TaskCancelResponse)
async def cancel_task(
    task_id: str,
    task_manager: Annotated[TaskQueueManager, Depends(get_task_manager)],
) -> TaskCancelResponse:
    """Cancel a running or queued redubbing task.

    Attempts graceful cancellation of the task. Queued tasks are
    removed immediately. Running tasks complete their current stage
    before terminating.

    Args:
        task_id: Task identifier to cancel.
        task_manager: TaskQueueManager injected from app state.

    Returns:
        Cancellation confirmation with updated status.

    Raises:
        HTTPException: 404 if task_id not found.
        HTTPException: 409 if task is already completed or failed.
    """
    # Check task exists
    task_status = await task_manager.get_status(task_id)
    if task_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}",
        )

    # Check if cancellable
    if task_status.status in ("completed", "failed"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Task already {task_status.status}, cannot cancel",
        )

    # Attempt cancellation
    success = await task_manager.cancel_task(task_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Task cannot be cancelled in current state",
        )

    return TaskCancelResponse(
        task_id=task_id,
        status="cancelled",
        message=f"Task {task_id} cancelled successfully",
    )


@router.get("/tasks", response_model=list[TaskStatusResponse])
async def list_tasks(
    task_manager: Annotated[TaskQueueManager, Depends(get_task_manager)],
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    project_id: Annotated[int | None, Query()] = None,
) -> list[TaskStatusResponse]:
    """List all tasks with optional filtering.

    Useful for displaying task history or monitoring all active jobs.
    Results are ordered by creation time, most recent first.

    Args:
        status_filter: Optional filter by status (queued/running/completed/failed).
        project_id: Optional filter by project ID.
        task_manager: TaskQueueManager injected from app state.

    Returns:
        List of matching tasks.
    """
    all_tasks = await task_manager.list_tasks()

    if status_filter:
        all_tasks = [t for t in all_tasks if t.status == status_filter]
    if project_id is not None:
        all_tasks = [t for t in all_tasks if t.project_id == project_id]

    return [_task_to_response(t) for t in all_tasks]
