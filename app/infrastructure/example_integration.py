"""Example integration of TaskQueueManager with FastAPI.

This file demonstrates how to integrate TaskQueueManager into the FastAPI
application lifecycle and access it from API routes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.core.config import settings
from app.infrastructure.task_queue import TaskQueueManager


@asynccontextmanager
async def lifespan_with_task_queue(app: FastAPI) -> AsyncGenerator[None, None]:
    """Example lifespan manager that initializes TaskQueueManager.

    Args:
        app: FastAPI application instance.

    Yields:
        Control back to FastAPI during application runtime.
    """
    # Startup: Initialize and start task queue workers
    task_manager = TaskQueueManager(
        max_queue_size=settings.task_queue_max_size,
        max_workers=4,  # Thread pool for blocking operations
    )

    await task_manager.start_workers(num_workers=settings.max_concurrent_redubs)

    # Store in app state for access from routes
    app.state.task_manager = task_manager

    # Ensure directories exist
    settings.tmp_path.mkdir(parents=True, exist_ok=True)
    settings.storage_path.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown: Stop workers gracefully
    await task_manager.stop_workers()


# Example API route handler
async def submit_redub_task(
    app: FastAPI,
    video_path: str,
    project_id: str,
) -> str:
    """Example route handler that submits a redubbing task.

    Args:
        app: FastAPI application instance (injected by FastAPI).
        video_path: Absolute path to video file.
        project_id: Project identifier.

    Returns:
        Task ID for tracking progress.

    Example:
        ```python
        from fastapi import FastAPI, Depends

        app = FastAPI(lifespan=lifespan_with_task_queue)

        @app.post("/api/tasks/submit")
        async def submit_task(
            video_path: str,
            project_id: str,
            app: FastAPI = Depends(lambda: app),
        ) -> dict[str, str]:
            task_id = await submit_redub_task(app, video_path, project_id)
            return {"task_id": task_id}
        ```
    """
    task_manager: TaskQueueManager = app.state.task_manager
    task_id = await task_manager.submit_task(video_path, project_id)
    return task_id


async def get_task_status(
    app: FastAPI,
    task_id: str,
) -> dict[str, str | int | None]:
    """Example route handler that retrieves task status.

    Args:
        app: FastAPI application instance.
        task_id: Task identifier.

    Returns:
        Task status dictionary or error if not found.

    Example:
        ```python
        @app.get("/api/tasks/{task_id}")
        async def get_status(
            task_id: str,
            app: FastAPI = Depends(lambda: app),
        ) -> dict[str, str | int | None]:
            return await get_task_status(app, task_id)
        ```
    """
    task_manager: TaskQueueManager = app.state.task_manager
    status = await task_manager.get_status(task_id)

    if status is None:
        return {"error": "Task not found"}

    return {
        "task_id": status.task_id,
        "video_path": status.video_path,
        "stage": status.stage,
        "progress": status.progress,
        "status": status.status,
        "error": status.error,
        "created_at": status.created_at.isoformat(),
        "started_at": status.started_at.isoformat() if status.started_at else None,
        "completed_at": (
            status.completed_at.isoformat() if status.completed_at else None
        ),
    }
