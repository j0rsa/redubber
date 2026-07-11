"""Task queue manager for background redubbing job processing.

Manages long-running video redubbing tasks with hybrid async/sync execution.
Uses asyncio.Queue for task scheduling and ThreadPoolExecutor for blocking operations.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable, Literal, Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)


TaskStatusType = Literal["queued", "running", "completed", "failed"]


@dataclass(frozen=True)
class TaskStatus:
    """Immutable snapshot of a task's current state.

    Represents the progress and status of a single redubbing job.
    All fields are immutable after construction.
    """

    task_id: str
    video_path: str
    stage: str
    progress: int  # 0-100
    status: TaskStatusType
    project_id: int | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    # Pipeline stage counters — populated as each stage completes
    audio_chunks: int | None = None
    transcripts: int | None = None
    tts_segments: int | None = None
    tts_total: int | None = None
    subtitles: int | None = None
    audio_assembled: int | None = None
    audio_assembled_total: int | None = None
    video_mixed: bool | None = None


class AsyncRedubberServiceProtocol(Protocol):
    """Protocol for async redubber service.

    Defines the contract for TTS segment processing.
    Implementations handle OpenAI TTS API calls with concurrency control.
    """

    async def tts_segments_async(
        self,
        segments: list,  # list[TranscriptionSegment] in actual implementation
        output_dir,  # Path in actual implementation
        progress_callback: Callable[[float], None] | None,
        max_concurrent: int,
    ) -> set[str]:
        """Generate TTS audio files for segments asynchronously.

        Args:
            segments: List of transcription segments to convert to speech.
            output_dir: Directory to write output audio files.
            progress_callback: Optional callback for progress updates (0.0-1.0).
            max_concurrent: Maximum concurrent TTS requests.

        Returns:
            Set of paths to generated audio files.

        Raises:
            Exception: If TTS generation fails for any segment.
        """
        ...


class TaskQueueManager:
    """Manages background redubbing task queue with hybrid async/sync execution.

    Orchestrates long-running video redubbing tasks using an asyncio queue
    for task scheduling and ThreadPoolExecutor for blocking operations.
    TTS generation (Stage 4) uses async execution for 5x performance gain.

    Hybrid execution pattern:
    - Stages 1-3 (extract, transcribe): blocking operations via executor
    - Stage 4 (TTS): async operations with high concurrency
    - Stages 5-11 (mix, finalize): blocking operations via executor
    """

    def __init__(
        self,
        max_queue_size: int = 100,
        max_workers: int = 4,
    ) -> None:
        """Initialize task queue manager.

        Args:
            max_queue_size: Maximum number of tasks that can be queued.
            max_workers: Maximum number of worker threads for blocking operations.
        """
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self._tasks: dict[str, TaskStatus] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task[None]] = []
        self._shutdown = False

    async def submit_task(
        self,
        video_path: str,
        project_id: str | int,
    ) -> str:
        """Submit a new redubbing task to the queue.

        Args:
            video_path: Absolute path to video file to redub.
            project_id: ID of the project this video belongs to.

        Returns:
            Unique task ID for tracking status.

        Raises:
            asyncio.QueueFull: If queue is at max capacity.
        """
        task_id = str(uuid4())

        initial_status = TaskStatus(
            task_id=task_id,
            video_path=video_path,
            stage="queued",
            progress=0,
            status="queued",
            project_id=int(project_id),
        )

        async with self._lock:
            self._tasks[task_id] = initial_status

        await self._queue.put(task_id)

        logger.info(
            "Task %s submitted for video %s (project %s)",
            task_id,
            video_path,
            project_id,
        )

        return task_id

    async def get_status(self, task_id: str) -> TaskStatus | None:
        """Get current status of a task.

        Args:
            task_id: Unique task identifier.

        Returns:
            Current TaskStatus snapshot, or None if task not found.
        """
        async with self._lock:
            return self._tasks.get(task_id)

    async def list_tasks(self) -> list[TaskStatus]:
        """Return tasks sorted: running first, queued next, then others by creation time."""
        _STATUS_ORDER = {"running": 0, "queued": 1, "failed": 2, "completed": 3}
        async with self._lock:
            return sorted(
                self._tasks.values(),
                key=lambda t: (_STATUS_ORDER.get(t.status, 9), t.created_at),
            )

    async def cancel_task(self, task_id: str) -> bool:
        """Mark a task for cooperative cancellation.

        Note: This is cooperative cancellation. The task will check for
        cancellation at safe points and stop gracefully.

        Args:
            task_id: Unique task identifier.

        Returns:
            True if task was marked for cancellation, False if not found.
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            if task.status in ("queued", "running"):
                # Mark as failed with cancellation message
                updated = replace(
                    task,
                    status="failed",
                    error="Task cancelled by user",
                    completed_at=datetime.now(),
                )
                self._tasks[task_id] = updated
                logger.info("Task %s marked for cancellation", task_id)
                return True

            return False

    async def start_workers(self, num_workers: int) -> None:
        """Start background worker tasks.

        Args:
            num_workers: Number of concurrent workers to start.
        """
        self._shutdown = False

        for worker_id in range(num_workers):
            worker = asyncio.create_task(
                self._worker(worker_id),
                name=f"task-worker-{worker_id}",
            )
            self._workers.append(worker)

        logger.info("Started %d task queue workers", num_workers)

    async def stop_workers(self) -> None:
        """Stop all background workers gracefully.

        Cancels all worker tasks and waits for them to complete.
        Running tasks will be allowed to finish their current stage.
        """
        self._shutdown = True

        logger.info("Stopping %d task queue workers", len(self._workers))

        for worker in self._workers:
            worker.cancel()

        # Wait for all workers to complete
        results = await asyncio.gather(*self._workers, return_exceptions=True)

        # Log any unexpected exceptions (not cancellation)
        for idx, result in enumerate(results):
            if isinstance(result, Exception) and not isinstance(
                result, asyncio.CancelledError
            ):
                logger.error(
                    "Worker %d failed with exception: %s",
                    idx,
                    result,
                )

        self._workers.clear()
        logger.info("All task queue workers stopped")

    async def _worker(self, worker_id: int) -> None:
        """Background worker that processes tasks from the queue.

        Args:
            worker_id: Unique worker identifier for logging.
        """
        logger.info("Worker %d started", worker_id)

        while not self._shutdown:
            try:
                # Wait for next task with timeout to check shutdown flag
                try:
                    task_id = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except TimeoutError:
                    continue

                logger.info("Worker %d processing task %s", worker_id, task_id)

                try:
                    await self._process_task(task_id)
                except Exception as e:
                    logger.exception(
                        "Worker %d failed to process task %s",
                        worker_id,
                        task_id,
                    )
                    await self._mark_task_failed(task_id, str(e))
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                logger.info("Worker %d cancelled", worker_id)
                break
            except Exception:
                logger.exception("Worker %d encountered unexpected error", worker_id)
                await asyncio.sleep(1.0)  # Backoff on unexpected errors

        logger.info("Worker %d stopped", worker_id)

    @staticmethod
    def _resolve_reproj_root(video_path: str, project_id: int | None) -> str:
        """Return the working-directory root for a Reproj, honouring project settings.

        Uses get_project_working_dir so artefacts land in the same place that the
        UI displays (e.g. /Users/.../course/.redubber/ or a global working_directory).
        Falls back to the legacy ./redubber_tmp when the project cannot be resolved.
        """
        try:
            from app.core.config import settings as _config_settings
            from app.core.project_paths import get_project_working_dir
            from database import DatabaseManager

            if project_id is not None:
                db = DatabaseManager(_config_settings.database_url)
                project = db.get_project_by_id(project_id)
                if project:
                    wd = get_project_working_dir(project["path"], project["name"])
                    wd.mkdir(parents=True, exist_ok=True)
                    return str(wd)
        except Exception:
            pass
        from app.core.config import settings as _config_settings
        return str(_config_settings.tmp_path)

    async def _process_task(self, task_id: str) -> None:
        """Process a single redubbing task with hybrid async/sync execution.

        Stages 1-3: Blocking operations (extract audio, transcribe)
        Stage 4: Async TTS generation (5x performance boost)
        Stages 5-11: Blocking operations (mix audio, finalize)

        Args:
            task_id: Unique task identifier.

        Raises:
            Exception: If any stage of processing fails.
        """
        from pathlib import Path

        from app.core.config import settings
        from app.infrastructure.async_redubber_service import AsyncRedubberService
        from redubber import Redubber
        from reproj import Reproj

        # Get task details
        task = self._tasks.get(task_id)
        if task is None:
            logger.error("Task %s not found in queue", task_id)
            return

        video_path = task.video_path
        project_id = task.project_id

        # Update status to running
        await self._update_task_status(
            task_id,
            stage="Initializing",
            progress=5,
            status="running",
            started_at=datetime.now(),
        )

        # Get event loop for running blocking operations
        loop = asyncio.get_event_loop()

        try:
            # Stage 1-3: Extract audio, transcribe, translate (blocking operations)
            # These use ThreadPoolExecutor to avoid blocking the event loop

            await self._update_task_status(
                task_id,
                stage="Extracting and transcribing audio",
                progress=10,
                status="running",
            )

            # Create Reproj for working directories
            reproj_root = self._resolve_reproj_root(video_path, task.project_id if task else None)

            def create_reproj_and_extract() -> tuple[Reproj, list]:
                """Blocking operation: create reproj and extract segments."""
                reproj = Reproj(
                    source=str(Path(video_path).parent),
                    file_path=video_path,
                    root=reproj_root,
                )

                # Pull operational settings from tool settings
                try:
                    from app.services.settings_service import get_settings as _get_settings
                    _tool_settings = _get_settings()
                    _stt_model = _tool_settings.stt_model
                    _base_url = _tool_settings.openai_base_url
                    _tts_speed = _tool_settings.tts_speed
                    _audio_chunk_duration = _tool_settings.audio_chunk_duration
                    _tts_concurrency = _tool_settings.tts_concurrency
                except Exception:
                    _stt_model = "whisper-1"
                    _base_url = ""
                    _tts_speed = 1.25
                    _audio_chunk_duration = 900
                    _tts_concurrency = 20

                # Look up target language from project settings
                _target_language = "eng"
                if project_id is not None:
                    try:
                        from database import DatabaseManager
                        _db = DatabaseManager(settings.database_url)
                        _project = _db.get_project_by_id(project_id)
                        _target_language = (
                            _project.get("target_language") or "eng"
                        ) if _project else "eng"
                    except Exception:
                        pass  # keep default

                redubber = Redubber(
                    openai_token=settings.openai_api_key,
                    interactive=False,
                    stt_model=_stt_model,
                    openai_base_url=_base_url,
                    tts_speed=_tts_speed,
                    audio_chunk_duration=_audio_chunk_duration,
                    tts_concurrency=_tts_concurrency,
                    target_language=_target_language,
                )

                # Extract and transcribe segments
                segments = redubber.get_text_and_segments(reproj, compact=True)

                return reproj, segments

            # Run blocking operation in executor
            reproj, segments = await loop.run_in_executor(
                self._executor,
                create_reproj_and_extract,
            )

            logger.info("Task %s: Extracted %d segments", task_id, len(segments))

            # Count audio chunks and transcripts (seg files) from disk after extraction
            _audio_chunk_count = 0
            _transcript_count = 0
            try:
                _ac_dir = Path(reproj.get_file_working_dir(Reproj.Section.SOURCE_AUDIO_CHUNKS))
                _audio_chunk_count = len([f for f in _ac_dir.iterdir() if f.suffix in (".m4a", ".mp3")])
                _stt_dir = Path(reproj.get_file_working_dir(Reproj.Section.STT))
                _transcript_count = len([f for f in _stt_dir.glob("*.seg")])
            except Exception:
                pass

            await self._update_task_status(
                task_id, stage="Transcription complete", progress=35, status="running",
                audio_chunks=_audio_chunk_count, transcripts=_transcript_count,
            )

            # Stage 3b: Generate subtitles
            await self._update_task_status(
                task_id, stage="Generating subtitles", progress=35, status="running",
                audio_chunks=_audio_chunk_count, transcripts=_transcript_count,
            )

            def generate_subtitles_step() -> None:
                from redubber import Redubber
                _r = Redubber(openai_token=settings.openai_api_key, interactive=False)
                _r.generate_subtitles(reproj, segments)

            await loop.run_in_executor(self._executor, generate_subtitles_step)

            await self._update_task_status(
                task_id, stage="Subtitles generated", progress=38, status="running",
                audio_chunks=_audio_chunk_count, transcripts=_transcript_count, subtitles=1,
            )

            # Stage 4: ASYNC TTS generation (KEY OPTIMIZATION - 5x faster!)
            await self._update_task_status(
                task_id, stage="Generating TTS (async)", progress=40, status="running"
            )

            # Create output directory for TTS
            tts_output_dir = Path(reproj.get_file_working_dir(Reproj.Section.TTS))

            # Initialize async TTS service
            _openai_timeout = 60.0
            _openai_retries = 3
            _tts_model = "gpt-4o-mini-tts"
            _project_voice = settings.openai_voice
            _project_voice_instructions = ""
            try:
                from app.services.settings_service import get_settings as _get_settings_tts
                _tts_settings = _get_settings_tts()
                _openai_timeout = _tts_settings.openai_timeout
                _openai_retries = _tts_settings.openai_retries
                _tts_model = _tts_settings.tts_model or _tts_model
            except Exception:
                pass

            # Load voice and instructions from project settings
            try:
                from database import DatabaseManager as _DM
                _db2 = _DM(settings.database_url)
                _proj2 = _db2.get_project_by_id(task.project_id) if task and task.project_id else None
                if _proj2:
                    _project_voice = _proj2.get("voice") or _project_voice
                    _project_voice_instructions = _proj2.get("voice_instructions") or ""
            except Exception:
                pass

            async_service = AsyncRedubberService(
                openai_token=settings.openai_api_key,
                voice=_project_voice,
                voice_instructions=_project_voice_instructions,
                openai_timeout=_openai_timeout,
                openai_retries=_openai_retries,
                tts_model=_tts_model,
            )

            _tts_total = len(segments)

            # Progress callback for TTS
            def tts_progress_callback(progress: float) -> None:
                """Update task progress during TTS generation."""
                task_progress = int(40 + (progress * 32))
                done = int(progress * _tts_total)
                asyncio.create_task(
                    self._update_task_status(
                        task_id,
                        stage=f"Generating TTS ({int(progress * 100)}%)",
                        progress=task_progress,
                        status="running",
                        audio_chunks=_audio_chunk_count,
                        transcripts=_transcript_count,
                        subtitles=1,
                        tts_segments=done,
                        tts_total=_tts_total,
                    )
                )

            # Generate TTS files asynchronously (100+ concurrent requests)
            tts_files = await async_service.tts_segments_async(
                segments=segments,
                output_dir=tts_output_dir,
                progress_callback=tts_progress_callback,
                max_concurrent=settings.tts_max_concurrent,
            )

            await async_service.close()

            _tts_done = len(tts_files)
            logger.info("Task %s: Generated %d TTS files", task_id, _tts_done)

            # Stage 5-11: Assemble audio and mix with video (blocking operations)
            await self._update_task_status(
                task_id, stage="Assembling audio", progress=75, status="running",
                audio_chunks=_audio_chunk_count, transcripts=_transcript_count,
                subtitles=1, tts_segments=_tts_done, tts_total=_tts_total,
            )

            _assembly_total_chunks = 1  # will be set inside assemble_and_mix

            def assemble_and_mix() -> str:
                """Blocking operation: assemble audio and mix with video."""
                try:
                    from app.services.settings_service import get_settings as _get_settings_mix
                    _mix_settings = _get_settings_mix()
                    _mix_tts_speed = _mix_settings.tts_speed
                    _mix_audio_chunk_duration = _mix_settings.audio_chunk_duration
                    _mix_tts_concurrency = _mix_settings.tts_concurrency
                except Exception:
                    _mix_tts_speed = 1.25
                    _mix_audio_chunk_duration = 900
                    _mix_tts_concurrency = 20

                redubber = Redubber(
                    openai_token=settings.openai_api_key,
                    interactive=False,
                    tts_speed=_mix_tts_speed,
                    audio_chunk_duration=_mix_audio_chunk_duration,
                    tts_concurrency=_mix_tts_concurrency,
                )

                # Get video duration
                duration = redubber.get_media_duration(video_path)

                # Compute total assembly chunks so we can report progress
                import math as _math
                _max_seg = 50  # matches assemble_long_audio default
                nonlocal _assembly_total_chunks
                _assembly_total_chunks = max(1, _math.ceil(len(segments) / _max_seg))
                _total_chunks = _assembly_total_chunks

                def _assembly_progress(progress: float) -> None:
                    done = max(1, int(progress / 0.8 * _total_chunks))
                    asyncio.run_coroutine_threadsafe(
                        self._update_task_status(
                            task_id,
                            stage=f"Assembling audio ({done}/{_total_chunks})",
                            progress=int(75 + progress * 10),
                            status="running",
                            audio_chunks=_audio_chunk_count,
                            transcripts=_transcript_count,
                            subtitles=1,
                            tts_segments=_tts_done,
                            tts_total=_tts_total,
                            audio_assembled=done,
                            audio_assembled_total=_total_chunks,
                        ),
                        loop,
                    )

                # Assemble long audio from TTS segments
                redubbed_audio_path = redubber.assemble_long_audio(
                    segments, reproj, duration, progress_callback=_assembly_progress
                )

                # Resolve language metadata from project settings
                _dubbed_lang = "eng"
                _source_lang = "und"
                try:
                    from database import DatabaseManager as _DM2
                    _db3 = _DM2(settings.database_url)
                    _proj3 = _db3.get_project_by_id(task.project_id) if task and task.project_id else None
                    if _proj3:
                        _dubbed_lang = _proj3.get("target_language") or "eng"
                        _source_lang = _proj3.get("source_language_override") or "und"
                except Exception:
                    pass

                # Mix audio with video — output goes into the working directory
                # finalize_redubbing will validate, then replace the original
                _root_dir = Path(reproj.get_file_working_dir(Reproj.Section.ROOT))
                _stem = Path(video_path).stem
                _ext = Path(video_path).suffix
                output_video = str(_root_dir / f"{_stem}.dubbed{_ext}")
                redubber.mix_audio_with_video(
                    reproj=reproj,
                    audio_file=redubbed_audio_path,
                    output_file=output_video,
                    languages=[_source_lang, _dubbed_lang],
                )

                # Validate → replace original with backup → clean up temp files
                from redubber import finalize_redubbing
                from database import DatabaseManager as _DM3
                _db4 = _DM3(settings.database_url)
                _replace = False
                try:
                    from app.services.settings_service import get_settings as _gs_final
                    _replace = _gs_final().auto_process
                except Exception:
                    pass
                final_path = finalize_redubbing(
                    db=_db4,
                    reproj=reproj,
                    final_video_path=output_video,
                    project_id=int(task.project_id) if task and task.project_id else 0,
                    replace_original=_replace,
                    target_language=_dubbed_lang,
                )
                return final_path

            # Run blocking operation in executor
            await self._update_task_status(
                task_id, stage="Mixing audio with video", progress=85, status="running",
                audio_chunks=_audio_chunk_count, transcripts=_transcript_count,
                subtitles=1, tts_segments=_tts_done, tts_total=_tts_total,
                audio_assembled=_assembly_total_chunks, audio_assembled_total=_assembly_total_chunks,
            )

            final_video_path = await loop.run_in_executor(
                self._executor,
                assemble_and_mix,
            )

            logger.info("Task %s: Created final video at %s", task_id, final_video_path)

            # Mark as completed with all counters
            await self._update_task_status(
                task_id,
                stage="Completed",
                progress=100,
                status="completed",
                completed_at=datetime.now(),
                audio_chunks=_audio_chunk_count, transcripts=_transcript_count,
                subtitles=1, tts_segments=_tts_done, tts_total=_tts_total,
                audio_assembled=_assembly_total_chunks, audio_assembled_total=_assembly_total_chunks,
                video_mixed=True,
            )

        except Exception as e:
            logger.exception("Task %s failed during processing", task_id)
            await self._mark_task_failed(task_id, str(e))
            raise

    async def _update_task_status(
        self,
        task_id: str,
        stage: str,
        progress: int,
        status: TaskStatusType,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        **counters: object,
    ) -> None:
        """Update task status with new values.

        Args:
            task_id: Unique task identifier.
            stage: Current processing stage.
            progress: Progress percentage (0-100).
            status: Current task status.
            started_at: Optional task start timestamp.
            completed_at: Optional task completion timestamp.
            **counters: Optional pipeline stage counters (audio_chunks, transcripts, etc.)
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.warning("Attempted to update non-existent task %s", task_id)
                return

            updates: dict = {
                "stage": stage,
                "progress": progress,
                "status": status,
            }

            if started_at is not None:
                updates["started_at"] = started_at
            if completed_at is not None:
                updates["completed_at"] = completed_at

            # Apply counter fields — only update if explicitly passed
            for key in ("audio_chunks", "transcripts", "tts_segments", "tts_total",
                        "subtitles", "audio_assembled", "audio_assembled_total", "video_mixed"):
                if key in counters:
                    updates[key] = counters[key]

            self._tasks[task_id] = replace(task, **updates)

    async def submit_transcription_task(
        self,
        video_path: str,
        project_id: str | int,
    ) -> str:
        """Submit a transcription-only task (extract audio + STT, no TTS).

        Runs the first two pipeline stages so that .seg files are written to disk
        and voice refinement becomes available before any TTS costs are incurred.

        Args:
            video_path: Absolute path to the video file.
            project_id: ID of the project.

        Returns:
            Unique task ID for status polling.
        """
        task_id = str(uuid4())

        initial_status = TaskStatus(
            task_id=task_id,
            video_path=video_path,
            stage="queued",
            progress=0,
            status="queued",
            project_id=int(project_id),
        )

        async with self._lock:
            self._tasks[task_id] = initial_status

        # Run directly in the executor — don't go through the redub queue
        asyncio.get_event_loop()
        asyncio.ensure_future(self._process_transcription_task(task_id))

        logger.info(
            "Transcription task %s submitted for video %s (project %s)",
            task_id,
            video_path,
            project_id,
        )
        return task_id

    async def _process_transcription_task(self, task_id: str) -> None:
        """Run extract-audio + STT only for a transcription task."""
        from pathlib import Path

        from app.core.config import settings
        from redubber import Redubber
        from reproj import Reproj

        task = self._tasks.get(task_id)
        if task is None:
            return

        video_path = task.video_path
        project_id = task.project_id

        await self._update_task_status(
            task_id, stage="Extracting audio", progress=5, status="running",
            started_at=datetime.now(),
        )

        loop = asyncio.get_event_loop()

        try:
            def run_stt() -> int:
                try:
                    from app.services.settings_service import get_settings as _gs
                    s = _gs()
                    stt_model = s.stt_model
                    base_url = s.openai_base_url
                    audio_chunk_duration = s.audio_chunk_duration
                except Exception:
                    stt_model = "whisper-1"
                    base_url = ""
                    audio_chunk_duration = 900

                target_language = "eng"
                if project_id is not None:
                    try:
                        from database import DatabaseManager
                        _db = DatabaseManager(settings.database_url)
                        _project = _db.get_project_by_id(project_id)
                        target_language = (_project.get("target_language") or "eng") if _project else "eng"
                    except Exception:
                        pass

                redubber = Redubber(
                    openai_token=settings.openai_api_key,
                    interactive=False,
                    stt_model=stt_model,
                    openai_base_url=base_url,
                    audio_chunk_duration=audio_chunk_duration,
                    target_language=target_language,
                )

                reproj = Reproj(
                    source=str(Path(video_path).parent),
                    file_path=video_path,
                    root=self._resolve_reproj_root(video_path, project_id),
                )

                segments = redubber.get_text_and_segments(reproj, compact=True)
                return len(segments)

            await self._update_task_status(
                task_id, stage="Transcribing", progress=20, status="running",
            )

            segment_count = await loop.run_in_executor(self._executor, run_stt)

            await self._update_task_status(
                task_id,
                stage=f"Done — {segment_count} segments",
                progress=100,
                status="completed",
                completed_at=datetime.now(),
            )
            logger.info("Transcription task %s completed: %d segments", task_id, segment_count)

        except Exception as e:
            logger.exception("Transcription task %s failed", task_id)
            await self._mark_task_failed(task_id, str(e))

    async def _mark_task_failed(
        self,
        task_id: str,
        error: str,
    ) -> None:
        """Mark a task as failed with error message.

        Args:
            task_id: Unique task identifier.
            error: Error message describing the failure.
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.warning("Attempted to fail non-existent task %s", task_id)
                return

            self._tasks[task_id] = replace(
                task,
                status="failed",
                error=error,
                completed_at=datetime.now(),
            )

            logger.error("Task %s failed: %s", task_id, error)
