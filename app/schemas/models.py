"""Pydantic schemas for Redubber API request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ProjectCreate(BaseModel):
    """Request schema for creating/opening a project."""

    path: str = Field(..., description="Absolute path to project directory")


class ProjectResponse(BaseModel):
    """Response schema for project information."""

    id: int = Field(..., description="Unique project identifier")
    path: str = Field(..., description="Absolute path to project directory")
    name: str = Field(..., description="Project display name")
    created_at: datetime = Field(..., description="Project creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    voice: str = Field(
        default="", description="TTS voice identifier (e.g., 'nova', 'alloy')"
    )
    voice_instructions: str = Field(
        default="", description="Custom instructions for TTS voice"
    )
    source_language_override: str = Field(
        default="", description="Auto-detected or user-overridden source language (e.g., 'rus', 'eng')"
    )
    target_language: str = Field(
        default="eng",
        description="Target language for dubbing output (ISO 639-2/B code, e.g. 'eng', 'spa', 'fra')",
    )
    working_directory: str = Field(
        default="", description="Resolved working directory for project artefacts"
    )

    @model_validator(mode="after")
    def _set_working_directory(self) -> "ProjectResponse":
        if not self.working_directory:
            try:
                from app.core.project_paths import get_project_working_dir
                self.working_directory = str(get_project_working_dir(self.path, self.name))
            except Exception:
                pass
        return self


class VoiceSettingsUpdate(BaseModel):
    """Request schema for updating project voice settings."""

    voice: str = Field(..., description="TTS voice identifier")
    # Accept both field names for backward compatibility
    instructions: str = Field(
        default="", description="Custom instructions for TTS voice"
    )
    voice_instructions: str = Field(
        default="", description="Custom instructions for TTS voice (alias)"
    )

    segment_used: str = Field(
        default="", description="Segment ID used for voice testing (for history)"
    )

    def get_instructions(self) -> str:
        """Return whichever instructions field is populated."""
        return self.voice_instructions or self.instructions


class SourceLanguageUpdate(BaseModel):
    """Request schema for updating project source language."""

    source_language: str = Field(
        ...,
        description="Source language code (e.g., 'eng', 'rus', 'zho'). Empty string to clear override.",
        max_length=10,
    )


class TargetLanguageUpdate(BaseModel):
    """Request schema for updating project target language."""

    target_language: str = Field(
        ...,
        description="Target language code (ISO 639-2/B, e.g. 'eng', 'spa', 'zho')",
        min_length=2,
        max_length=10,
    )


class AudioStream(BaseModel):
    """Audio stream metadata from video file."""

    index: int = Field(..., description="Stream index in video container")
    language: str = Field(..., description="Language code (e.g., 'en', 'de')")
    codec: str = Field(..., description="Audio codec name (e.g., 'aac', 'mp3')")
    channels: int | str = Field(
        ..., description="Number of audio channels or descriptive string"
    )
    sample_rate: int | str = Field(
        ..., description="Audio sample rate in Hz or descriptive string"
    )


class SubtitleInfo(BaseModel):
    """Subtitle file metadata."""

    language: str = Field(..., description="Detected language code")
    embedded: bool = Field(
        ..., description="True if subtitle is embedded in video container"
    )
    path: str = Field(default="", description="File path for external subtitles")


class PipelineStatusResponse(BaseModel):
    """Pipeline processing status for a video."""

    progress: int = Field(ge=0, le=100, description="Progress percentage (0-100)")
    current_stage: str = Field(..., description="Current pipeline stage name")
    is_complete: bool = Field(..., description="True if all pipeline stages complete")
    failed: bool = Field(default=False, description="True if the last task for this video failed")
    error: str = Field(default="", description="Error message if failed")
    replaced: bool = Field(default=False, description="True if original file has been replaced (finalization complete)")


class VideoAnalysis(BaseModel):
    """Comprehensive video file analysis with pipeline status."""

    id: int = Field(..., description="Unique video analysis record ID")
    filename: str = Field(..., description="Video filename without path")
    path: str = Field(..., description="Absolute path to video file")
    size_mb: float = Field(..., description="File size in megabytes")
    duration_seconds: float = Field(..., description="Video duration in seconds")
    audio_streams: list[AudioStream] = Field(
        ..., description="List of detected audio streams"
    )
    subtitles: list[SubtitleInfo] = Field(
        ..., description="List of available subtitles"
    )
    pipeline_status: PipelineStatusResponse | None = Field(
        default=None, description="Pipeline processing status (null if not started)"
    )


class TaskCreate(BaseModel):
    """Request schema for submitting a redubbing task."""

    video_path: str = Field(..., description="Absolute path to video file")
    project_id: int = Field(..., description="Associated project ID")


class TaskStatusResponse(BaseModel):
    """Response schema for task status polling."""

    task_id: str = Field(..., description="Unique task identifier (UUID)")
    video_path: str = Field(..., description="Target video file path")
    status: Literal["queued", "running", "completed", "failed"] = Field(
        ..., description="Current task status"
    )
    stage: str = Field(..., description="Current processing stage description")
    progress: int = Field(ge=0, le=100, description="Progress percentage (0-100)")
    error: str | None = Field(
        default=None, description="Error message if status is 'failed'"
    )
    created_at: datetime = Field(..., description="Task creation timestamp")
    # Pipeline stage counters
    audio_chunks: int | None = Field(default=None)
    transcripts: int | None = Field(default=None)
    tts_segments: int | None = Field(default=None)
    tts_total: int | None = Field(default=None)
    subtitles: int | None = Field(default=None)
    audio_assembled: int | None = Field(default=None)
    audio_assembled_total: int | None = Field(default=None)
    video_mixed: bool | None = Field(default=None)


class ScanTriggerResponse(BaseModel):
    """Response schema for triggering project scan."""

    project_id: int = Field(..., description="Project ID being scanned")
    status: Literal["started", "already_running"] = Field(
        ..., description="Scan operation status"
    )
    message: str = Field(..., description="Human-readable status message")


class TaskCancelResponse(BaseModel):
    """Response schema for task cancellation."""

    task_id: str = Field(..., description="Cancelled task identifier")
    status: str = Field(..., description="Updated task status after cancellation")
    message: str = Field(..., description="Human-readable confirmation message")
