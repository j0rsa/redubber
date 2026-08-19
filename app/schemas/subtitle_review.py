"""Pydantic schemas for the subtitle review screen."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubtitleReviewFileOption(BaseModel):
    """One subtitle file available for review."""

    path: str = Field(..., description="Absolute path to the subtitle file")
    label: str = Field(..., description="Short display label (usually the filename)")
    source: str = Field(
        ...,
        description="Where the file lives: generated, sidecar, or working_dir",
    )


class SubtitleReviewHallucinationWarning(BaseModel):
    """One STT-quality warning detected in subtitle text."""

    code: str = Field(..., description="Heuristic code, e.g. known_hallucination_phrase")
    message: str = Field(..., description="Human-readable explanation")
    segment_index: int | None = Field(
        default=None,
        description="0-based cue index when the warning applies to one cue",
    )


class SubtitleReviewOriginalAudio(BaseModel):
    """How to seek and play a cue from a source audio chunk."""

    chunk_url: str = Field(..., description="URL of the source audio chunk file")
    chunk_name: str = Field(
        ..., description="Chunk filename within 01_source_audio_chunks"
    )
    seek_start: float = Field(
        ..., description="Start offset within the chunk (seconds)"
    )
    seek_end: float = Field(..., description="End offset within the chunk (seconds)")


class SubtitleReviewSegment(BaseModel):
    """One subtitle cue, aligned with TTS index when a TTS file exists."""

    index: int = Field(..., description="0-based cue index (matches TTS filename)")
    start: float = Field(..., description="Cue start on the video timeline (seconds)")
    end: float = Field(..., description="Cue end on the video timeline (seconds)")
    duration: float = Field(..., description="Cue duration in seconds")
    text: str = Field(..., description="Subtitle text")
    original: SubtitleReviewOriginalAudio | None = Field(
        default=None,
        description="Present when a source audio chunk covers this cue",
    )
    tts_url: str | None = Field(
        default=None,
        description="URL of the matching TTS segment, if generated",
    )


class SubtitleReviewResponse(BaseModel):
    """Full generated script for a video, with playback handles."""

    video_id: int
    filename: str
    srt_path: str
    available_files: list[SubtitleReviewFileOption] = Field(
        default_factory=list,
        description="All subtitle files detected for this video",
    )
    segments: list[SubtitleReviewSegment]
    total: int
    returned: int
    has_chunks: bool = Field(
        default=False, description="True if source audio chunks are available"
    )
    has_tts: bool = Field(
        default=False, description="True if any TTS segment files are available"
    )
    hallucination_warnings: list[SubtitleReviewHallucinationWarning] = Field(
        default_factory=list,
        description="STT-quality warnings found in the loaded subtitle text",
    )
