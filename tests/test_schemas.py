"""Tests for Pydantic schemas and data models."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.schemas.models import (
    AudioStream,
    PipelineStatusResponse,
    ProjectCreate,
    ProjectResponse,
    ScanTriggerResponse,
    SubtitleInfo,
    TaskCancelResponse,
    TaskCreate,
    TaskStatusResponse,
    VideoAnalysis,
    VoiceSettingsUpdate,
)


class TestProjectSchemas:
    """Test suite for project-related schemas."""

    def test_project_create_valid(self) -> None:
        """ProjectCreate accepts valid path."""
        data = ProjectCreate(path="/test/project")
        assert data.path == "/test/project"

    def test_project_create_requires_path(self) -> None:
        """ProjectCreate requires path field."""
        with pytest.raises(ValidationError) as exc_info:
            ProjectCreate()  # type: ignore[call-arg]

        assert "path" in str(exc_info.value)

    def test_project_response_all_fields(self) -> None:
        """ProjectResponse populates all fields correctly."""
        now = datetime.now()
        data = ProjectResponse(
            id=1,
            path="/test/project",
            name="Test Project",
            created_at=now,
            updated_at=now,
            voice="nova",
            voice_instructions="Speak clearly",
        )

        assert data.id == 1
        assert data.path == "/test/project"
        assert data.name == "Test Project"
        assert data.voice == "nova"
        assert data.voice_instructions == "Speak clearly"

    def test_project_response_optional_voice_fields(self) -> None:
        """ProjectResponse voice fields default to empty strings."""
        now = datetime.now()
        data = ProjectResponse(
            id=1,
            path="/test/project",
            name="Test Project",
            created_at=now,
            updated_at=now,
        )

        assert data.voice == ""
        assert data.voice_instructions == ""

    def test_voice_settings_update_valid(self) -> None:
        """VoiceSettingsUpdate accepts valid voice and instructions."""
        data = VoiceSettingsUpdate(voice="alloy", instructions="Professional tone")

        assert data.voice == "alloy"
        assert data.instructions == "Professional tone"

    def test_voice_settings_update_requires_voice(self) -> None:
        """VoiceSettingsUpdate requires voice field."""
        with pytest.raises(ValidationError) as exc_info:
            VoiceSettingsUpdate(instructions="test")  # type: ignore[call-arg]

        assert "voice" in str(exc_info.value)

    def test_voice_settings_instructions_optional(self) -> None:
        """VoiceSettingsUpdate instructions field is optional."""
        data = VoiceSettingsUpdate(voice="nova")
        assert data.instructions == ""


class TestVideoSchemas:
    """Test suite for video-related schemas."""

    def test_audio_stream_valid(self) -> None:
        """AudioStream accepts valid audio metadata."""
        data = AudioStream(
            index=0, language="en", codec="aac", channels=2, sample_rate=48000
        )

        assert data.index == 0
        assert data.language == "en"
        assert data.codec == "aac"
        assert data.channels == 2
        assert data.sample_rate == 48000

    def test_audio_stream_flexible_types(self) -> None:
        """AudioStream accepts int or str for channels and sample_rate."""
        # Integer values
        data1 = AudioStream(
            index=0, language="en", codec="aac", channels=2, sample_rate=48000
        )
        assert data1.channels == 2
        assert data1.sample_rate == 48000

        # String values (for descriptive formats)
        data2 = AudioStream(
            index=1,
            language="de",
            codec="mp3",
            channels="stereo",
            sample_rate="44.1kHz",
        )
        assert data2.channels == "stereo"
        assert data2.sample_rate == "44.1kHz"

    def test_subtitle_info_embedded(self) -> None:
        """SubtitleInfo correctly represents embedded subtitles."""
        data = SubtitleInfo(language="en", embedded=True)

        assert data.language == "en"
        assert data.embedded is True
        assert data.path == ""

    def test_subtitle_info_external(self) -> None:
        """SubtitleInfo correctly represents external subtitle files."""
        data = SubtitleInfo(language="de", embedded=False, path="/test/subtitles.srt")

        assert data.language == "de"
        assert data.embedded is False
        assert data.path == "/test/subtitles.srt"

    def test_pipeline_status_response_valid(self) -> None:
        """PipelineStatusResponse accepts valid progress data."""
        data = PipelineStatusResponse(
            progress=50, current_stage="Generate TTS", is_complete=False
        )

        assert data.progress == 50
        assert data.current_stage == "Generate TTS"
        assert data.is_complete is False

    def test_pipeline_status_progress_bounds(self) -> None:
        """PipelineStatusResponse validates progress is between 0-100."""
        # Valid boundaries
        PipelineStatusResponse(progress=0, current_stage="Start", is_complete=False)
        PipelineStatusResponse(progress=100, current_stage="Complete", is_complete=True)

        # Invalid: below 0
        with pytest.raises(ValidationError):
            PipelineStatusResponse(
                progress=-1, current_stage="Invalid", is_complete=False
            )

        # Invalid: above 100
        with pytest.raises(ValidationError):
            PipelineStatusResponse(
                progress=101, current_stage="Invalid", is_complete=False
            )

    def test_video_analysis_complete(self) -> None:
        """VideoAnalysis accepts complete video metadata."""
        audio = AudioStream(
            index=0, language="en", codec="aac", channels=2, sample_rate=48000
        )
        subtitle = SubtitleInfo(language="en", embedded=False, path="/test/sub.srt")
        status = PipelineStatusResponse(
            progress=75, current_stage="Mix Final", is_complete=False
        )

        data = VideoAnalysis(
            id=1,
            filename="test.mp4",
            path="/test/test.mp4",
            size_mb=150.5,
            duration_seconds=3600.0,
            audio_streams=[audio],
            subtitles=[subtitle],
            pipeline_status=status,
        )

        assert data.id == 1
        assert data.filename == "test.mp4"
        assert len(data.audio_streams) == 1
        assert len(data.subtitles) == 1
        assert data.pipeline_status is not None
        assert data.pipeline_status.progress == 75

    def test_video_analysis_no_pipeline_status(self) -> None:
        """VideoAnalysis allows null pipeline_status for unprocessed videos."""
        data = VideoAnalysis(
            id=2,
            filename="new.mp4",
            path="/test/new.mp4",
            size_mb=100.0,
            duration_seconds=1800.0,
            audio_streams=[],
            subtitles=[],
            pipeline_status=None,
        )

        assert data.pipeline_status is None

    def test_scan_trigger_response_valid(self) -> None:
        """ScanTriggerResponse accepts valid scan status."""
        data = ScanTriggerResponse(
            project_id=1, status="started", message="Scan initiated successfully"
        )

        assert data.project_id == 1
        assert data.status == "started"
        assert data.message == "Scan initiated successfully"

    def test_scan_trigger_status_literal(self) -> None:
        """ScanTriggerResponse validates status is valid literal."""
        # Valid values
        ScanTriggerResponse(project_id=1, status="started", message="OK")
        ScanTriggerResponse(project_id=1, status="already_running", message="OK")

        # Invalid value
        with pytest.raises(ValidationError):
            ScanTriggerResponse(project_id=1, status="invalid", message="OK")  # type: ignore[arg-type]


class TestTaskSchemas:
    """Test suite for task-related schemas."""

    def test_task_create_valid(self) -> None:
        """TaskCreate accepts valid task data."""
        data = TaskCreate(video_path="/test/video.mp4", project_id=1)

        assert data.video_path == "/test/video.mp4"
        assert data.project_id == 1

    def test_task_create_requires_both_fields(self) -> None:
        """TaskCreate requires both video_path and project_id."""
        with pytest.raises(ValidationError):
            TaskCreate(video_path="/test/video.mp4")  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            TaskCreate(project_id=1)  # type: ignore[call-arg]

    def test_task_status_response_queued(self) -> None:
        """TaskStatusResponse correctly represents queued task."""
        now = datetime.now()
        data = TaskStatusResponse(
            task_id="abc-123",
            video_path="/test/video.mp4",
            status="queued",
            stage="Waiting",
            progress=0,
            created_at=now,
        )

        assert data.status == "queued"
        assert data.error is None

    def test_task_status_response_failed(self) -> None:
        """TaskStatusResponse correctly represents failed task with error."""
        now = datetime.now()
        data = TaskStatusResponse(
            task_id="abc-123",
            video_path="/test/video.mp4",
            status="failed",
            stage="Transcribe",
            progress=25,
            error="OpenAI API timeout",
            created_at=now,
        )

        assert data.status == "failed"
        assert data.error == "OpenAI API timeout"

    def test_task_status_valid_literals(self) -> None:
        """TaskStatusResponse validates status literal values."""
        now = datetime.now()

        # All valid values
        for status_val in ["queued", "running", "completed", "failed"]:
            TaskStatusResponse(
                task_id="test",
                video_path="/test",
                status=status_val,  # type: ignore[arg-type]
                stage="Test",
                progress=0,
                created_at=now,
            )

        # Invalid value
        with pytest.raises(ValidationError):
            TaskStatusResponse(
                task_id="test",
                video_path="/test",
                status="invalid",  # type: ignore[arg-type]
                stage="Test",
                progress=0,
                created_at=now,
            )

    def test_task_cancel_response_valid(self) -> None:
        """TaskCancelResponse accepts valid cancellation data."""
        data = TaskCancelResponse(
            task_id="abc-123", status="cancelled", message="Task cancelled successfully"
        )

        assert data.task_id == "abc-123"
        assert data.status == "cancelled"
        assert data.message == "Task cancelled successfully"


class TestSchemaSerializationDeserialization:
    """Test schema JSON serialization and deserialization."""

    def test_project_response_json_serialization(self) -> None:
        """ProjectResponse correctly serializes to JSON."""
        now = datetime.now()
        data = ProjectResponse(
            id=1,
            path="/test",
            name="Test",
            created_at=now,
            updated_at=now,
            voice="nova",
            voice_instructions="Test",
        )

        json_data = data.model_dump()
        assert json_data["id"] == 1
        assert json_data["voice"] == "nova"

    def test_video_analysis_json_roundtrip(self) -> None:
        """VideoAnalysis correctly round-trips through JSON."""
        audio = AudioStream(
            index=0, language="en", codec="aac", channels=2, sample_rate=48000
        )
        original = VideoAnalysis(
            id=1,
            filename="test.mp4",
            path="/test",
            size_mb=100.0,
            duration_seconds=1800.0,
            audio_streams=[audio],
            subtitles=[],
            pipeline_status=None,
        )

        # Serialize to dict
        json_data = original.model_dump()

        # Deserialize back
        restored = VideoAnalysis(**json_data)

        assert restored.id == original.id
        assert restored.filename == original.filename
        assert len(restored.audio_streams) == 1
        assert restored.audio_streams[0].language == "en"
