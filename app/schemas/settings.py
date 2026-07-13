"""Pydantic schemas for tool-level settings API request/response models."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Valid TTS model identifiers
TtsModel = Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]

# Valid OpenAI TTS voice identifiers
VoiceName = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


class Settings(BaseModel):
    """Persistent tool-level settings for the Redubber application.

    Stored as JSON on disk and loaded at startup. All fields have sane
    defaults so the app works out of the box without any configuration.
    The OpenAI API key is the only sensitive field and is masked on read.
    """

    # AI settings
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Masked on GET (last 4 chars only).",
    )
    openai_base_url: str = Field(
        default="",
        description=(
            "Custom OpenAI-compatible API base URL. "
            "Empty string uses the default https://api.openai.com/v1 endpoint. "
            "Set to a custom URL to use a compatible provider (e.g. Azure, local LM Studio)."
        ),
    )
    stt_model: str = Field(
        default="whisper-1",
        description=(
            "OpenAI model used for speech-to-text transcription. "
            "Only 'whisper-1' is supported — it is the only model that returns per-segment "
            "timestamps (verbose_json format), which the pipeline requires to align TTS audio "
            "with original video timing. gpt-4o-transcribe models produce no timestamps."
        ),
    )
    tts_model: TtsModel = Field(
        default="gpt-4o-mini-tts",
        description="OpenAI TTS model to use for speech synthesis.",
    )
    voice_analysis_model: str = Field(
        default="o4-mini",
        description="LLM model used for AI-powered voice instruction generation.",
    )
    voice_analysis_audio_model: str = Field(
        default="gpt-audio-1",
        description="Multimodal model used when audio is available for voice analysis (gender/accent detection). Options: 'gpt-audio-1' (best accuracy), 'gpt-audio-mini' (faster, cheaper).",
    )
    default_voice: VoiceName = Field(
        default="nova",
        description="Default TTS voice applied to new projects.",
    )

    # Operational settings
    tts_concurrency: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of parallel threads for TTS segment generation. Higher = faster but more API load.",
    )
    openai_timeout: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Timeout in seconds for OpenAI API requests.",
    )
    openai_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retries for failed OpenAI API requests.",
    )
    tts_speed: float = Field(
        default=1.25,
        ge=0.5,
        le=2.0,
        description="TTS audio speed multiplier. 1.25 helps dubs fit original timing. 1.0 = natural pace.",
    )
    audio_chunk_duration: int = Field(
        default=900,  # 15 minutes
        ge=60,
        le=3600,
        description="Duration in seconds for audio chunks sent to Whisper. Max ~25MB per chunk.",
    )

    # Workspace settings
    projects_root_path: str = Field(
        default_factory=lambda: os.environ.get("REDUBBER_PROJECTS_ROOT", ""),
        description=(
            "Starting directory for the file browser when creating a new project. "
            "Defaults to REDUBBER_PROJECTS_ROOT env var if set, otherwise the filesystem root."
        ),
    )
    working_directory: str = Field(
        default_factory=lambda: os.environ.get("REDUBBER_WORKING_DIR", ""),
        description=(
            "Root directory where project folders are created. "
            "Defaults to REDUBBER_WORKING_DIR env var if set, otherwise empty (uses app working dir). "
            "Can be overridden in settings."
        ),
    )
    auto_process: bool = Field(
        default=False,
        description="When enabled, automatically runs all redub steps and replaces the original file.",
    )


class SettingsUpdate(BaseModel):
    """Partial update schema for tool-level settings.

    All fields are optional. Only provided fields are updated; the rest
    retain their current values.
    """

    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key. Masked on GET (last 4 chars only).",
    )
    openai_base_url: str | None = Field(
        default=None,
        description="Custom OpenAI-compatible API base URL. Empty uses the default endpoint.",
    )
    stt_model: str | None = Field(
        default=None,
        description="OpenAI STT model for transcription.",
    )
    tts_model: TtsModel | None = Field(
        default=None,
        description="OpenAI TTS model to use for speech synthesis.",
    )
    voice_analysis_model: str | None = Field(
        default=None,
        description="LLM model used for AI-powered voice instruction generation.",
    )
    voice_analysis_audio_model: str | None = Field(
        default=None,
        description="Multimodal model used when audio is available for voice analysis.",
    )
    default_voice: VoiceName | None = Field(
        default=None,
        description="Default TTS voice applied to new projects.",
    )
    tts_concurrency: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of parallel threads for TTS segment generation. Higher = faster but more API load.",
    )
    openai_timeout: float | None = Field(
        default=None,
        ge=5.0,
        le=600.0,
        description="Timeout in seconds for OpenAI API requests.",
    )
    openai_retries: int | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Number of retries for failed OpenAI API requests.",
    )
    tts_speed: float | None = Field(
        default=None,
        ge=0.5,
        le=2.0,
        description="TTS audio speed multiplier. 1.25 helps dubs fit original timing. 1.0 = natural pace.",
    )
    audio_chunk_duration: int | None = Field(
        default=None,
        ge=60,
        le=3600,
        description="Duration in seconds for audio chunks sent to Whisper. Max ~25MB per chunk.",
    )
    projects_root_path: str | None = Field(
        default=None,
        description="Starting directory for the file browser. Set to empty string to fall back to REDUBBER_PROJECTS_ROOT env var.",
    )
    working_directory: str | None = Field(
        default=None,
        description=(
            "Root directory where project folders are created. "
            "Set to empty string to fall back to REDUBBER_WORKING_DIR env var or app working dir."
        ),
    )
    auto_process: bool | None = Field(
        default=None,
        description="When enabled, automatically runs all redub steps and replaces the original file.",
    )


class SettingsResponse(BaseModel):
    """Response schema for tool-level settings.

    Identical to Settings except the OpenAI API key is masked:
    - Empty string if no key is set.
    - ``"sk-...{last4}"`` if a key is stored (full key is never returned).
    """

    # AI settings
    openai_api_key: str = Field(
        description=(
            "Masked OpenAI API key. "
            'Returns empty string when not set, or "sk-...{last4}" when set.'
        ),
        examples=["", "sk-...xYzW"],
    )
    openai_base_url: str = Field(
        description="Custom API base URL. Empty string = default OpenAI endpoint.",
        examples=["", "https://my-proxy.example.com/v1"],
    )
    stt_model: str = Field(
        description="OpenAI STT model for transcription.",
        examples=["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
    )
    tts_model: TtsModel = Field(
        description="OpenAI TTS model used for speech synthesis.",
        examples=["tts-1", "tts-1-hd", "gpt-4o-mini-tts"],
    )
    voice_analysis_model: str = Field(
        description="LLM model used for AI-powered voice instruction generation.",
        examples=["gpt-4o", "gpt-4o-mini"],
    )
    voice_analysis_audio_model: str = Field(
        default="gpt-4o-audio-preview",
        description="Multimodal model used when audio is available for voice analysis.",
        examples=["gpt-4o-audio-preview"],
    )
    default_voice: VoiceName = Field(
        description="Default TTS voice applied to new projects.",
        examples=["nova", "alloy", "shimmer"],
    )

    # Operational settings
    tts_concurrency: int = Field(
        description="Number of parallel threads for TTS segment generation.",
        examples=[20, 50],
    )
    openai_timeout: float = Field(
        description="Timeout in seconds for OpenAI API requests.",
        examples=[60.0, 120.0],
    )
    openai_retries: int = Field(
        description="Number of retries for failed OpenAI API requests.",
        examples=[3, 5],
    )
    tts_speed: float = Field(
        description="TTS audio speed multiplier. 1.25 helps dubs fit original timing.",
        examples=[1.0, 1.25],
    )
    audio_chunk_duration: int = Field(
        description="Duration in seconds for audio chunks sent to Whisper.",
        examples=[900, 1800],
    )

    # Workspace settings
    projects_root_path: str = Field(
        description=(
            "Starting directory for the file browser when creating a new project. "
            "Populated from REDUBBER_PROJECTS_ROOT env var if not explicitly set."
        ),
        examples=["", "/Users/alice/videos"],
    )
    working_directory: str = Field(
        description=(
            "Root directory where project folders are created. "
            "Populated from REDUBBER_WORKING_DIR env var if not explicitly set."
        ),
        examples=["", "/Users/alice/videos"],
    )
    auto_process: bool = Field(
        description="Whether fully automated redubbing is enabled.",
        examples=[False, True],
    )

    # Fields whose values are controlled by an environment variable.
    # When non-empty, the UI should show those fields as read-only.
    env_overrides: list[str] = Field(
        default_factory=list,
        description=(
            "List of setting field names whose values are set by an environment variable "
            "and cannot be changed via the UI (e.g. ['openai_api_key', 'working_directory'])."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "openai_api_key": "sk-...xYzW",
                    "tts_model": "tts-1",
                    "voice_analysis_model": "gpt-4o",
                    "default_voice": "nova",
                    "working_directory": "",
                    "auto_process": False,
                    "env_overrides": ["openai_api_key"],
                }
            ]
        }
    }

    @field_validator("openai_api_key")
    @classmethod
    def key_must_be_masked(cls, v: str) -> str:
        """Ensure that a full API key is never accidentally stored here.

        This validator is a safety net; masking must be applied by the
        service layer before constructing a ``SettingsResponse``.

        Args:
            v: The value supplied for openai_api_key.

        Returns:
            The original value if it looks masked or is empty.

        Raises:
            ValueError: If the value appears to be a raw API key.
        """
        # A raw OpenAI key starts with "sk-" and is much longer than 12 chars
        if v.startswith("sk-") and len(v) > 12 and "..." not in v:
            raise ValueError(
                "openai_api_key in SettingsResponse must be masked. "
                "Use SettingsService to build the response."
            )
        return v
