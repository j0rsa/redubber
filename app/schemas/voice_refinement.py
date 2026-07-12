"""Pydantic schemas for voice refinement API request/response models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    """Transcription segment from video processing."""

    id: str = Field(
        ...,
        description="Unique segment identifier (e.g., 'video1_segment_0')",
        examples=["project_42_segment_0", "intro_video_segment_1"],
    )
    video_filename: str = Field(
        ...,
        description="Source video filename",
        examples=["intro_video.mp4", "tutorial_part1.mov"],
    )
    start_time: float = Field(
        ...,
        description="Segment start time in seconds",
        examples=[0.0, 15.5, 120.75],
        ge=0.0,
    )
    end_time: float = Field(
        ...,
        description="Segment end time in seconds",
        examples=[10.5, 30.2, 145.8],
        gt=0.0,
    )
    duration: float = Field(
        ...,
        description="Segment duration in seconds",
        examples=[10.5, 14.7, 25.05],
        gt=0.0,
    )
    original_text: str = Field(
        ...,
        description="Original transcribed text",
        examples=[
            "Welcome to this demonstration. Today we'll explore the main features.",
            "In this tutorial, we'll learn how to build a web application.",
        ],
    )
    translated_text: str = Field(
        ...,
        description="Translated text in target language",
        examples=[
            "Добро пожаловать на эту демонстрацию. Сегодня мы рассмотрим основные функции.",
            "En este tutorial, aprenderemos cómo construir una aplicación web.",
        ],
    )
    audio_url: str = Field(
        ...,
        description="URL to original audio segment",
        examples=[
            "/api/audio/segments/project_42_segment_0_original.mp3",
            "/api/audio/segments/intro_video_segment_1_original.wav",
        ],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "project_42_segment_0",
                    "video_filename": "intro_video.mp4",
                    "start_time": 0.0,
                    "end_time": 10.5,
                    "duration": 10.5,
                    "original_text": "Welcome to this demonstration. Today we'll explore the main features.",
                    "translated_text": "Добро пожаловать на эту демонстрацию. Сегодня мы рассмотрим основные функции.",
                    "audio_url": "/api/audio/segments/project_42_segment_0_original.mp3",
                }
            ]
        }
    }


class TranscriptionSegmentsResponse(BaseModel):
    """Paginated and sampled response for transcription segments.

    Supports smart sampling across the timeline and keyword search,
    enabling efficient browsing of large segment collections.
    """

    segments: list[TranscriptionSegment] = Field(
        ...,
        description="Transcription segments in this response batch",
    )
    total_candidates: int = Field(
        ...,
        description="Total segments after duration filter, before search/sampling",
        examples=[200, 87, 15],
        ge=0,
    )
    total_matched: int = Field(
        ...,
        description="Total segments after search filter (equals total_candidates when no search)",
        examples=[200, 12, 3],
        ge=0,
    )
    returned: int = Field(
        ...,
        description="Number of segments in this response",
        examples=[20, 12, 3],
        ge=0,
    )
    has_more: bool = Field(
        ...,
        description="Whether more results exist beyond offset + returned",
        examples=[True, False],
    )
    sample_size: int = Field(
        ...,
        description="The requested sample size query parameter",
        examples=[20, 50, 100],
        gt=0,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "segments": [],
                    "total_candidates": 200,
                    "total_matched": 200,
                    "returned": 20,
                    "has_more": False,
                    "sample_size": 20,
                }
            ]
        }
    }


class VoiceInstructionContext(BaseModel):
    """Optional context for voice instruction generation."""

    content_type: str = Field(
        default="general",
        description="Content type (e.g., 'educational', 'entertainment', 'news')",
        examples=["educational", "entertainment", "news", "documentary", "tutorial"],
    )
    speaker_gender: str = Field(
        default="unknown",
        description="Speaker gender ('male', 'female', 'unknown')",
        examples=["male", "female", "unknown"],
    )
    speaker_age: str = Field(
        default="adult",
        description="Speaker age ('child', 'young_adult', 'adult', 'senior')",
        examples=["child", "young_adult", "adult", "senior"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "content_type": "educational",
                    "speaker_gender": "male",
                    "speaker_age": "adult",
                }
            ]
        }
    }


class VoiceInstructionAnalyzeRequest(BaseModel):
    """Request schema for analyzing voice and generating instructions."""

    segment_id: str = Field(
        ...,
        description="ID of the transcription segment to analyze",
        examples=["project_42_segment_0", "intro_video_segment_1"],
    )
    original_text: str = Field(
        ...,
        description="Original transcribed text",
        examples=[
            "Welcome to this demonstration. Today we'll explore the main features.",
            "In this tutorial, we'll learn step by step.",
        ],
    )
    translated_text: str = Field(
        ...,
        description="Translated text in target language",
        examples=[
            "Добро пожаловать на эту демонстрацию. Сегодня мы рассмотрим основные функции.",
            "En este tutorial, aprenderemos paso a paso.",
        ],
    )
    context: Optional[VoiceInstructionContext] = Field(
        default=None, description="Optional context for better voice analysis"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "segment_id": "project_42_segment_0",
                    "original_text": "Welcome to this demonstration. Today we'll explore the main features.",
                    "translated_text": "Добро пожаловать на эту демонстрацию. Сегодня мы рассмотрим основные функции.",
                    "context": {
                        "content_type": "educational",
                        "speaker_gender": "male",
                        "speaker_age": "adult",
                    },
                }
            ]
        }
    }


class DetectedCharacteristics(BaseModel):
    """Detected voice characteristics from LLM analysis."""

    tone: str = Field(
        ...,
        description="Detected tone (e.g., 'warm, professional')",
        examples=["warm, professional", "friendly, casual", "authoritative, serious"],
    )
    pace: str = Field(
        ...,
        description="Detected pace (e.g., 'moderate')",
        examples=["moderate", "fast", "slow", "deliberate", "energetic"],
    )
    emotion: str = Field(
        ...,
        description="Detected emotion (e.g., 'enthusiastic, engaged')",
        examples=[
            "enthusiastic, engaged",
            "calm, measured",
            "excited, passionate",
            "neutral, informative",
        ],
    )
    energy: str = Field(
        default="",
        description="Detected energy level (e.g., 'calm authority')",
        examples=["calm authority", "high energy", "soft and gentle"],
    )
    speaker_gender: str = Field(
        default="unknown",
        description="Inferred speaker gender: 'male', 'female', or 'unknown'",
        examples=["male", "female", "unknown"],
    )
    style: str = Field(
        ...,
        description="Detected style (e.g., 'conversational, authoritative')",
        examples=[
            "conversational, authoritative",
            "formal, academic",
            "casual, friendly",
            "storytelling, narrative",
        ],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tone": "warm, professional",
                    "pace": "moderate",
                    "emotion": "enthusiastic, engaged",
                    "style": "conversational, authoritative",
                }
            ]
        }
    }


class VoiceInstructionResponse(BaseModel):
    """Response schema for voice instruction generation."""

    voice_instructions: str = Field(
        ...,
        description="Generated voice instructions for TTS",
        examples=[
            "Speak with a warm, professional tone. Maintain a moderate pace with clear enunciation. "
            "Convey enthusiasm and engagement while keeping an authoritative, conversational style. "
            "Emphasize key words naturally and use slight pauses for emphasis."
        ],
    )
    detected_characteristics: DetectedCharacteristics = Field(
        ..., description="Extracted voice characteristics"
    )
    llm_model: str = Field(
        ...,
        description="LLM model used for generation",
        examples=["gpt-4o", "gpt-4o-mini"],
    )
    generation_id: int = Field(
        ...,
        description="Database ID of generation record",
        examples=[1, 42, 123],
        gt=0,
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if generation partially failed",
        examples=[None, "LLM rate limit exceeded, using fallback instructions"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "voice_instructions": "Speak with a warm, professional tone. Maintain a moderate pace with clear enunciation. "
                    "Convey enthusiasm and engagement while keeping an authoritative, conversational style. "
                    "Emphasize key words naturally and use slight pauses for emphasis.",
                    "detected_characteristics": {
                        "tone": "warm, professional",
                        "pace": "moderate",
                        "emotion": "enthusiastic, engaged",
                        "style": "conversational, authoritative",
                    },
                    "llm_model": "gpt-4o",
                    "generation_id": 42,
                    "error": None,
                }
            ]
        }
    }


class VoiceInstructionRegenerateRequest(BaseModel):
    """Request schema for regenerating instructions with user feedback."""

    segment_id: str = Field(
        ...,
        description="ID of the transcription segment",
        examples=["project_42_segment_0"],
    )
    original_text: str = Field(
        ...,
        description="Original transcribed text",
        examples=[
            "Welcome to this demonstration. Today we'll explore the main features."
        ],
    )
    translated_text: str = Field(
        ...,
        description="Translated text",
        examples=[
            "Добро пожаловать на эту демонстрацию. Сегодня мы рассмотрим основные функции."
        ],
    )
    previous_instructions: str = Field(
        ...,
        description="Previously generated instructions",
        examples=[
            "Speak with a warm, professional tone. Maintain a moderate pace with clear enunciation."
        ],
    )
    user_feedback: str = Field(
        ...,
        description="User feedback on what to improve (e.g., 'Make it more energetic')",
        examples=[
            "Make it more energetic and enthusiastic",
            "Slow down the pace slightly",
            "Less formal, more conversational",
            "Add more emotion and excitement",
        ],
    )
    context: Optional[VoiceInstructionContext] = Field(
        default=None, description="Optional context information"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "segment_id": "project_42_segment_0",
                    "original_text": "Welcome to this demonstration. Today we'll explore the main features.",
                    "translated_text": "Добро пожаловать на эту демонстрацию. Сегодня мы рассмотрим основные функции.",
                    "previous_instructions": "Speak with a warm, professional tone. Maintain a moderate pace with clear enunciation.",
                    "user_feedback": "Make it more energetic and enthusiastic",
                    "context": {
                        "content_type": "educational",
                        "speaker_gender": "male",
                        "speaker_age": "adult",
                    },
                }
            ]
        }
    }


class VoicePreviewGenerateRequest(BaseModel):
    """Request schema for generating TTS previews for all voices."""

    translated_text: str = Field(
        ...,
        description="Text to generate audio for",
        examples=[
            "Добро пожаловать на эту демонстрацию. Сегодня мы рассмотрим основные функции.",
            "En este tutorial, aprenderemos paso a paso.",
        ],
    )
    voice_instructions: str = Field(
        ...,
        description="Voice instructions to apply",
        examples=[
            "Speak with a warm, professional tone. Maintain a moderate pace with clear enunciation.",
            "Deliver with high energy and enthusiasm. Use a fast pace with dynamic intonation.",
        ],
    )
    voices: list[str] = Field(
        default=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        description="List of voice names to generate previews for",
        examples=[
            ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            ["nova", "shimmer"],
        ],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "translated_text": "Добро пожаловать на эту демонстрацию. Сегодня мы рассмотрим основные функции.",
                    "voice_instructions": "Speak with a warm, professional tone. Maintain a moderate pace with clear enunciation.",
                    "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                }
            ]
        }
    }


class VoicePreviewItem(BaseModel):
    """Single voice preview result."""

    voice: str = Field(
        ...,
        description="Voice name",
        examples=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    )
    audio_url: str = Field(
        ...,
        description="URL to generated audio file",
        examples=[
            "/api/audio/previews/a1b2c3d4e5f6_nova.mp3",
            "/api/audio/previews/f6e5d4c3b2a1_shimmer.mp3",
        ],
    )
    duration_ms: int = Field(
        ...,
        description="Audio duration in milliseconds",
        examples=[5000, 12500, 8750],
        gt=0,
    )
    cached: bool = Field(
        ...,
        description="True if result was retrieved from cache",
        examples=[True, False],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "voice": "nova",
                    "audio_url": "/api/audio/previews/a1b2c3d4e5f6_nova.mp3",
                    "duration_ms": 5000,
                    "cached": False,
                }
            ]
        }
    }


class VoicePreviewResponse(BaseModel):
    """Response schema for voice preview generation."""

    previews: list[VoicePreviewItem] = Field(
        ..., description="List of generated previews"
    )
    instructions_hash: str = Field(
        ...,
        description="Hash of instructions+text for caching",
        examples=["a1b2c3d4e5f6", "f6e5d4c3b2a1"],
    )
    cache_hits: int = Field(
        ...,
        description="Number of cache hits",
        examples=[0, 2, 6],
        ge=0,
    )
    cache_misses: int = Field(
        ...,
        description="Number of cache misses (new generations)",
        examples=[6, 4, 0],
        ge=0,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "previews": [
                        {
                            "voice": "nova",
                            "audio_url": "/api/audio/previews/a1b2c3d4e5f6_nova.mp3",
                            "duration_ms": 5000,
                            "cached": False,
                        },
                        {
                            "voice": "shimmer",
                            "audio_url": "/api/audio/previews/a1b2c3d4e5f6_shimmer.mp3",
                            "duration_ms": 5100,
                            "cached": True,
                        },
                    ],
                    "instructions_hash": "a1b2c3d4e5f6",
                    "cache_hits": 1,
                    "cache_misses": 1,
                }
            ]
        }
    }


class VoiceSaveRequest(BaseModel):
    """Request schema for saving selected voice settings."""

    voice: str = Field(
        ...,
        description="Selected voice name (e.g., 'nova')",
        examples=["nova", "shimmer", "alloy", "echo", "fable", "onyx"],
    )
    voice_instructions: str = Field(
        ...,
        description="Voice instructions to use",
        examples=[
            "Speak with a warm, professional tone. Maintain a moderate pace with clear enunciation.",
            "Deliver with high energy and enthusiasm. Use a fast pace with dynamic intonation.",
        ],
    )
    segment_used: str = Field(
        default="",
        description="Segment ID that was used for voice testing. Empty when voice was set manually.",
        examples=["project_42_segment_0", "intro_video_segment_1", ""],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "voice": "nova",
                    "voice_instructions": "Speak with a warm, professional tone. Maintain a moderate pace with clear enunciation.",
                    "segment_used": "project_42_segment_0",
                }
            ]
        }
    }
