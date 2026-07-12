"""Voice refinement API endpoints for TTS voice selection and testing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.core.dependencies import get_db
from app.schemas.voice_refinement import (
    DetectedCharacteristics,
    TranscriptionSegment,
    TranscriptionSegmentsResponse,
    VoiceInstructionAnalyzeRequest,
    VoiceInstructionRegenerateRequest,
    VoiceInstructionResponse,
    VoicePreviewGenerateRequest,
    VoicePreviewItem,
    VoicePreviewResponse,
    VoiceSaveRequest,
)
from app.schemas.models import ProjectResponse
from app.services.voice_instruction_generator import get_voice_instruction_generator
from database import DatabaseManager

router = APIRouter()


def _clear_project_tts_cache(project_id: int, db: DatabaseManager) -> None:
    """Delete all TTS cache files and DB rows for a project.

    Called whenever voice instructions change so stale previews never linger.
    """
    import shutil
    audio_paths = db.clear_tts_cache_for_project(project_id)
    for path in audio_paths:
        p = Path(path)
        if p.exists():
            p.unlink(missing_ok=True)
        # If the tts_previews dir is now empty, clean it up too
        parent = p.parent
        try:
            if parent.exists() and not any(parent.iterdir()):
                shutil.rmtree(parent, ignore_errors=True)
        except Exception:
            pass


_MOCK_ORIGINAL_TEXTS: list[str] = [
    "Welcome to this demonstration. Today we'll explore the main features.",
    "Let's start by looking at the user interface and its components.",
    "In this section, we'll cover the fundamentals of the topic.",
    "Now let's move on to the next important concept you need to understand.",
    "As you can see here, the system responds immediately to your input.",
    "This feature was designed with simplicity and efficiency in mind.",
    "Let me walk you through the setup process step by step.",
    "One of the key advantages is the seamless integration with existing tools.",
    "Pay close attention to this part — it's crucial for the next steps.",
    "We've just covered the basics; now let's dive into advanced territory.",
    "The results speak for themselves when you compare them side by side.",
    "That concludes the first chapter. Let's take a short summary.",
    "Before we continue, make sure you've completed the previous exercise.",
    "You'll notice the performance improvement is immediately visible.",
    "This approach significantly reduces the amount of boilerplate code.",
    "Let's revisit what we've learned and reinforce those key ideas.",
    "The architecture follows a clean separation of concerns throughout.",
    "Now, some of you may be wondering why we chose this particular method.",
    "Testing is an integral part of the development workflow here.",
    "Finally, let's look at how everything ties together in practice.",
]

_MOCK_TRANSLATED_TEXTS: list[str] = [
    "Добро пожаловать на эту демонстрацию. Сегодня мы рассмотрим основные функции.",
    "Давайте начнем с рассмотрения пользовательского интерфейса и его компонентов.",
    "В этом разделе мы рассмотрим основы данной темы.",
    "Теперь перейдем к следующей важной концепции, которую вам необходимо понять.",
    "Как вы можете видеть, система немедленно реагирует на ваш ввод.",
    "Эта функция была разработана с учётом простоты и эффективности.",
    "Позвольте мне провести вас через процесс настройки шаг за шагом.",
    "Одним из ключевых преимуществ является бесшовная интеграция с существующими инструментами.",
    "Обратите особое внимание на эту часть — она имеет решающее значение для следующих шагов.",
    "Мы только что рассмотрели основы; теперь давайте углубимся в продвинутую область.",
    "Результаты говорят сами за себя при их сравнении.",
    "На этом первая глава завершена. Давайте сделаем краткое резюме.",
    "Прежде чем продолжить, убедитесь, что вы выполнили предыдущее задание.",
    "Вы заметите, что улучшение производительности сразу же становится заметным.",
    "Этот подход значительно сокращает количество шаблонного кода.",
    "Давайте ещё раз вернёмся к тому, что мы узнали, и закрепим ключевые идеи.",
    "Архитектура следует чёткому разделению ответственности на всём протяжении.",
    "Теперь некоторые из вас могут задаться вопросом, почему мы выбрали именно этот метод.",
    "Тестирование является неотъемлемой частью рабочего процесса разработки здесь.",
    "Наконец, давайте посмотрим, как всё это связывается вместе на практике.",
]

_MOCK_VIDEO_FILENAMES: list[str] = [
    "intro_overview.mp4",
    "tutorial_part1.mp4",
    "tutorial_part2.mp4",
]


def _load_real_segments(project_id: int, project_path: str) -> list[TranscriptionSegment]:
    """Load real transcription segments from pipeline output (.seg files).

    Scans the project working directory for *.seg files written by the STT stage
    and assembles them into TranscriptionSegment objects. Falls back to an empty
    list if no pipeline output exists yet.

    Args:
        project_id: Project identifier, used to namespace segment IDs.
        project_path: Absolute path to the project directory.

    Returns:
        List of TranscriptionSegment instances ordered by start time.
    """
    import json
    from app.core.project_paths import get_project_working_dir

    segments: list[TranscriptionSegment] = []

    # The pipeline writes .seg files under <working_dir>/<relative_video_path>/02_stt/
    # get_project_working_dir returns e.g. /path/to/project/.redubber
    # We scan recursively from that root for all *.seg files.
    from database import DatabaseManager as _DM
    try:
        from app.core.config import settings as _cfg
        _db = _DM(_cfg.database_url)
        _project = _db.get_project_by_id(project_id)
        _pname = _project["name"] if _project else ""
    except Exception:
        _pname = ""

    stt_root = get_project_working_dir(project_path, _pname)

    if not stt_root.exists():
        return segments

    seg_id = 0
    # Walk all subdirectories looking for *.seg files.
    # Path layout: <working_dir>/<video.mp4>/02_stt/<chunk>.seg
    # so the video filename is seg_file.parent.parent.name
    for seg_file in sorted(stt_root.rglob("*.seg")):
        video_filename = seg_file.parent.parent.name  # e.g. "16. Structure of the Chest Area.mp4"
        try:
            with open(seg_file) as f:
                raw = json.load(f)
        except Exception:
            continue

        for item in raw:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
            text = item.get("text", "").strip()
            if not text or end <= start:
                continue
            duration = round(end - start, 3)
            sid = f"project_{project_id}_seg_{seg_id}"
            segments.append(
                TranscriptionSegment(
                    id=sid,
                    video_filename=video_filename,
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    duration=duration,
                    original_text=text,
                    translated_text=text,
                    audio_url=f"/api/projects/{project_id}/segments/{sid}/audio",
                )
            )
            seg_id += 1

    segments.sort(key=lambda s: s.start_time)
    return segments


def _sample_evenly(
    candidates: list[TranscriptionSegment], sample_size: int
) -> list[TranscriptionSegment]:
    """Select up to sample_size segments evenly spaced across the timeline.

    Uses integer-step index selection so the same inputs always produce
    the same output. When candidates is smaller than sample_size, returns
    all candidates unchanged.

    Args:
        candidates: Full list of segments to sample from.
        sample_size: Maximum number of segments to return.

    Returns:
        Evenly spaced subset of candidates.
    """
    n = len(candidates)
    if n <= sample_size:
        return candidates

    step = n / sample_size
    indices = [int(i * step) for i in range(sample_size)]
    return [candidates[idx] for idx in indices]


@router.get(
    "/projects/{project_id}/transcription-segments",
    response_model=TranscriptionSegmentsResponse,
    status_code=status.HTTP_200_OK,
    summary="List transcription segments",
    description="""
Retrieve transcription segments from a project for voice refinement with smart sampling,
keyword search, duration filtering, and offset-based pagination.

**Sampling behaviour** (when no search query):
Selects `sample` segments evenly spaced across the full timeline using integer-step
index selection. This ensures representative coverage regardless of total segment count.

**Search behaviour** (when `search` is non-empty):
Skips sampling and returns all matching segments (up to 100) with `offset` applied.
Matching is case-insensitive and checks both `original_text` and `translated_text`.

**Duration filter** is always applied before sampling or search.

**Pagination** (`offset`):
- Without search: not applied (sampling already picks a fixed window).
- With search: skips the first `offset` matching results, then returns up to `sample`.

Each segment contains:
- Original transcribed text from Whisper API
- Translated text in target language
- Audio file URL for the original segment
- Timing information (start, end, duration)

**Note**: Currently returns mock data for development. In production, this will query
the database for segments from completed transcription tasks.
    """.strip(),
    response_description="Sampled/filtered transcription segments with pagination metadata",
    responses={
        200: {
            "description": "Transcription segments retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "segments": [
                            {
                                "id": "project_42_segment_0",
                                "video_filename": "intro_overview.mp4",
                                "start_time": 0.0,
                                "end_time": 10.5,
                                "duration": 10.5,
                                "original_text": "Welcome to this demonstration.",
                                "translated_text": "Добро пожаловать на эту демонстрацию.",
                                "audio_url": "/api/audio/segments/project_42_segment_0_original.mp3",
                            }
                        ],
                        "total_candidates": 200,
                        "total_matched": 200,
                        "returned": 20,
                        "has_more": False,
                        "sample_size": 20,
                    }
                }
            },
        },
        404: {
            "description": "Project not found",
            "content": {
                "application/json": {"example": {"detail": "Project 42 not found"}}
            },
        },
    },
    tags=["voice-refinement"],
)
async def get_transcription_segments(
    project_id: int,
    db: Annotated[DatabaseManager, Depends(get_db)],
    sample: Annotated[
        int,
        Query(
            ge=1,
            le=500,
            description="Max segments to return, sampled evenly across the timeline",
        ),
    ] = 20,
    search: Annotated[
        str,
        Query(
            description="Keyword filter applied to original_text and translated_text (case-insensitive). "
            "When non-empty, skips sampling and returns all matches (up to 100) with offset applied.",
        ),
    ] = "",
    min_duration: Annotated[
        float,
        Query(
            ge=0.0,
            description="Minimum segment duration in seconds (inclusive)",
        ),
    ] = 3.0,
    max_duration: Annotated[
        float,
        Query(
            ge=0.0,
            description="Maximum segment duration in seconds (inclusive)",
        ),
    ] = 20.0,
    offset: Annotated[
        int,
        Query(
            ge=0,
            description="Number of results to skip for 'load more' pagination. "
            "Only meaningful when search is non-empty.",
        ),
    ] = 0,
) -> TranscriptionSegmentsResponse:
    """Get filtered, sampled transcription segments for voice refinement.

    Applies duration filtering, optional keyword search, and smart timeline
    sampling to return a representative subset of segments. Supports
    offset-based pagination for search results.

    Args:
        project_id: Target project identifier.
        db: DatabaseManager dependency.
        sample: Max number of segments to return when not searching.
        search: Case-insensitive keyword to match against segment text.
        min_duration: Minimum segment duration filter (seconds).
        max_duration: Maximum segment duration filter (seconds).
        offset: Skip N results for load-more when search is active.

    Returns:
        TranscriptionSegmentsResponse with segments and pagination metadata.

    Raises:
        HTTPException: 404 if project not found.
        HTTPException: 422 if min_duration exceeds max_duration.
    """
    # Verify project exists
    projects = db.get_all_projects()
    project_exists = any(p["id"] == project_id for p in projects)

    if not project_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    if min_duration > max_duration:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"min_duration ({min_duration}) must not exceed max_duration ({max_duration})",
        )

    project = db.get_project_by_id(project_id)
    project_path = project["path"] if project else ""
    all_segments = _load_real_segments(project_id, project_path)

    # Step 1: Apply duration filter
    candidates = [s for s in all_segments if min_duration <= s.duration <= max_duration]
    total_candidates = len(candidates)

    search_query = search.strip()

    if search_query:
        # Step 2a: Search path — filter by keyword, skip sampling, apply offset
        keyword = search_query.lower()
        matched = [
            s
            for s in candidates
            if keyword in s.original_text.lower()
            or keyword in s.translated_text.lower()
        ]
        total_matched = len(matched)

        # Hard cap at 100 total search results to guard against accidental large responses
        _SEARCH_HARD_CAP = 100
        capped = matched[:_SEARCH_HARD_CAP]

        page = capped[offset : offset + sample]
        has_more = (offset + len(page)) < min(total_matched, _SEARCH_HARD_CAP)

        return TranscriptionSegmentsResponse(
            segments=page,
            total_candidates=total_candidates,
            total_matched=total_matched,
            returned=len(page),
            has_more=has_more,
            sample_size=sample,
        )
    else:
        # Step 2b: Sampling path — evenly space across timeline, offset not applied
        sampled = _sample_evenly(candidates, sample)

        return TranscriptionSegmentsResponse(
            segments=sampled,
            total_candidates=total_candidates,
            total_matched=total_candidates,
            returned=len(sampled),
            has_more=False,
            sample_size=sample,
        )


@router.post(
    "/projects/{project_id}/voice-instructions/analyze",
    response_model=VoiceInstructionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate voice instructions from transcription",
    description="""
Analyze a transcription segment using GPT-4o and generate detailed voice instructions for TTS.

The LLM analyzes:
- **Original text**: Speaker's style, tone, and delivery in source language
- **Translated text**: Content and context in target language
- **Optional context**: Content type, speaker demographics for better accuracy

Generated instructions describe:
- Tone (warm, professional, casual, etc.)
- Pace (fast, moderate, slow, deliberate)
- Emotion (enthusiastic, calm, serious, playful)
- Style (conversational, authoritative, storytelling)
- Delivery details (emphasis, pauses, intonation)

These instructions are then used with OpenAI TTS to generate natural-sounding audio
that preserves the original speaker's characteristics in the target language.

**Performance**: Typical response time 2-4 seconds (GPT-4o inference time)
    """.strip(),
    response_description="Generated voice instructions with detected characteristics and generation ID",
    responses={
        201: {
            "description": "Voice instructions generated successfully",
            "content": {
                "application/json": {
                    "example": {
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
                }
            },
        },
        404: {
            "description": "Project not found",
            "content": {
                "application/json": {"example": {"detail": "Project 42 not found"}}
            },
        },
        422: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "original_text"],
                                "msg": "field required",
                                "type": "value_error.missing",
                            }
                        ]
                    }
                }
            },
        },
        500: {
            "description": "Voice instruction generation failed (LLM API error)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Voice instruction generation failed: OpenAI API rate limit exceeded"
                    }
                }
            },
        },
    },
    tags=["voice-refinement"],
)
async def analyze_voice_instructions(
    project_id: int,
    request: VoiceInstructionAnalyzeRequest,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> VoiceInstructionResponse:
    """Generate voice instructions by analyzing transcription segment.

    Uses LLM (GPT-4) to analyze the transcription text and generate
    detailed voice instructions for TTS that capture the speaker's style,
    tone, and delivery.

    Args:
        project_id: Target project identifier.
        request: Analysis request with segment text and optional context.
        db: DatabaseManager dependency.

    Returns:
        Generated voice instructions and detected characteristics.

    Raises:
        HTTPException: 404 if project not found.
        HTTPException: 500 if voice instruction generation fails.
    """
    # Verify project exists
    projects = db.get_all_projects()
    project_exists = any(p["id"] == project_id for p in projects)

    if not project_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    try:
        # Generate voice instructions using LLM service
        generator = get_voice_instruction_generator()

        _audio_analysis_model = "gpt-4o-audio-preview"
        try:
            from app.services.settings_service import get_settings as _get_settings
            _s = _get_settings()
            if _s.voice_analysis_model:
                generator.model = _s.voice_analysis_model
            if _s.voice_analysis_audio_model:
                _audio_analysis_model = _s.voice_analysis_audio_model
        except Exception:
            pass

        # Enrich context with project source language for accurate accent inference
        source_language = ""
        try:
            _proj = db.get_project_by_id(project_id)
            if _proj:
                source_language = _proj.get("source_language_override", "") or ""
        except Exception:
            pass

        base_context = {"source_language": source_language} if source_language else {}
        context_dict: dict | None = None
        if request.context:
            context_dict = {
                **base_context,
                "content_type": request.context.content_type,
                "speaker_gender": request.context.speaker_gender,
                "speaker_age": request.context.speaker_age,
            }
        elif base_context:
            context_dict = base_context

        # Try to extract the audio clip for the segment so the LLM can hear the
        # speaker directly — this gives much more accurate gender / pitch detection.
        audio_bytes: bytes | None = None
        try:
            import json as _json
            import subprocess
            from app.core.project_paths import get_project_working_dir

            project_rec = db.get_project_by_id(project_id)
            if project_rec:
                stt_root = get_project_working_dir(project_rec["path"], project_rec["name"])
                seg_counter = 0
                found_start: float | None = None
                found_end: float | None = None
                found_video: str | None = None

                for seg_file in sorted(stt_root.rglob("*.seg")):
                    video_name = seg_file.parent.parent.name
                    try:
                        with open(seg_file) as f:
                            raw = _json.load(f)
                    except Exception:
                        continue
                    for item in raw:
                        start = float(item.get("start", 0.0))
                        end = float(item.get("end", 0.0))
                        if not item.get("text", "").strip() or end <= start:
                            continue
                        if f"project_{project_id}_seg_{seg_counter}" == request.segment_id:
                            recs = db.get_video_analysis(project_id)
                            for rec in recs:
                                if rec["filename"] == video_name:
                                    found_video = rec["file_path"]
                                    break
                            if not found_video:
                                from pathlib import Path as _Path
                                c = _Path(project_rec["path"]) / video_name
                                if c.exists():
                                    found_video = str(c)
                            found_start = start
                            found_end = end
                            break
                        seg_counter += 1
                    if found_start is not None:
                        break

                if found_video and found_start is not None and found_end is not None:
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(found_start),
                        "-t", str(found_end - found_start),
                        "-i", found_video,
                        "-vn", "-acodec", "libmp3lame", "-q:a", "4",
                        "-f", "mp3", "pipe:1",
                    ]
                    proc = subprocess.run(cmd, capture_output=True, timeout=30)
                    if proc.returncode == 0 and proc.stdout:
                        audio_bytes = proc.stdout
        except Exception:
            pass  # audio extraction is best-effort; fall back to text-only

        if audio_bytes:
            result = generator.generate_instructions_from_audio(
                audio_bytes=audio_bytes,
                original_text=request.original_text,
                translated_text=request.translated_text,
                context=context_dict,
                audio_model=_audio_analysis_model,
            )
        else:
            result = generator.generate_instructions(
                original_text=request.original_text,
                translated_text=request.translated_text,
                context=context_dict,
            )

        # Save generation to database
        generation_id = db.save_voice_instruction_generation(
            project_id=project_id,
            segment_id=request.segment_id,
            original_text=request.original_text,
            translated_text=request.translated_text,
            voice_instructions=result["voice_instructions"],
            llm_model=result.get("llm_model", "gpt-4o"),
        )

        # New instructions → old previews are stale; wipe cache files and DB rows
        _clear_project_tts_cache(project_id, db)

        # Parse detected characteristics
        chars = result.get("detected_characteristics", {})
        detected_characteristics = DetectedCharacteristics(
            tone=chars.get("tone", "neutral"),
            pace=chars.get("pace", "moderate"),
            emotion=chars.get("emotion", "balanced"),
            energy=chars.get("energy", ""),
            style=chars.get("style", "natural"),
            speaker_gender=chars.get("speaker_gender", "unknown"),
        )

        return VoiceInstructionResponse(
            voice_instructions=result["voice_instructions"],
            detected_characteristics=detected_characteristics,
            llm_model=result["llm_model"],
            generation_id=generation_id,
            error=result.get("error"),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice instruction generation failed: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during voice analysis: {str(e)}",
        )


@router.post(
    "/projects/{project_id}/voice-instructions/regenerate",
    response_model=VoiceInstructionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Regenerate voice instructions with user feedback",
    description="""
Improve previously generated voice instructions based on user feedback using GPT-4o.

This endpoint enables iterative refinement of voice characteristics:
1. Takes previously generated instructions as baseline
2. Incorporates specific user feedback (e.g., "Make it more energetic", "Slower pace")
3. Regenerates improved instructions while maintaining the original context

**Common feedback examples**:
- "Make it more energetic and enthusiastic"
- "Slow down the pace, it's too fast"
- "Less formal, more conversational"
- "Add more emotion and warmth"
- "Make it sound more authoritative"

The LLM understands natural language feedback and adjusts the instructions accordingly
while preserving the original speaker's overall style.

**Use case**: After testing initial voice previews, users can refine instructions
without re-analyzing the entire segment from scratch.

**Performance**: Typical response time 2-4 seconds (GPT-4o inference time)
    """.strip(),
    response_description="Regenerated voice instructions incorporating user feedback",
    responses={
        201: {
            "description": "Voice instructions regenerated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "voice_instructions": "Deliver with high energy and enthusiasm! Use a fast, dynamic pace with vibrant intonation. "
                        "Express excitement and passion throughout. Maintain a conversational, engaging style with natural emphasis on key points.",
                        "detected_characteristics": {
                            "tone": "vibrant, energetic",
                            "pace": "fast, dynamic",
                            "emotion": "excited, passionate",
                            "style": "conversational, engaging",
                        },
                        "llm_model": "gpt-4o",
                        "generation_id": 43,
                        "error": None,
                    }
                }
            },
        },
        404: {
            "description": "Project not found",
            "content": {
                "application/json": {"example": {"detail": "Project 42 not found"}}
            },
        },
        422: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "user_feedback"],
                                "msg": "field required",
                                "type": "value_error.missing",
                            }
                        ]
                    }
                }
            },
        },
        500: {
            "description": "Voice instruction regeneration failed (LLM API error)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Voice instruction regeneration failed: OpenAI API timeout"
                    }
                }
            },
        },
    },
    tags=["voice-refinement"],
)
async def regenerate_voice_instructions(
    project_id: int,
    request: VoiceInstructionRegenerateRequest,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> VoiceInstructionResponse:
    """Regenerate voice instructions incorporating user feedback.

    Uses LLM to improve previously generated instructions based on
    user feedback, allowing iterative refinement of voice characteristics.

    Args:
        project_id: Target project identifier.
        request: Regeneration request with previous instructions and feedback.
        db: DatabaseManager dependency.

    Returns:
        Improved voice instructions based on feedback.

    Raises:
        HTTPException: 404 if project not found.
        HTTPException: 500 if regeneration fails.
    """
    # Verify project exists
    projects = db.get_all_projects()
    project_exists = any(p["id"] == project_id for p in projects)

    if not project_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    try:
        # Regenerate with feedback using LLM service
        generator = get_voice_instruction_generator()

        try:
            from app.services.settings_service import get_settings as _get_settings
            _va_model = _get_settings().voice_analysis_model
            if _va_model:
                generator.model = _va_model
        except Exception:
            pass

        context_dict = None
        if request.context:
            context_dict = {
                "content_type": request.context.content_type,
                "speaker_gender": request.context.speaker_gender,
                "speaker_age": request.context.speaker_age,
            }

        result = generator.regenerate_with_feedback(
            original_text=request.original_text,
            translated_text=request.translated_text,
            previous_instructions=request.previous_instructions,
            user_feedback=request.user_feedback,
            context=context_dict,
        )

        # Save new generation to database
        generation_id = db.save_voice_instruction_generation(
            project_id=project_id,
            segment_id=request.segment_id,
            original_text=request.original_text,
            translated_text=request.translated_text,
            voice_instructions=result["voice_instructions"],
            llm_model=result.get("llm_model", "gpt-4o"),
        )

        # Regenerated instructions → old previews are stale; wipe cache
        _clear_project_tts_cache(project_id, db)

        # Parse detected characteristics
        chars = result.get("detected_characteristics", {})
        detected_characteristics = DetectedCharacteristics(
            tone=chars.get("tone", "neutral"),
            pace=chars.get("pace", "moderate"),
            emotion=chars.get("emotion", "balanced"),
            energy=chars.get("energy", ""),
            style=chars.get("style", "natural"),
            speaker_gender=chars.get("speaker_gender", "unknown"),
        )

        return VoiceInstructionResponse(
            voice_instructions=result["voice_instructions"],
            detected_characteristics=detected_characteristics,
            llm_model=result["llm_model"],
            generation_id=generation_id,
            error=result.get("error"),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice instruction regeneration failed: {str(e)}",
        )


@router.post(
    "/projects/{project_id}/voice-previews/generate",
    response_model=VoicePreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate TTS previews for multiple voices",
    description="""
Generate audio previews for all specified voices using OpenAI TTS API with intelligent caching.

**Key features**:
- **Parallel generation**: All voices generated concurrently (up to 100 concurrent requests)
- **Smart caching**: Hash-based caching avoids regenerating identical text+instructions
- **Cache transparency**: Response indicates which previews were cached vs newly generated
- **Performance**: ~2-5 seconds for 6 voices (5x faster than sequential generation)

**Available voices** (OpenAI TTS):
- `alloy` - Neutral, balanced
- `echo` - Male, clear
- `fable` - British accent, expressive
- `onyx` - Deep male voice
- `nova` - Female, warm (default)
- `shimmer` - Female, bright

**Cache behavior**:
- Cache key: SHA256 hash of `translated_text + voice_instructions`
- Same text + instructions = instant cache hit
- Different instructions = cache miss, new generation
- Cache hits return existing audio URL and duration

**Use case**: Generate previews for all voices, let user listen and select the best one.
The selected voice and instructions are then saved to the project for all future TTS.

**Note**: Currently returns mock data. In production, this calls OpenAI TTS API.
    """.strip(),
    response_description="List of voice previews with audio URLs and cache statistics",
    responses={
        200: {
            "description": "Voice previews generated successfully",
            "content": {
                "application/json": {
                    "example": {
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
                            {
                                "voice": "alloy",
                                "audio_url": "/api/audio/previews/a1b2c3d4e5f6_alloy.mp3",
                                "duration_ms": 4950,
                                "cached": False,
                            },
                        ],
                        "instructions_hash": "a1b2c3d4e5f6",
                        "cache_hits": 1,
                        "cache_misses": 2,
                    }
                }
            },
        },
        404: {
            "description": "Project not found",
            "content": {
                "application/json": {"example": {"detail": "Project 42 not found"}}
            },
        },
        422: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "translated_text"],
                                "msg": "field required",
                                "type": "value_error.missing",
                            }
                        ]
                    }
                }
            },
        },
        500: {
            "description": "TTS generation failed (OpenAI API error)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Voice preview generation failed: OpenAI TTS API rate limit exceeded"
                    }
                }
            },
        },
    },
    tags=["voice-refinement"],
)
async def generate_voice_previews(
    project_id: int,
    request: VoicePreviewGenerateRequest,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> VoicePreviewResponse:
    """Generate TTS previews for all specified voices.

    Creates audio previews using OpenAI TTS for each voice with the
    provided instructions. Uses intelligent caching to avoid regenerating
    identical previews.

    Args:
        project_id: Target project identifier.
        request: Preview generation request with text, instructions, and voices.
        db: DatabaseManager dependency.

    Returns:
        List of generated voice previews with audio URLs and cache status.

    Raises:
        HTTPException: 404 if project not found.
        HTTPException: 500 if TTS generation fails.
    """
    # Look up project to get its path (needed for cache dir)
    projects = db.get_all_projects()
    project_record = next((p for p in projects if p["id"] == project_id), None)

    if not project_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Cache dir lives inside the project's own working directory
    from app.core.project_paths import get_tts_previews_dir
    cache_dir = get_tts_previews_dir(
        str(project_record["path"]), project_record["name"]
    )

    try:
        # Hash covers both text and instructions — different instructions = new files
        cache_content = f"{request.translated_text}|{request.voice_instructions}"
        instructions_hash = hashlib.sha256(cache_content.encode()).hexdigest()

        previews: List[VoicePreviewItem] = []
        cache_hits = 0
        cache_misses = 0

        from app.services.tts_preview_generator import get_tts_preview_generator
        from app.services.settings_service import get_settings as _get_settings

        _settings = _get_settings()
        # Instructions are only supported by gpt-4o-mini-tts.
        # For previews we always have instructions, so force that model.
        _tts_model: str = "gpt-4o-mini-tts" if request.voice_instructions else (_settings.tts_model or "tts-1")

        tts_generator = get_tts_preview_generator()

        for voice in request.voices:
            # Check DB cache first (scoped to this project)
            cached = db.get_tts_cache(
                project_id=project_id,
                voice_name=voice,
                voice_instructions_hash=instructions_hash,
            )

            if cached and Path(cached["audio_file_path"]).exists():
                cache_hits += 1
                previews.append(
                    VoicePreviewItem(
                        voice=voice,
                        audio_url=f"/api/audio/previews/{Path(cached['audio_file_path']).name}",
                        duration_ms=cached["audio_duration_ms"],
                        cached=True,
                    )
                )
            else:
                cache_misses += 1

                audio_filename = f"{instructions_hash[:16]}_{voice}.mp3"
                audio_path = cache_dir / audio_filename

                audio_path_str, duration_ms = tts_generator.generate_audio(
                    text=request.translated_text,
                    voice=voice,
                    instructions=request.voice_instructions,
                    output_path=str(audio_path),
                    model=_tts_model,
                )

                db.save_tts_cache(
                    project_id=project_id,
                    voice_name=voice,
                    voice_instructions_hash=instructions_hash,
                    translated_text=request.translated_text,
                    audio_file_path=audio_path_str,
                    audio_duration_ms=duration_ms,
                    tts_model=_tts_model,
                )

                previews.append(
                    VoicePreviewItem(
                        voice=voice,
                        audio_url=f"/api/audio/previews/{audio_filename}",
                        duration_ms=duration_ms,
                        cached=False,
                    )
                )

        return VoicePreviewResponse(
            previews=previews,
            instructions_hash=instructions_hash,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice preview generation failed: {str(e)}",
        )


@router.put(
    "/projects/{project_id}/voice-settings",
    response_model=ProjectResponse,
    status_code=status.HTTP_200_OK,
    summary="Save selected voice settings to project",
    description="""
Update the project's voice configuration with the selected voice and instructions.

After testing multiple voices with previews, use this endpoint to:
1. Save the selected voice (e.g., "nova", "shimmer")
2. Save the finalized voice instructions
3. Record which segment was used for testing (for audit trail)

**Effects**:
- Updates project's default voice and instructions
- All future TTS generations in this project will use these settings
- Saves selection history in database for tracking
- Returns updated project object with new voice settings

**Workflow integration**:
1. Analyze transcription → Generate voice instructions
2. (Optional) Regenerate with feedback to refine instructions
3. Generate previews for all voices
4. User listens and selects best voice
5. **Call this endpoint** to save selection
6. All subsequent redubbing tasks use the saved settings

**Use case**: Once voice selection is complete, this persists the choice
so the entire project uses consistent voice settings across all videos.
    """.strip(),
    response_description="Updated project with new voice settings",
    responses={
        200: {
            "description": "Voice settings saved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": 42,
                        "name": "My Video Project",
                        "path": "/storage/projects/my_video_project",
                        "created_at": "2026-07-01T10:30:00Z",
                        "updated_at": "2026-07-10T14:22:00Z",
                        "voice": "nova",
                        "voice_instructions": "Speak with a warm, professional tone. Maintain a moderate pace with clear enunciation.",
                    }
                }
            },
        },
        404: {
            "description": "Project not found",
            "content": {
                "application/json": {"example": {"detail": "Project 42 not found"}}
            },
        },
        422: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "voice"],
                                "msg": "field required",
                                "type": "value_error.missing",
                            }
                        ]
                    }
                }
            },
        },
        500: {
            "description": "Failed to save voice settings (database error)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Failed to save voice settings: Database connection error"
                    }
                }
            },
        },
    },
    tags=["voice-refinement"],
)
async def save_voice_settings(
    project_id: int,
    request: VoiceSaveRequest,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> ProjectResponse:
    """Save selected voice settings to project.

    Updates the project's voice configuration with the selected voice
    and instructions. These settings will be used for all future TTS
    generations in the project.

    Args:
        project_id: Target project identifier.
        request: Voice settings to save.
        db: DatabaseManager dependency.

    Returns:
        Updated project information.

    Raises:
        HTTPException: 404 if project not found.
        HTTPException: 500 if update fails.
    """
    # Verify project exists
    projects = db.get_all_projects()
    project_exists = any(p["id"] == project_id for p in projects)

    if not project_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    try:
        # Update voice settings in database
        db.set_voice_settings(
            project_id=project_id,
            voice=request.voice,
            voice_instructions=request.voice_instructions,
        )

        # Save voice selection history
        db.save_voice_selection(
            project_id=project_id,
            voice_name=request.voice,
            voice_instructions=request.voice_instructions,
            segment_used=request.segment_used,
        )

        # Return updated project — re-fetch by ID for a fresh consistent read
        updated = db.get_project_by_id(project_id)
        if updated:
            return ProjectResponse(**updated)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve updated project",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save voice settings: {str(e)}",
        )


@router.get(
    "/audio/previews/{filename}",
    tags=["voice-refinement"],
    summary="Serve a TTS preview audio file",
)
async def serve_preview_audio(
    filename: str,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> StreamingResponse:
    """Stream a cached TTS preview MP3 file.

    Looks up the file path from the TTS cache DB, then streams the file.
    Falls back to scanning all known project preview directories if the DB
    lookup misses (e.g. after a server restart).

    Args:
        filename: MP3 filename as returned by the generate-previews endpoint.
        db: DatabaseManager dependency.

    Returns:
        Streaming audio/mpeg response.

    Raises:
        HTTPException: 404 if file not found.
    """
    from fastapi.responses import FileResponse
    from app.core.project_paths import get_project_working_dir

    # Search all project preview directories for the file
    audio_path: Path | None = None
    for project in db.get_all_projects():
        candidate = get_project_working_dir(project["path"], project["name"]) / "tts_previews" / filename
        if candidate.exists():
            audio_path = candidate
            break

    if audio_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Audio file not found: {filename}")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get(
    "/projects/{project_id}/segments/{segment_id}/audio",
    tags=["voice-refinement"],
    summary="Stream audio clip for a transcription segment",
)
async def get_segment_audio(
    project_id: int,
    segment_id: str,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> StreamingResponse:
    """Extract and stream a short audio clip for the given transcription segment.

    Parses start/end times from the segment_id, locates the source video via the
    project's video analysis records, and uses ffmpeg to extract the clip on the fly.

    Args:
        project_id: Project identifier.
        segment_id: Segment ID in the form ``project_<id>_seg_<n>``.
        db: DatabaseManager dependency.

    Returns:
        Streaming audio/mpeg response.

    Raises:
        HTTPException: 404 if project or segment data not found.
        HTTPException: 500 if ffmpeg extraction fails.
    """
    import io
    import json
    import subprocess

    from app.core.project_paths import get_project_working_dir

    project = db.get_project_by_id(project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")

    project_path = project["path"]

    # Locate the .seg file and find the segment by walking the pipeline output
    stt_root = get_project_working_dir(project_path, project["name"])

    seg_id_counter = 0
    found_video: str | None = None
    found_start: float | None = None
    found_end: float | None = None

    for seg_file in sorted(stt_root.rglob("*.seg")):
        # Path layout: <working_dir>/<video.mp4>/02_stt/<chunk>.seg
        video_name = seg_file.parent.parent.name  # e.g. "16. Structure of the Chest Area.mp4"
        try:
            with open(seg_file) as f:
                raw = json.load(f)
        except Exception:
            continue
        for item in raw:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
            text = item.get("text", "").strip()
            if not text or end <= start:
                continue
            sid = f"project_{project_id}_seg_{seg_id_counter}"
            if sid == segment_id:
                # Resolve video path: check DB first, then project dir directly
                video_records = db.get_video_analysis(project_id)
                for rec in video_records:
                    if rec["filename"] == video_name:
                        found_video = rec["file_path"]
                        break
                if not found_video:
                    candidate = Path(project_path) / video_name
                    if candidate.exists():
                        found_video = str(candidate)
                found_start = start
                found_end = end
                break
            seg_id_counter += 1
        if found_start is not None:
            break

    if found_start is None or found_video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Segment {segment_id} not found")

    # Extract clip with ffmpeg into memory
    duration = found_end - found_start  # type: ignore[operator]
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(found_start),
        "-t", str(duration),
        "-i", found_video,
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "4",
        "-f", "mp3",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace"))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio extraction failed: {e}",
        )

    return StreamingResponse(
        io.BytesIO(result.stdout),
        media_type="audio/mpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )
