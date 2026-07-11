"""High-performance async TTS service for parallel OpenAI API calls."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Final

import httpx
from openai import AsyncOpenAI
from openai.types.audio.transcription_segment import TranscriptionSegment

logger: Final = logging.getLogger(__name__)

_MAX_SPEED = 2.0  # Never speed up more than 2x


def _get_audio_duration(path: Path) -> float:
    """Return audio duration in seconds via ffprobe, or 0.0 on failure."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _adjust_speed(input_path: Path, output_path: Path, speed: float) -> None:
    """Speed-adjust audio using ffmpeg atempo (chains filters for speed > 2x)."""
    filters = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.6f}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path),
         "-filter:a", ",".join(filters), "-vn", str(output_path)],
        capture_output=True, check=True, timeout=60,
    )


class AsyncRedubberService:
    """High-performance async TTS service using AsyncOpenAI with connection pooling.

    Processes TTS segments with high concurrency (100+ concurrent API calls) using
    asyncio.Semaphore for rate limiting and httpx connection pooling for performance.

    Performance characteristics:
        - 1600 segments with max_concurrent=100: ~16 seconds
        - 1600 segments with ThreadPoolExecutor(20): ~80 seconds
        - 5x speedup with async + high concurrency

    Attributes:
        client: AsyncOpenAI client with httpx connection pooling
        voice: OpenAI voice name (nova, alloy, echo, fable, onyx, shimmer)
        voice_instructions: Optional voice refinement instructions for gpt-4o-mini-tts
    """

    def __init__(
        self,
        openai_token: str,
        voice: str = "nova",
        voice_instructions: str = "",
        openai_timeout: float = 60.0,
        openai_retries: int = 3,
        tts_model: str = "gpt-4o-mini-tts",
    ) -> None:
        """Initialize AsyncRedubberService with OpenAI client and connection pooling.

        Args:
            openai_token: OpenAI API token for authentication
            voice: Voice name for TTS (default: nova)
            voice_instructions: Optional instructions for voice refinement with gpt-4o-mini-tts
            openai_timeout: Timeout in seconds for OpenAI API requests (default: 60.0)
            openai_retries: Number of retries for failed OpenAI API requests (default: 3)
        """
        # Configure httpx limits for high concurrency
        limits = httpx.Limits(
            max_connections=200,  # High concurrency pool
            max_keepalive_connections=50,  # Reuse connections
            keepalive_expiry=30.0,  # 30s keepalive
        )

        # Create httpx client with connection pooling
        http_client = httpx.AsyncClient(limits=limits)

        # Initialize AsyncOpenAI with connection pooling
        self.client = AsyncOpenAI(
            api_key=openai_token,
            max_retries=openai_retries,
            timeout=httpx.Timeout(openai_timeout, connect=10.0),
            http_client=http_client,
        )
        self.voice = voice
        self.voice_instructions = voice_instructions
        self.tts_model = tts_model

    async def tts_segments_async(
        self,
        segments: list[TranscriptionSegment],
        output_dir: Path,
        progress_callback: Callable[[float], None] | None = None,
        max_concurrent: int = 100,
    ) -> set[str]:
        """Process TTS segments with high concurrency.

        Fires all TTS requests concurrently using asyncio.gather, limited by a
        semaphore to respect OpenAI rate limits. Tracks progress and calls
        progress_callback with completion percentage.

        Args:
            segments: List of transcription segments to convert to speech
            output_dir: Directory to save TTS audio files
            progress_callback: Optional callback(progress: float) where progress is 0.0-1.0
            max_concurrent: Maximum concurrent API calls (default 100)

        Returns:
            Set of output file paths for successfully generated audio files

        Performance:
            - 1600 segments with max_concurrent=100: ~16 seconds
            - 1600 segments with ThreadPoolExecutor(20): ~80 seconds
            - 5x speedup with async + high concurrency
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Semaphore limits concurrent API calls (respects OpenAI rate limits)
        sem = asyncio.Semaphore(max_concurrent)
        total_segments = len(segments)
        completed = 0

        async def process_one_with_progress(
            i: int, segment: TranscriptionSegment
        ) -> str | None:
            """Process one segment and update progress counter."""
            nonlocal completed
            async with sem:
                result = await self._process_tts_segment_async(segment, output_dir, i)
                completed += 1
                if progress_callback:
                    progress_callback(completed / total_segments)
                return result

        # Fire all tasks concurrently (asyncio handles scheduling)
        tasks = [process_one_with_progress(i, seg) for i, seg in enumerate(segments)]

        # Gather all results (failures return None via exception handling)
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Filter out None values from failures
        successful_paths = {r for r in results if r is not None}

        logger.info(
            f"TTS completed: {len(successful_paths)}/{total_segments} segments successful"
        )

        return successful_paths

    async def _process_tts_segment_async(
        self,
        segment: TranscriptionSegment,
        output_dir: Path,
        index: int,
    ) -> str | None:
        """Process a single TTS segment with retry logic.

        Checks if the output file already exists (idempotency). Uses gpt-4o-mini-tts
        if voice_instructions provided, else tts-1. Writes response bytes to file.

        Args:
            segment: Transcription segment with text to convert to speech
            output_dir: Directory to save TTS audio file
            index: Segment index for filename

        Returns:
            Output file path on success, None on exception

        Raises:
            No exceptions raised - failures are logged and return None for graceful degradation
        """
        output_path = output_dir / f"{index:03d}.en.m4a"

        # Skip if already processed (idempotency)
        if output_path.exists():
            logger.debug(f"Skipping existing segment {index}: {output_path}")
            return str(output_path)

        try:
            if self.voice_instructions:
                response = await self.client.audio.speech.create(
                    model=self.tts_model,
                    voice=self.voice,  # type: ignore[arg-type]
                    input=segment.text,
                    instructions=self.voice_instructions,
                )
            else:
                response = await self.client.audio.speech.create(
                    model=self.tts_model,
                    voice=self.voice,  # type: ignore[arg-type]
                    input=segment.text,
                )

            # Write response.content bytes to a temp file first
            temp_path = output_path.with_suffix(".tmp.m4a")
            temp_path.write_bytes(response.content)

            # Speed-adjust if the generated audio is longer than the subtitle window
            segment_duration = segment.end - segment.start
            actual_duration = _get_audio_duration(temp_path)

            if actual_duration > 0 and actual_duration > segment_duration:
                speed = min(actual_duration / segment_duration, _MAX_SPEED)
                logger.debug(
                    f"Segment {index}: speeding up {speed:.2f}x "
                    f"({actual_duration:.2f}s → {segment_duration:.2f}s)"
                )
                try:
                    _adjust_speed(temp_path, output_path, speed)
                    temp_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Segment {index}: speed adjust failed ({e}), using original")
                    temp_path.rename(output_path)
            else:
                temp_path.rename(output_path)

            logger.debug(f"TTS segment {index} completed: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"TTS failed for segment {index}: {e}")
            return None  # Graceful degradation

    async def close(self) -> None:
        """Close the httpx client connection pool.

        Must be called to cleanup httpx connections. Typically called after
        all TTS processing is complete.
        """
        await self.client.close()
        logger.debug("AsyncRedubberService closed")
