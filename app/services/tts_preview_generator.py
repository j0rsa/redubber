"""
TTS Preview Generation Service for Voice Refinement System.
Generates audio previews for all OpenAI voices with caching support.
"""

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

# Import database manager from root (since it's not in app/ yet)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from database import DatabaseManager

logger = logging.getLogger(__name__)

# Available OpenAI TTS voices
AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# TTS model to use
TTS_MODEL = "tts-1"
TTS_MODEL_HD = "tts-1-hd"


class TTSPreviewGenerator:
    """
    Generates TTS audio previews for voice refinement.

    Features:
    - Generates audio for all 6 OpenAI voices
    - Hash-based caching to avoid redundant API calls
    - Parallel generation for performance
    - Duration extraction from audio files
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        db_manager: Optional[DatabaseManager] = None,
        cache_dir: str = "tts_cache",
    ):
        """
        Initialize TTS Preview Generator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            db_manager: Database manager instance (creates new if None)
            cache_dir: Directory to store cached audio files
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

        self.client = OpenAI(api_key=self.api_key)
        self.db_manager = db_manager or DatabaseManager()

        # Ensure cache directory exists
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_cache_key(
        self, translated_text: str, voice_instructions: str, voice: str
    ) -> str:
        """
        Generate SHA256 hash for cache key.

        Args:
            translated_text: Text to be converted to speech
            voice_instructions: TTS voice instructions
            voice: Voice name

        Returns:
            SHA256 hash as hex string
        """
        content = f"{translated_text}|{voice_instructions}|{voice}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_audio_duration_ms(self, audio_file_path: str) -> int:
        """
        Extract audio duration in milliseconds using ffprobe.

        Args:
            audio_file_path: Path to audio file

        Returns:
            Duration in milliseconds
        """
        import subprocess
        import json

        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                audio_file_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
            duration_seconds = float(probe_data["format"]["duration"])
            return int(duration_seconds * 1000)
        except Exception as e:
            logger.error(f"Failed to extract duration from {audio_file_path}: {e}")
            return 0

    def generate_audio(
        self,
        text: str,
        voice: str,
        instructions: str,
        output_path: str,
        use_hd: bool = False,
        model: Optional[str] = None,
        # Legacy param names kept for internal callers
        translated_text: Optional[str] = None,
        voice_instructions: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Generate TTS audio using OpenAI API and save to output_path.

        Args:
            text: Text to convert to speech
            voice: Voice name (alloy, echo, fable, onyx, nova, shimmer)
            instructions: Voice style/tone instructions
            output_path: Absolute path where the .mp3 should be written
            use_hd: Use tts-1-hd model (higher quality, slower)
            model: Override TTS model name; when provided, ``use_hd`` is
                ignored.  Defaults to ``TTS_MODEL`` (or ``TTS_MODEL_HD``
                when *use_hd* is ``True``).

        Returns:
            Tuple of (audio_file_path, duration_ms)

        Raises:
            Exception: If TTS generation fails
        """
        # Normalise legacy callers
        translated_text = translated_text or text
        voice_instructions = voice_instructions or instructions

        if voice not in AVAILABLE_VOICES:
            raise ValueError(
                f"Invalid voice '{voice}'. Must be one of: {AVAILABLE_VOICES}"
            )

        audio_file_path = Path(output_path)
        audio_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate audio with OpenAI — explicit model takes precedence
        if model:
            pass  # use caller-supplied model as-is
        elif use_hd:
            model = TTS_MODEL_HD
        else:
            model = TTS_MODEL

        # Instructions require gpt-4o-mini-tts; upgrade automatically when needed.
        INSTRUCTIONS_MODEL = "gpt-4o-mini-tts"
        if voice_instructions and model not in (INSTRUCTIONS_MODEL,):
            logger.info(
                f"Upgrading TTS model from '{model}' to '{INSTRUCTIONS_MODEL}' "
                "because voice instructions are provided."
            )
            model = INSTRUCTIONS_MODEL

        try:
            if voice_instructions:
                response = self.client.audio.speech.create(
                    model=model,
                    voice=voice,
                    input=translated_text,
                    instructions=voice_instructions,
                    response_format="mp3",
                )
            else:
                response = self.client.audio.speech.create(
                    model=model,
                    voice=voice,
                    input=translated_text,
                    response_format="mp3",
                )

            # Save to file
            response.stream_to_file(str(audio_file_path))

            # Extract duration
            duration_ms = self.get_audio_duration_ms(str(audio_file_path))

            logger.info(
                f"Generated TTS audio for voice '{voice}': {audio_file_path} ({duration_ms}ms)"
            )

            return str(audio_file_path), duration_ms

        except Exception as e:
            logger.error(f"Failed to generate TTS for voice '{voice}': {e}")
            raise

    def generate_preview(
        self,
        project_id: int,
        voice: str,
        translated_text: str,
        voice_instructions: str,
        use_hd: bool = False,
    ) -> Dict:
        """
        Generate or retrieve cached TTS preview for a single voice.

        Args:
            project_id: Project ID
            voice: Voice name
            translated_text: Text to convert to speech
            voice_instructions: Voice style instructions
            use_hd: Use HD model

        Returns:
            Dictionary with audio_file_path, duration_ms, and cached flag
        """
        # Generate cache key
        cache_key = self.generate_cache_key(translated_text, voice_instructions, voice)

        # Check database cache (scoped to this project)
        cached_entry = self.db_manager.get_tts_cache(
            project_id=project_id,
            voice_name=voice,
            voice_instructions_hash=cache_key,
        )

        if cached_entry:
            audio_file_path = cached_entry["audio_file_path"]
            if os.path.exists(audio_file_path):
                logger.info(f"Cache hit for voice '{voice}': {audio_file_path}")
                return {
                    "voice": voice,
                    "audio_file_path": audio_file_path,
                    "duration_ms": cached_entry["audio_duration_ms"],
                    "cached": True,
                }
            else:
                logger.warning(f"Cache entry exists but file missing: {audio_file_path}")

        # Cache miss — derive output path from service cache_dir (legacy path for generate_preview)
        output_path = str(self.cache_dir / f"{cache_key[:16]}_{voice}.mp3")

        logger.info(f"Cache miss for voice '{voice}', generating new preview")
        audio_file_path, duration_ms = self.generate_audio(
            text=translated_text,
            voice=voice,
            instructions=voice_instructions,
            output_path=output_path,
            use_hd=use_hd,
        )

        # Save to database cache
        self.db_manager.save_tts_cache(
            project_id=project_id,
            voice_name=voice,
            voice_instructions_hash=cache_key,
            translated_text=translated_text,
            audio_file_path=audio_file_path,
            audio_duration_ms=duration_ms,
            tts_model="gpt-4o-mini-tts" if voice_instructions else TTS_MODEL,
        )

        return {
            "voice": voice,
            "audio_file_path": audio_file_path,
            "duration_ms": duration_ms,
            "cached": False,
        }

    def generate_all_previews(
        self,
        project_id: int,
        translated_text: str,
        voice_instructions: str,
        voices: Optional[List[str]] = None,
        use_hd: bool = False,
        max_workers: int = 6,
    ) -> Dict:
        """
        Generate TTS previews for all voices in parallel.

        Args:
            project_id: Project ID
            translated_text: Text to convert to speech
            voice_instructions: Voice style instructions
            voices: List of voices to generate (defaults to all 6)
            use_hd: Use HD model
            max_workers: Maximum parallel workers

        Returns:
            Dictionary with previews list, cache statistics, and hash
        """
        voices = voices or AVAILABLE_VOICES

        # Validate voices
        invalid_voices = [v for v in voices if v not in AVAILABLE_VOICES]
        if invalid_voices:
            raise ValueError(
                f"Invalid voices: {invalid_voices}. Must be from: {AVAILABLE_VOICES}"
            )

        logger.info(
            f"Generating previews for {len(voices)} voices with {max_workers} workers"
        )

        previews = []
        cache_hits = 0
        cache_misses = 0

        # Generate in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_voice = {
                executor.submit(
                    self.generate_preview,
                    project_id,
                    voice,
                    translated_text,
                    voice_instructions,
                    use_hd,
                ): voice
                for voice in voices
            }

            # Collect results as they complete
            for future in as_completed(future_to_voice):
                voice = future_to_voice[future]
                try:
                    result = future.result()
                    previews.append(result)

                    if result["cached"]:
                        cache_hits += 1
                    else:
                        cache_misses += 1

                except Exception as e:
                    logger.error(f"Failed to generate preview for voice '{voice}': {e}")
                    # Include error in results
                    previews.append(
                        {
                            "voice": voice,
                            "audio_file_path": None,
                            "duration_ms": 0,
                            "cached": False,
                            "error": str(e),
                        }
                    )
                    cache_misses += 1

        # Sort previews by voice name for consistent ordering
        previews.sort(key=lambda x: x["voice"])

        # Generate instruction hash for API response
        instructions_hash = self.generate_cache_key(
            translated_text, voice_instructions, ""
        )[:16]

        logger.info(
            f"Generated {len(previews)} previews: {cache_hits} cache hits, {cache_misses} cache misses"
        )

        return {
            "previews": previews,
            "instructions_hash": instructions_hash,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        }

    def cleanup_cache(self, days: int = 30, max_entries_per_project: int = 100):
        """
        Clean up old cache entries from database and filesystem.

        Args:
            days: Delete entries older than this many days
            max_entries_per_project: Keep only this many recent entries per project
        """
        logger.info(f"Cleaning up TTS cache: {days} days, max {max_entries_per_project} per project")

        # Clean up database entries
        self.db_manager.cleanup_old_tts_cache(
            days=days, max_entries_per_project=max_entries_per_project
        )

        # TODO: Clean up orphaned audio files from filesystem
        # This would require comparing database entries with files in cache_dir
        logger.info("Cache cleanup completed")


# Singleton instance
_generator_instance: Optional[TTSPreviewGenerator] = None


def get_tts_preview_generator(
    api_key: Optional[str] = None,
    db_manager: Optional[DatabaseManager] = None,
    cache_dir: str = "tts_cache",
) -> TTSPreviewGenerator:
    """
    Get or create the singleton TTS preview generator.

    Args:
        api_key: OpenAI API key
        db_manager: Database manager instance
        cache_dir: Cache directory path

    Returns:
        TTSPreviewGenerator instance
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = TTSPreviewGenerator(
            api_key=api_key, db_manager=db_manager, cache_dir=cache_dir
        )
    return _generator_instance
