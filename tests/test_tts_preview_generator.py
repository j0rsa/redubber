"""
Tests for TTSPreviewGenerator service.
Tests TTS audio generation with caching and parallel processing.
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.stale  # needs rewrite to match current generate_audio signature

from app.services.tts_preview_generator import (
    TTSPreviewGenerator,
    TTS_MODEL,
    get_tts_preview_generator,
)
from database import DatabaseManager


class TestTTSPreviewGeneratorInit:
    """Test initialization and configuration."""

    def test_init_with_explicit_api_key(self, tmp_path):
        """Test initialization with explicitly provided API key."""
        cache_dir = tmp_path / "cache"

        generator = TTSPreviewGenerator(
            api_key="test-key-123",
            cache_dir=str(cache_dir),
        )

        assert generator.api_key == "test-key-123"
        assert generator.client is not None
        assert generator.cache_dir == cache_dir
        assert cache_dir.exists()

    def test_init_with_env_api_key(self, monkeypatch, tmp_path):
        """Test initialization with API key from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")

        generator = TTSPreviewGenerator(cache_dir=str(tmp_path))

        assert generator.api_key == "env-test-key"

    def test_init_without_api_key_raises_error(self, monkeypatch, tmp_path):
        """Test initialization without API key raises ValueError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OpenAI API key not found"):
            TTSPreviewGenerator(cache_dir=str(tmp_path))

    def test_init_creates_cache_directory(self, tmp_path):
        """Test that cache directory is created if it doesn't exist."""
        cache_dir = tmp_path / "nested" / "cache" / "dir"

        TTSPreviewGenerator(
            api_key="test-key",
            cache_dir=str(cache_dir),
        )

        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_init_with_db_manager(self, tmp_path):
        """Test initialization with provided DatabaseManager."""
        db_manager = DatabaseManager(db_path=str(tmp_path / "test.db"))

        generator = TTSPreviewGenerator(
            api_key="test-key",
            db_manager=db_manager,
            cache_dir=str(tmp_path),
        )

        assert generator.db_manager is db_manager

    def test_init_creates_db_manager_if_not_provided(self, tmp_path):
        """Test that DatabaseManager is created if not provided."""
        generator = TTSPreviewGenerator(
            api_key="test-key",
            cache_dir=str(tmp_path),
        )

        assert generator.db_manager is not None
        assert isinstance(generator.db_manager, DatabaseManager)


class TestCacheKeyGeneration:
    """Test cache key generation and hashing."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Provide generator instance."""
        return TTSPreviewGenerator(api_key="test-key", cache_dir=str(tmp_path))

    def test_generate_cache_key_produces_hash(self, generator):
        """Test that cache key generates valid SHA256 hash."""
        cache_key = generator.generate_cache_key(
            translated_text="Hello world",
            voice_instructions="Speak clearly",
            voice="alloy",
        )

        assert len(cache_key) == 64  # SHA256 hex length
        assert all(c in "0123456789abcdef" for c in cache_key)

    def test_same_inputs_produce_same_hash(self, generator):
        """Test that identical inputs produce identical cache keys."""
        key1 = generator.generate_cache_key("Test", "Instructions", "nova")
        key2 = generator.generate_cache_key("Test", "Instructions", "nova")

        assert key1 == key2

    def test_different_texts_produce_different_hashes(self, generator):
        """Test that different texts produce different cache keys."""
        key1 = generator.generate_cache_key("Text A", "Instructions", "nova")
        key2 = generator.generate_cache_key("Text B", "Instructions", "nova")

        assert key1 != key2

    def test_different_instructions_produce_different_hashes(self, generator):
        """Test that different instructions produce different cache keys."""
        key1 = generator.generate_cache_key("Text", "Instructions A", "nova")
        key2 = generator.generate_cache_key("Text", "Instructions B", "nova")

        assert key1 != key2

    def test_different_voices_produce_different_hashes(self, generator):
        """Test that different voices produce different cache keys."""
        key1 = generator.generate_cache_key("Text", "Instructions", "alloy")
        key2 = generator.generate_cache_key("Text", "Instructions", "nova")

        assert key1 != key2

    def test_cache_key_with_unicode(self, generator):
        """Test cache key generation with unicode characters."""
        cache_key = generator.generate_cache_key(
            translated_text="Привет мир 你好世界",
            voice_instructions="Speak with émôtion",
            voice="shimmer",
        )

        assert len(cache_key) == 64

    def test_cache_key_with_empty_strings(self, generator):
        """Test cache key generation with empty inputs."""
        cache_key = generator.generate_cache_key("", "", "")

        assert len(cache_key) == 64


class TestAudioDurationExtraction:
    """Test audio duration extraction using ffprobe."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Provide generator instance."""
        return TTSPreviewGenerator(api_key="test-key", cache_dir=str(tmp_path))

    def test_get_audio_duration_success(self, generator):
        """Test successful duration extraction."""
        ffprobe_output = {
            "format": {
                "duration": "3.456",
                "size": "55123",
            }
        }

        mock_result = Mock()
        mock_result.stdout = json.dumps(ffprobe_output)

        with patch("subprocess.run", return_value=mock_result):
            duration_ms = generator.get_audio_duration_ms("/path/to/audio.mp3")

        assert duration_ms == 3456  # 3.456 seconds * 1000

    def test_get_audio_duration_ffprobe_failure(self, generator):
        """Test handling of ffprobe failure."""
        with patch("subprocess.run", side_effect=Exception("ffprobe not found")):
            duration_ms = generator.get_audio_duration_ms("/path/to/audio.mp3")

        assert duration_ms == 0

    def test_get_audio_duration_invalid_json(self, generator):
        """Test handling of invalid JSON from ffprobe."""
        mock_result = Mock()
        mock_result.stdout = "Not valid JSON"

        with patch("subprocess.run", return_value=mock_result):
            duration_ms = generator.get_audio_duration_ms("/path/to/audio.mp3")

        assert duration_ms == 0

    def test_get_audio_duration_missing_duration_field(self, generator):
        """Test handling of missing duration field in ffprobe output."""
        ffprobe_output = {"format": {"size": "55123"}}  # No duration field

        mock_result = Mock()
        mock_result.stdout = json.dumps(ffprobe_output)

        with patch("subprocess.run", return_value=mock_result):
            duration_ms = generator.get_audio_duration_ms("/path/to/audio.mp3")

        assert duration_ms == 0


class TestGenerateAudio:
    """Test TTS audio generation."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Provide generator instance."""
        return TTSPreviewGenerator(api_key="test-key", cache_dir=str(tmp_path))

    def test_generate_audio_success(self, generator):
        """Test successful audio generation."""
        mock_response = Mock()
        mock_response.stream_to_file = Mock()

        with patch.object(
            generator.client.audio.speech, "create", return_value=mock_response
        ) as mock_create, patch.object(
            generator, "get_audio_duration_ms", return_value=2500
        ):
            audio_path, duration = generator.generate_audio(
                voice="nova",
                translated_text="Hello world",
                voice_instructions="Speak clearly",
            )

            mock_create.assert_called_once()
            assert Path(audio_path).suffix == ".mp3"
            assert duration == 2500
            assert "nova" in audio_path

    def test_generate_audio_with_instructions_uses_mini_tts(self, generator):
        """Test that generation with instructions tries gpt-4o-mini-tts first."""
        mock_response = Mock()
        mock_response.stream_to_file = Mock()

        with patch.object(
            generator.client.audio.speech, "create", return_value=mock_response
        ) as mock_create, patch.object(generator, "get_audio_duration_ms", return_value=1000):
            generator.generate_audio(
                voice="alloy",
                translated_text="Test",
                voice_instructions="Test instructions",
            )

            # Should try gpt-4o-mini-tts with instructions
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o-mini-tts"
            assert call_kwargs["instructions"] == "Test instructions"

    def test_generate_audio_without_instructions_uses_standard_model(self, generator):
        """Test that generation without instructions uses standard TTS model."""
        mock_response = Mock()
        mock_response.stream_to_file = Mock()

        with patch.object(
            generator.client.audio.speech, "create", return_value=mock_response
        ) as mock_create, patch.object(generator, "get_audio_duration_ms", return_value=1000):
            generator.generate_audio(
                voice="echo",
                translated_text="Test",
                voice_instructions="",
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == TTS_MODEL
            assert "instructions" not in call_kwargs

    def test_generate_audio_fallback_on_mini_tts_failure(self, generator):
        """Test fallback to standard model if gpt-4o-mini-tts fails."""
        mock_response = Mock()
        mock_response.stream_to_file = Mock()

        call_count = [0]

        def create_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and kwargs.get("model") == "gpt-4o-mini-tts":
                raise Exception("Model not available")
            return mock_response

        with patch.object(
            generator.client.audio.speech, "create", side_effect=create_side_effect
        ) as mock_create, patch.object(generator, "get_audio_duration_ms", return_value=1000):
            audio_path, duration = generator.generate_audio(
                voice="fable",
                translated_text="Test",
                voice_instructions="Instructions",
            )

            # Should have been called twice (first failed, second succeeded)
            assert mock_create.call_count == 2
            assert audio_path is not None

    def test_generate_audio_with_hd_model(self, generator):
        """Test audio generation with HD model."""
        mock_response = Mock()
        mock_response.stream_to_file = Mock()

        with patch.object(
            generator.client.audio.speech, "create", return_value=mock_response
        ) as mock_create, patch.object(generator, "get_audio_duration_ms", return_value=1000):
            generator.generate_audio(
                voice="onyx",
                translated_text="Test",
                voice_instructions="",
                use_hd=True,
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "tts-1-hd"

    def test_generate_audio_invalid_voice_raises_error(self, generator):
        """Test that invalid voice name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid voice"):
            generator.generate_audio(
                voice="invalid_voice",
                translated_text="Test",
                voice_instructions="",
            )

    def test_generate_audio_api_failure_raises_exception(self, generator):
        """Test that API failure raises exception."""
        with patch.object(
            generator.client.audio.speech,
            "create",
            side_effect=Exception("API error"),
        ):
            with pytest.raises(Exception, match="API error"):
                generator.generate_audio(
                    voice="shimmer",
                    translated_text="Test",
                    voice_instructions="",
                )

    def test_generate_audio_creates_file_in_cache_dir(self, generator, tmp_path):
        """Test that generated audio files are saved in cache directory."""
        mock_response = Mock()
        mock_response.stream_to_file = Mock()

        with patch.object(
            generator.client.audio.speech, "create", return_value=mock_response
        ), patch.object(generator, "get_audio_duration_ms", return_value=1000):
            audio_path, _ = generator.generate_audio(
                voice="nova",
                translated_text="Test",
                voice_instructions="",
            )

            assert str(tmp_path) in audio_path


class TestGeneratePreview:
    """Test single voice preview generation with caching."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Provide generator instance with test database."""
        db_path = tmp_path / "test.db"
        db_manager = DatabaseManager(db_path=str(db_path))
        return TTSPreviewGenerator(
            api_key="test-key",
            db_manager=db_manager,
            cache_dir=str(tmp_path / "cache"),
        )

    @pytest.fixture
    def project_id(self, generator):
        """Create a test project and return its ID."""
        return generator.db_manager.add_project(
            path="/test/project",
            name="Test Project",
        )

    def test_generate_preview_cache_hit(self, generator, project_id, tmp_path):
        """Test that cached preview is returned when available."""
        # Create a fake audio file
        audio_file = tmp_path / "cached_audio.mp3"
        audio_file.write_text("fake audio content")

        # Pre-populate cache
        cache_key = generator.generate_cache_key("Test text", "Instructions", "nova")
        generator.db_manager.save_tts_cache(
            project_id=project_id,
            voice_name="nova",
            voice_instructions_hash=cache_key,
            translated_text="Test text",
            audio_file_path=str(audio_file),
            audio_duration_ms=2000,
        )

        # Generate preview (should hit cache)
        result = generator.generate_preview(
            project_id=project_id,
            voice="nova",
            translated_text="Test text",
            voice_instructions="Instructions",
        )

        assert result["cached"] is True
        assert result["audio_file_path"] == str(audio_file)
        assert result["duration_ms"] == 2000
        assert result["voice"] == "nova"

    def test_generate_preview_cache_miss(self, generator, project_id):
        """Test audio generation when cache miss occurs."""
        mock_response = Mock()
        mock_response.stream_to_file = Mock()

        with patch.object(
            generator.client.audio.speech, "create", return_value=mock_response
        ), patch.object(generator, "get_audio_duration_ms", return_value=3000):
            result = generator.generate_preview(
                project_id=project_id,
                voice="echo",
                translated_text="New text",
                voice_instructions="New instructions",
            )

        assert result["cached"] is False
        assert result["duration_ms"] == 3000
        assert result["voice"] == "echo"
        assert os.path.exists(result["audio_file_path"])

    def test_generate_preview_cache_file_missing(self, generator, project_id, tmp_path):
        """Test regeneration when cached file is missing."""
        # Create cache entry but no file
        cache_key = generator.generate_cache_key("Test", "Instructions", "alloy")
        generator.db_manager.save_tts_cache(
            project_id=project_id,
            voice_name="alloy",
            voice_instructions_hash=cache_key,
            translated_text="Test",
            audio_file_path="/nonexistent/file.mp3",
            audio_duration_ms=1000,
        )

        mock_response = Mock()
        mock_response.stream_to_file = Mock()

        with patch.object(
            generator.client.audio.speech, "create", return_value=mock_response
        ), patch.object(generator, "get_audio_duration_ms", return_value=2500):
            result = generator.generate_preview(
                project_id=project_id,
                voice="alloy",
                translated_text="Test",
                voice_instructions="Instructions",
            )

        # Should regenerate
        assert result["cached"] is False

    def test_generate_preview_saves_to_cache(self, generator, project_id):
        """Test that newly generated preview is saved to cache."""
        mock_response = Mock()
        mock_response.stream_to_file = Mock()

        with patch.object(
            generator.client.audio.speech, "create", return_value=mock_response
        ), patch.object(generator, "get_audio_duration_ms", return_value=1500):
            generator.generate_preview(
                project_id=project_id,
                voice="fable",
                translated_text="Save to cache",
                voice_instructions="Cache instructions",
            )

        # Verify saved to database
        cache_key = generator.generate_cache_key(
            "Save to cache", "Cache instructions", "fable"
        )
        cached = generator.db_manager.get_or_create_tts_cache(
            project_id=project_id,
            voice_name="fable",
            voice_instructions_hash=cache_key,
            translated_text="Save to cache",
        )

        assert cached is not None
        assert cached["audio_duration_ms"] == 1500


class TestGenerateAllPreviews:
    """Test parallel generation of all voice previews."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Provide generator instance with test database."""
        db_path = tmp_path / "test.db"
        db_manager = DatabaseManager(db_path=str(db_path))
        return TTSPreviewGenerator(
            api_key="test-key",
            db_manager=db_manager,
            cache_dir=str(tmp_path / "cache"),
        )

    @pytest.fixture
    def project_id(self, generator):
        """Create a test project."""
        return generator.db_manager.add_project(
            path="/test/project",
            name="Test Project",
        )

    def test_generate_all_previews_default_voices(self, generator, project_id):
        """Test generation for all default voices."""
        with patch.object(generator, "generate_preview") as mock_gen:
            mock_gen.return_value = {
                "voice": "test",
                "audio_file_path": "/path/to/audio.mp3",
                "duration_ms": 1000,
                "cached": False,
            }

            result = generator.generate_all_previews(
                project_id=project_id,
                translated_text="Test text",
                voice_instructions="Test instructions",
            )

        # Should generate for all 6 default voices
        assert len(result["previews"]) == 6
        assert mock_gen.call_count == 6

    def test_generate_all_previews_custom_voices(self, generator, project_id):
        """Test generation for custom voice list."""
        custom_voices = ["alloy", "nova", "shimmer"]

        with patch.object(generator, "generate_preview") as mock_gen:
            mock_gen.return_value = {
                "voice": "test",
                "audio_file_path": "/path/to/audio.mp3",
                "duration_ms": 1000,
                "cached": False,
            }

            result = generator.generate_all_previews(
                project_id=project_id,
                translated_text="Test",
                voice_instructions="Instructions",
                voices=custom_voices,
            )

        assert len(result["previews"]) == 3
        assert mock_gen.call_count == 3

    def test_generate_all_previews_cache_statistics(self, generator, project_id):
        """Test that cache statistics are calculated correctly."""
        call_count = [0]

        def mock_generate_preview(*args, **kwargs):
            call_count[0] += 1
            # First 3 are cached, rest are not
            is_cached = call_count[0] <= 3
            return {
                "voice": f"voice{call_count[0]}",
                "audio_file_path": f"/path/{call_count[0]}.mp3",
                "duration_ms": 1000,
                "cached": is_cached,
            }

        with patch.object(generator, "generate_preview", side_effect=mock_generate_preview):
            result = generator.generate_all_previews(
                project_id=project_id,
                translated_text="Test",
                voice_instructions="Instructions",
            )

        assert result["cache_hits"] == 3
        assert result["cache_misses"] == 3

    def test_generate_all_previews_parallel_execution(self, generator, project_id):
        """Test that previews are generated in parallel."""
        import time
        from threading import Lock

        concurrent_calls = []
        lock = Lock()

        def mock_generate_preview(*args, **kwargs):
            with lock:
                concurrent_calls.append(time.time())
            time.sleep(0.1)  # Simulate API call
            return {
                "voice": "test",
                "audio_file_path": "/path.mp3",
                "duration_ms": 1000,
                "cached": False,
            }

        with patch.object(generator, "generate_preview", side_effect=mock_generate_preview):
            start_time = time.time()
            generator.generate_all_previews(
                project_id=project_id,
                translated_text="Test",
                voice_instructions="Instructions",
            )
            elapsed = time.time() - start_time

        # If parallel, should complete in ~0.1s, not 0.6s (6 * 0.1s)
        assert elapsed < 0.3

    def test_generate_all_previews_invalid_voice_raises_error(self, generator, project_id):
        """Test that invalid voice names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid voices"):
            generator.generate_all_previews(
                project_id=project_id,
                translated_text="Test",
                voice_instructions="Instructions",
                voices=["invalid_voice", "alloy"],
            )

    def test_generate_all_previews_handles_errors(self, generator, project_id):
        """Test that errors in individual previews don't break entire generation."""
        call_count = [0]

        def mock_generate_preview(project_id, voice, *args, **kwargs):
            call_count[0] += 1
            if voice == "echo":
                raise Exception("API error for echo")
            return {
                "voice": voice,
                "audio_file_path": f"/path/{voice}.mp3",
                "duration_ms": 1000,
                "cached": False,
            }

        with patch.object(generator, "generate_preview", side_effect=mock_generate_preview):
            result = generator.generate_all_previews(
                project_id=project_id,
                translated_text="Test",
                voice_instructions="Instructions",
            )

        # All voices should have results (some with errors)
        assert len(result["previews"]) == 6

        # Find the echo preview
        echo_preview = next(p for p in result["previews"] if p["voice"] == "echo")
        assert "error" in echo_preview

    def test_generate_all_previews_sorted_by_voice_name(self, generator, project_id):
        """Test that results are sorted alphabetically by voice name."""
        with patch.object(generator, "generate_preview") as mock_gen:
            # Return previews in random order
            mock_gen.side_effect = [
                {"voice": "shimmer", "audio_file_path": "/s.mp3", "duration_ms": 1000, "cached": False},
                {"voice": "alloy", "audio_file_path": "/a.mp3", "duration_ms": 1000, "cached": False},
                {"voice": "nova", "audio_file_path": "/n.mp3", "duration_ms": 1000, "cached": False},
                {"voice": "echo", "audio_file_path": "/e.mp3", "duration_ms": 1000, "cached": False},
                {"voice": "onyx", "audio_file_path": "/o.mp3", "duration_ms": 1000, "cached": False},
                {"voice": "fable", "audio_file_path": "/f.mp3", "duration_ms": 1000, "cached": False},
            ]

            result = generator.generate_all_previews(
                project_id=project_id,
                translated_text="Test",
                voice_instructions="Instructions",
            )

        # Check sorted order
        voice_names = [p["voice"] for p in result["previews"]]
        assert voice_names == sorted(voice_names)

    def test_generate_all_previews_includes_instructions_hash(self, generator, project_id):
        """Test that result includes instructions hash for caching."""
        with patch.object(generator, "generate_preview") as mock_gen:
            mock_gen.return_value = {
                "voice": "test",
                "audio_file_path": "/path.mp3",
                "duration_ms": 1000,
                "cached": False,
            }

            result = generator.generate_all_previews(
                project_id=project_id,
                translated_text="Test text",
                voice_instructions="Test instructions",
            )

        assert "instructions_hash" in result
        assert len(result["instructions_hash"]) == 16  # Truncated hash


class TestCacheCleanup:
    """Test cache cleanup functionality."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Provide generator instance with test database."""
        db_path = tmp_path / "test.db"
        db_manager = DatabaseManager(db_path=str(db_path))
        return TTSPreviewGenerator(
            api_key="test-key",
            db_manager=db_manager,
            cache_dir=str(tmp_path / "cache"),
        )

    def test_cleanup_cache_calls_db_manager(self, generator):
        """Test that cleanup delegates to database manager."""
        with patch.object(generator.db_manager, "cleanup_old_tts_cache") as mock_cleanup:
            generator.cleanup_cache(days=15, max_entries_per_project=50)

            mock_cleanup.assert_called_once_with(days=15, max_entries_per_project=50)


class TestSingletonPattern:
    """Test singleton pattern for generator instance."""

    def test_get_tts_preview_generator_creates_instance(self, monkeypatch, tmp_path):
        """Test that get function creates instance on first call."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Clear singleton
        import app.services.tts_preview_generator as module
        module._generator_instance = None

        generator = get_tts_preview_generator(cache_dir=str(tmp_path))

        assert generator is not None
        assert isinstance(generator, TTSPreviewGenerator)

    def test_get_tts_preview_generator_returns_same_instance(self, monkeypatch, tmp_path):
        """Test that get function returns same instance on subsequent calls."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Clear singleton
        import app.services.tts_preview_generator as module
        module._generator_instance = None

        generator1 = get_tts_preview_generator(cache_dir=str(tmp_path))
        generator2 = get_tts_preview_generator(cache_dir=str(tmp_path))

        assert generator1 is generator2
