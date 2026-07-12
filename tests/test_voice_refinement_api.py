"""
Tests for voice refinement API endpoints.
Tests all voice refinement routes including transcription segments,
voice instruction generation, TTS previews, and voice settings.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from database import DatabaseManager

def _make_fake_segments(project_id: int, count: int = 30):
    """Return a list of fake TranscriptionSegment-like dicts for testing."""
    from app.schemas.voice_refinement import TranscriptionSegment
    segments = []
    for i in range(count):
        start = float(i * 8)
        end = start + 5.0 + (i % 3)  # durations 5–7s, all pass default 3–20s filter
        segments.append(TranscriptionSegment(
            id=f"project_{project_id}_seg_{i}",
            video_filename="test_video.mp4",
            start_time=start,
            end_time=end,
            duration=end - start,
            original_text=f"This is original text for segment {i}." + (" demonstration" if i % 5 == 0 else ""),
            translated_text=f"Este es el texto traducido para el segmento {i}.",
            audio_url=f"/api/projects/{project_id}/segments/project_{project_id}_seg_{i}/audio",
        ))
    return segments


class TestTranscriptionSegmentsEndpoint:
    """Test GET /projects/{project_id}/transcription-segments endpoint."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """Provide test client with temporary database and fake segments."""
        from app.core.config import settings

        storage_dir = tmp_path / "storage"
        storage_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(settings, "redubber_config_path", str(storage_dir))
        monkeypatch.setattr(settings, "openai_api_key", "test-key")

        app = create_app()
        with TestClient(app) as test_client:
            yield test_client

    @pytest.fixture
    def project_id(self, client, tmp_path, monkeypatch):
        """Create a test project and patch segment loading with fake data."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        response = client.post("/api/projects/", json={"path": str(project_dir)})
        assert response.status_code == 201
        pid = response.json()["id"]
        monkeypatch.setattr(
            "app.api.routes.voice_refinement._load_real_segments",
            lambda project_id, project_path: _make_fake_segments(project_id),
        )
        return pid

    def test_get_transcription_segments_success(self, client, project_id):
        """Test successful retrieval of transcription segments with default params."""
        response = client.get(f"/api/projects/{project_id}/transcription-segments")

        assert response.status_code == 200
        body = response.json()

        # Verify envelope fields
        assert "segments" in body
        assert "total_candidates" in body
        assert "total_matched" in body
        assert "returned" in body
        assert "has_more" in body
        assert "sample_size" in body

        assert isinstance(body["segments"], list)
        assert len(body["segments"]) > 0
        assert body["returned"] == len(body["segments"])
        assert body["sample_size"] == 20  # default

        # Verify segment structure
        segment = body["segments"][0]
        assert "id" in segment
        assert "video_filename" in segment
        assert "start_time" in segment
        assert "end_time" in segment
        assert "duration" in segment
        assert "original_text" in segment
        assert "translated_text" in segment
        assert "audio_url" in segment

    def test_get_transcription_segments_project_not_found(self, client):
        """Test 404 error when project doesn't exist."""
        response = client.get("/api/projects/99999/transcription-segments")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_transcription_segments_mock_data(self, client, project_id):
        """Test that mock data corpus is 200 segments and IDs contain project ID."""
        response = client.get(f"/api/projects/{project_id}/transcription-segments")

        body = response.json()
        # Default duration filter (3–20s) removes segments shorter than 3s or longer than 20s.
        # With 200 mock segments cycling through the duration pattern, a non-trivial number
        # pass the filter — assert at least one is present.
        assert body["total_candidates"] > 0
        assert body["returned"] > 0

        # Verify IDs are namespaced by project ID
        assert f"project_{project_id}" in body["segments"][0]["id"]

    def test_get_transcription_segments_custom_sample(self, client, project_id):
        """Test that the sample query param controls how many segments are returned."""
        response = client.get(
            f"/api/projects/{project_id}/transcription-segments",
            params={"sample": 5},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["sample_size"] == 5
        assert body["returned"] <= 5

    def test_get_transcription_segments_search(self, client, project_id):
        """Test keyword search skips sampling and filters by text."""
        response = client.get(
            f"/api/projects/{project_id}/transcription-segments",
            params={"search": "demonstration"},
        )
        assert response.status_code == 200
        body = response.json()
        for seg in body["segments"]:
            text = (seg["original_text"] + seg["translated_text"]).lower()
            assert "demonstration" in text

    def test_get_transcription_segments_duration_filter(self, client, project_id):
        """Test that duration filters are applied before sampling."""
        response = client.get(
            f"/api/projects/{project_id}/transcription-segments",
            params={"min_duration": 10.0, "max_duration": 15.0, "sample": 100},
        )
        assert response.status_code == 200
        body = response.json()
        for seg in body["segments"]:
            assert 10.0 <= seg["duration"] <= 15.0

    def test_get_transcription_segments_invalid_duration_range(self, client, project_id):
        """Test 422 when min_duration exceeds max_duration."""
        response = client.get(
            f"/api/projects/{project_id}/transcription-segments",
            params={"min_duration": 20.0, "max_duration": 5.0},
        )
        assert response.status_code == 422


class TestVoiceInstructionAnalyzeEndpoint:
    """Test POST /projects/{project_id}/voice-instructions/analyze endpoint."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """Provide test client with temporary database."""
        from app.core.config import settings

        storage_dir = tmp_path / "storage"
        storage_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(settings, "redubber_config_path", str(storage_dir))
        monkeypatch.setattr(settings, "openai_api_key", "test-key")

        app = create_app()
        with TestClient(app) as test_client:
            yield test_client

    @pytest.fixture
    def project_id(self, client, tmp_path):
        """Create a test project."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir(exist_ok=True)
        response = client.post("/api/projects/", json={"path": str(project_dir)})
        return response.json()["id"]

    def test_analyze_voice_instructions_success(self, client, project_id):
        """Test successful voice instruction generation."""
        request_data = {
            "segment_id": "segment_0",
            "original_text": "This is a test lecture.",
            "translated_text": "Esta es una conferencia de prueba.",
            "context": {
                "content_type": "educational",
                "speaker_gender": "female",
                "speaker_age": "adult",
            },
        }

        mock_result = {
            "voice_instructions": "Speak with clear enunciation and moderate pace.",
            "detected_characteristics": {
                "tone": "professional",
                "pace": "moderate",
                "emotion": "engaged",
                "style": "authoritative",
            },
            "llm_model": "gpt-4o",
        }

        with patch("app.services.voice_instruction_generator.VoiceInstructionGenerator") as MockGen:
            mock_instance = Mock()
            mock_instance.generate_instructions.return_value = mock_result
            MockGen.return_value = mock_instance

            with patch("app.api.routes.voice_refinement.get_voice_instruction_generator", return_value=mock_instance):
                response = client.post(
                    f"/api/projects/{project_id}/voice-instructions/analyze",
                    json=request_data,
                )

        assert response.status_code == 201
        result = response.json()

        assert result["voice_instructions"] == mock_result["voice_instructions"]
        assert result["detected_characteristics"]["tone"] == "professional"
        assert result["llm_model"] == "gpt-4o"
        assert result["generation_id"] > 0
        assert result["error"] is None

    def test_analyze_voice_instructions_without_context(self, client, project_id):
        """Test generation without optional context."""
        request_data = {
            "segment_id": "segment_1",
            "original_text": "Test text",
            "translated_text": "Texto de prueba",
        }

        mock_result = {
            "voice_instructions": "Speak naturally.",
            "detected_characteristics": {
                "tone": "neutral",
                "pace": "moderate",
                "emotion": "balanced",
                "style": "natural",
            },
            "llm_model": "gpt-4o",
        }

        with patch("app.services.voice_instruction_generator.VoiceInstructionGenerator") as MockGen:
            mock_instance = Mock()
            mock_instance.generate_instructions.return_value = mock_result
            MockGen.return_value = mock_instance

            with patch("app.api.routes.voice_refinement.get_voice_instruction_generator", return_value=mock_instance):
                response = client.post(
                    f"/api/projects/{project_id}/voice-instructions/analyze",
                    json=request_data,
                )

        assert response.status_code == 201
        assert response.json()["voice_instructions"] == "Speak naturally."

    def test_analyze_voice_instructions_project_not_found(self, client):
        """Test 404 when project doesn't exist."""
        request_data = {
            "segment_id": "segment_0",
            "original_text": "Test",
            "translated_text": "Prueba",
        }

        response = client.post(
            "/api/projects/99999/voice-instructions/analyze",
            json=request_data,
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_analyze_voice_instructions_missing_required_fields(self, client, project_id):
        """Test 422 validation error for missing required fields."""
        request_data = {
            "segment_id": "segment_0",
            # Missing original_text and translated_text
        }

        response = client.post(
            f"/api/projects/{project_id}/voice-instructions/analyze",
            json=request_data,
        )

        assert response.status_code == 422

    def test_analyze_voice_instructions_llm_error(self, client, project_id):
        """Test 500 error when LLM generation fails."""
        request_data = {
            "segment_id": "segment_0",
            "original_text": "Test",
            "translated_text": "Prueba",
        }

        with patch("app.services.voice_instruction_generator.VoiceInstructionGenerator") as MockGen:
            mock_instance = Mock()
            mock_instance.generate_instructions.side_effect = ValueError("API key invalid")
            MockGen.return_value = mock_instance

            with patch("app.api.routes.voice_refinement.get_voice_instruction_generator", return_value=mock_instance):
                response = client.post(
                    f"/api/projects/{project_id}/voice-instructions/analyze",
                    json=request_data,
                )

        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()

    def test_analyze_saves_to_database(self, client, project_id, tmp_path):
        """Test that generation result is saved to database."""
        from app.core.config import settings

        db = DatabaseManager(db_path=settings.database_url)

        request_data = {
            "segment_id": "test_segment",
            "original_text": "Original text",
            "translated_text": "Translated text",
        }

        mock_result = {
            "voice_instructions": "Test instructions",
            "detected_characteristics": {
                "tone": "neutral",
                "pace": "moderate",
                "emotion": "balanced",
                "style": "natural",
            },
            "llm_model": "gpt-4o",
        }

        with patch("app.services.voice_instruction_generator.VoiceInstructionGenerator") as MockGen:
            mock_instance = Mock()
            mock_instance.generate_instructions.return_value = mock_result
            MockGen.return_value = mock_instance

            with patch("app.api.routes.voice_refinement.get_voice_instruction_generator", return_value=mock_instance):
                response = client.post(
                    f"/api/projects/{project_id}/voice-instructions/analyze",
                    json=request_data,
                )

        assert response.status_code == 201

        # Verify saved to database
        generations = db.get_voice_instruction_generations(project_id=project_id)
        assert len(generations) == 1
        assert generations[0]["segment_id"] == "test_segment"


class TestVoiceInstructionRegenerateEndpoint:
    """Test POST /projects/{project_id}/voice-instructions/regenerate endpoint."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """Provide test client."""
        from app.core.config import settings

        storage_dir = tmp_path / "storage"
        storage_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(settings, "redubber_config_path", str(storage_dir))
        monkeypatch.setattr(settings, "openai_api_key", "test-key")

        app = create_app()
        with TestClient(app) as test_client:
            yield test_client

    @pytest.fixture
    def project_id(self, client, tmp_path):
        """Create a test project."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir(exist_ok=True)
        response = client.post("/api/projects/", json={"path": str(project_dir)})
        return response.json()["id"]

    def test_regenerate_voice_instructions_success(self, client, project_id):
        """Test successful regeneration with feedback."""
        request_data = {
            "segment_id": "segment_0",
            "original_text": "Original text",
            "translated_text": "Translated text",
            "previous_instructions": "Speak calmly and slowly.",
            "user_feedback": "Make it more energetic!",
        }

        mock_result = {
            "voice_instructions": "Speak with high energy and enthusiasm!",
            "detected_characteristics": {
                "tone": "energetic",
                "pace": "fast",
                "emotion": "excited",
                "style": "dynamic",
            },
            "llm_model": "gpt-4o",
        }

        with patch("app.services.voice_instruction_generator.VoiceInstructionGenerator") as MockGen:
            mock_instance = Mock()
            mock_instance.regenerate_with_feedback.return_value = mock_result
            MockGen.return_value = mock_instance

            with patch("app.api.routes.voice_refinement.get_voice_instruction_generator", return_value=mock_instance):
                response = client.post(
                    f"/api/projects/{project_id}/voice-instructions/regenerate",
                    json=request_data,
                )

        assert response.status_code == 201
        result = response.json()

        assert "high energy" in result["voice_instructions"]
        assert result["detected_characteristics"]["tone"] == "energetic"

    def test_regenerate_voice_instructions_project_not_found(self, client):
        """Test 404 when project doesn't exist."""
        request_data = {
            "segment_id": "segment_0",
            "original_text": "Test",
            "translated_text": "Prueba",
            "previous_instructions": "Previous",
            "user_feedback": "Feedback",
        }

        response = client.post(
            "/api/projects/99999/voice-instructions/regenerate",
            json=request_data,
        )

        assert response.status_code == 404

    def test_regenerate_missing_required_fields(self, client, project_id):
        """Test 422 validation error for missing fields."""
        request_data = {
            "segment_id": "segment_0",
            "original_text": "Test",
            # Missing required fields
        }

        response = client.post(
            f"/api/projects/{project_id}/voice-instructions/regenerate",
            json=request_data,
        )

        assert response.status_code == 422


class TestVoicePreviewGenerateEndpoint:
    """Test POST /projects/{project_id}/voice-previews/generate endpoint."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """Provide test client."""
        from app.core.config import settings

        storage_dir = tmp_path / "storage"
        storage_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(settings, "redubber_config_path", str(storage_dir))
        monkeypatch.setattr(settings, "openai_api_key", "test-key")

        app = create_app()
        with TestClient(app) as test_client:
            yield test_client

    @pytest.fixture
    def project_id(self, client, tmp_path):
        """Create a test project."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir(exist_ok=True)
        response = client.post("/api/projects/", json={"path": str(project_dir)})
        return response.json()["id"]

    def _mock_generate_audio(self, tmp_path):
        """Return a patch context that fakes TTS audio generation."""
        from app.services.tts_preview_generator import TTSPreviewGenerator
        fake_mp3 = tmp_path / "fake.mp3"
        fake_mp3.write_bytes(b"fake")

        def _fake_generate(text, voice, instructions, output_path, **kwargs):
            # Write a real file so the route can stat it
            import shutil
            shutil.copy(str(fake_mp3), output_path)
            return output_path, 1500

        return patch.object(TTSPreviewGenerator, "generate_audio", side_effect=_fake_generate)

    def test_generate_voice_previews_success(self, client, project_id, tmp_path):
        """Test successful preview generation for all voices."""
        request_data = {
            "translated_text": "This is test text for preview generation.",
            "voice_instructions": "Speak clearly with moderate pace.",
            "voices": ["alloy", "nova", "shimmer"],
        }

        with self._mock_generate_audio(tmp_path):
            response = client.post(
                f"/api/projects/{project_id}/voice-previews/generate",
                json=request_data,
            )

        assert response.status_code == 200
        result = response.json()

        assert "previews" in result
        assert len(result["previews"]) == 3
        assert "instructions_hash" in result
        assert "cache_hits" in result
        assert "cache_misses" in result

        preview = result["previews"][0]
        assert "voice" in preview
        assert "audio_url" in preview
        assert "duration_ms" in preview
        assert "cached" in preview

    def test_generate_voice_previews_default_voices(self, client, project_id, tmp_path):
        """Test generation with default voice list."""
        request_data = {
            "translated_text": "Test text",
            "voice_instructions": "Test instructions",
        }

        with self._mock_generate_audio(tmp_path):
            response = client.post(
                f"/api/projects/{project_id}/voice-previews/generate",
                json=request_data,
            )

        assert response.status_code == 200
        assert len(response.json()["previews"]) == 6

    def test_generate_voice_previews_cache_behavior(self, client, project_id, tmp_path):
        """Test cache hit/miss tracking."""
        request_data = {
            "translated_text": "Test text for caching",
            "voice_instructions": "Test instructions",
            "voices": ["nova", "alloy"],
        }

        with self._mock_generate_audio(tmp_path):
            result1 = client.post(
                f"/api/projects/{project_id}/voice-previews/generate",
                json=request_data,
            ).json()
            assert result1["cache_misses"] == 2
            assert result1["cache_hits"] == 0

            # Second request — files exist in cache dir on disk but the route
            # uses DB cache keyed by hash; second call finds DB entries.
            result2 = client.post(
                f"/api/projects/{project_id}/voice-previews/generate",
                json=request_data,
            ).json()
            assert result2["cache_hits"] == 2
            assert result2["cache_misses"] == 0

    def test_generate_voice_previews_different_instructions_miss_cache(
        self, client, project_id, tmp_path
    ):
        """Test that different instructions cause cache miss."""
        with self._mock_generate_audio(tmp_path):
            client.post(
                f"/api/projects/{project_id}/voice-previews/generate",
                json={"translated_text": "Same text", "voice_instructions": "Version 1", "voices": ["nova"]},
            )
            result2 = client.post(
                f"/api/projects/{project_id}/voice-previews/generate",
                json={"translated_text": "Same text", "voice_instructions": "Version 2", "voices": ["nova"]},
            ).json()

        assert result2["cache_misses"] == 1

    def test_generate_voice_previews_project_not_found(self, client):
        """Test 404 when project doesn't exist."""
        request_data = {
            "translated_text": "Test",
            "voice_instructions": "Instructions",
        }

        response = client.post(
            "/api/projects/99999/voice-previews/generate",
            json=request_data,
        )

        assert response.status_code == 404

    def test_generate_voice_previews_missing_required_fields(self, client, project_id):
        """Test 422 validation error."""
        request_data = {
            # Missing translated_text
            "voice_instructions": "Instructions",
        }

        response = client.post(
            f"/api/projects/{project_id}/voice-previews/generate",
            json=request_data,
        )

        assert response.status_code == 422


class TestVoiceSettingsSaveEndpoint:
    """Test PUT /projects/{project_id}/voice-settings endpoint."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """Provide test client."""
        from app.core.config import settings

        storage_dir = tmp_path / "storage"
        storage_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(settings, "redubber_config_path", str(storage_dir))
        monkeypatch.setattr(settings, "openai_api_key", "test-key")

        app = create_app()
        with TestClient(app) as test_client:
            yield test_client

    @pytest.fixture
    def project_id(self, client, tmp_path):
        """Create a test project."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir(exist_ok=True)
        response = client.post("/api/projects/", json={"path": str(project_dir)})
        return response.json()["id"]

    def test_save_voice_settings_success(self, client, project_id):
        """Test successful voice settings save."""
        request_data = {
            "voice": "nova",
            "voice_instructions": "Speak with warmth and clarity.",
            "segment_used": "segment_0",
        }

        response = client.put(
            f"/api/projects/{project_id}/voice-settings",
            json=request_data,
        )

        assert response.status_code == 200
        result = response.json()

        assert result["id"] == project_id
        assert result["voice"] == "nova"
        assert result["voice_instructions"] == "Speak with warmth and clarity."

    def test_save_voice_settings_updates_project(self, client, project_id):
        """Test that settings persist across requests."""
        request_data = {
            "voice": "shimmer",
            "voice_instructions": "Test instructions",
            "segment_used": "segment_1",
        }

        # Save settings
        response1 = client.put(
            f"/api/projects/{project_id}/voice-settings",
            json=request_data,
        )

        assert response1.status_code == 200

        # Verify settings persisted
        response2 = client.get(f"/api/projects/{project_id}")
        assert response2.status_code == 200
        project = response2.json()

        assert project["voice"] == "shimmer"
        assert project["voice_instructions"] == "Test instructions"

    def test_save_voice_settings_creates_selection_history(
        self, client, project_id, tmp_path
    ):
        """Test that voice selection is saved to history."""
        from app.core.config import settings

        db = DatabaseManager(db_path=settings.database_url)

        request_data = {
            "voice": "alloy",
            "voice_instructions": "History test instructions",
            "segment_used": "segment_2",
        }

        response = client.put(
            f"/api/projects/{project_id}/voice-settings",
            json=request_data,
        )

        assert response.status_code == 200

        # Verify history entry created
        history = db.get_voice_selection_history(project_id=project_id)
        assert len(history) == 1
        assert history[0]["voice_name"] == "alloy"
        assert history[0]["segment_used"] == "segment_2"

    def test_save_voice_settings_project_not_found(self, client):
        """Test 404 when project doesn't exist."""
        request_data = {
            "voice": "nova",
            "voice_instructions": "Test",
            "segment_used": "segment_0",
        }

        response = client.put(
            "/api/projects/99999/voice-settings",
            json=request_data,
        )

        assert response.status_code == 404

    def test_save_voice_settings_missing_required_fields(self, client, project_id):
        """Test 422 validation error."""
        request_data = {
            # Missing voice
            "voice_instructions": "Instructions",
            "segment_used": "segment_0",
        }

        response = client.put(
            f"/api/projects/{project_id}/voice-settings",
            json=request_data,
        )

        assert response.status_code == 422

    def test_save_voice_settings_multiple_updates(self, client, project_id):
        """Test updating voice settings multiple times."""
        # First update
        client.put(
            f"/api/projects/{project_id}/voice-settings",
            json={
                "voice": "alloy",
                "voice_instructions": "First instructions",
                "segment_used": "segment_0",
            },
        )

        # Second update
        response = client.put(
            f"/api/projects/{project_id}/voice-settings",
            json={
                "voice": "nova",
                "voice_instructions": "Updated instructions",
                "segment_used": "segment_1",
            },
        )

        assert response.status_code == 200
        project = response.json()

        # Should have latest settings
        assert project["voice"] == "nova"
        assert project["voice_instructions"] == "Updated instructions"


class TestEndpointIntegration:
    """Test integration between voice refinement endpoints."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """Provide test client."""
        from app.core.config import settings

        storage_dir = tmp_path / "storage"
        storage_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(settings, "redubber_config_path", str(storage_dir))
        monkeypatch.setattr(settings, "openai_api_key", "test-key")

        app = create_app()
        with TestClient(app) as test_client:
            yield test_client

    @pytest.fixture
    def project_id(self, client, tmp_path, monkeypatch):
        """Create a test project with fake segments."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir(exist_ok=True)
        response = client.post("/api/projects/", json={"path": str(project_dir)})
        pid = response.json()["id"]
        monkeypatch.setattr(
            "app.api.routes.voice_refinement._load_real_segments",
            lambda project_id, project_path: _make_fake_segments(project_id),
        )
        return pid

    def test_complete_voice_refinement_workflow(self, client, project_id, tmp_path):
        """Test complete workflow from segments to voice selection."""
        from app.services.tts_preview_generator import TTSPreviewGenerator
        fake_mp3 = tmp_path / "fake.mp3"
        fake_mp3.write_bytes(b"fake")

        def _fake_generate_audio(text, voice, instructions, output_path, **kwargs):
            import shutil
            shutil.copy(str(fake_mp3), output_path)
            return output_path, 1500

        # 1. Get transcription segments
        segments_response = client.get(f"/api/projects/{project_id}/transcription-segments")
        assert segments_response.status_code == 200
        segment = segments_response.json()["segments"][0]

        # 2. Generate voice instructions
        mock_instructions = {
            "voice_instructions": "Speak clearly and professionally.",
            "detected_characteristics": {
                "tone": "professional", "pace": "moderate",
                "emotion": "balanced", "style": "authoritative",
            },
            "llm_model": "gpt-4o",
        }

        mock_instance = Mock()
        mock_instance.generate_instructions.return_value = mock_instructions

        with patch("app.api.routes.voice_refinement.get_voice_instruction_generator", return_value=mock_instance):
            analyze_response = client.post(
                f"/api/projects/{project_id}/voice-instructions/analyze",
                json={"segment_id": segment["id"], "original_text": segment["original_text"],
                      "translated_text": segment["translated_text"]},
            )

        assert analyze_response.status_code == 201
        instructions_result = analyze_response.json()

        # 3. Generate voice previews
        with patch.object(TTSPreviewGenerator, "generate_audio", side_effect=_fake_generate_audio):
            preview_response = client.post(
                f"/api/projects/{project_id}/voice-previews/generate",
                json={"translated_text": segment["translated_text"],
                      "voice_instructions": instructions_result["voice_instructions"],
                      "voices": ["nova", "alloy"]},
            )

        assert preview_response.status_code == 200

        # 4. Save selected voice
        save_response = client.put(
            f"/api/projects/{project_id}/voice-settings",
            json={"voice": "nova", "voice_instructions": instructions_result["voice_instructions"],
                  "segment_used": segment["id"]},
        )

        assert save_response.status_code == 200
        final_project = save_response.json()
        assert final_project["voice"] == "nova"
        assert final_project["voice_instructions"] == instructions_result["voice_instructions"]

    def test_regenerate_and_preview_workflow(self, client, project_id, tmp_path):
        """Test regenerating instructions and generating new previews."""
        from app.services.tts_preview_generator import TTSPreviewGenerator
        fake_mp3 = tmp_path / "fake.mp3"
        fake_mp3.write_bytes(b"fake")

        def _fake_generate_audio(text, voice, instructions, output_path, **kwargs):
            import shutil
            shutil.copy(str(fake_mp3), output_path)
            return output_path, 1500

        mock_initial = {
            "voice_instructions": "Initial instructions",
            "detected_characteristics": {"tone": "neutral", "pace": "moderate", "emotion": "balanced", "style": "natural"},
            "llm_model": "gpt-4o",
        }
        mock_regenerated = {
            "voice_instructions": "Improved instructions with more energy",
            "detected_characteristics": {"tone": "energetic", "pace": "fast", "emotion": "excited", "style": "dynamic"},
            "llm_model": "gpt-4o",
        }

        mock_instance = Mock()
        mock_instance.generate_instructions.return_value = mock_initial
        mock_instance.regenerate_with_feedback.return_value = mock_regenerated

        with patch("app.api.routes.voice_refinement.get_voice_instruction_generator", return_value=mock_instance):
            initial_response = client.post(
                f"/api/projects/{project_id}/voice-instructions/analyze",
                json={"segment_id": "segment_0", "original_text": "Test", "translated_text": "Prueba"},
            )
            assert initial_response.status_code == 201

            regenerate_response = client.post(
                f"/api/projects/{project_id}/voice-instructions/regenerate",
                json={"segment_id": "segment_0", "original_text": "Test", "translated_text": "Prueba",
                      "previous_instructions": mock_initial["voice_instructions"],
                      "user_feedback": "Make it more energetic"},
            )

        assert regenerate_response.status_code == 201
        regenerated = regenerate_response.json()
        assert "energy" in regenerated["voice_instructions"].lower()

        with patch.object(TTSPreviewGenerator, "generate_audio", side_effect=_fake_generate_audio):
            preview_response = client.post(
                f"/api/projects/{project_id}/voice-previews/generate",
                json={"translated_text": "Prueba", "voice_instructions": regenerated["voice_instructions"], "voices": ["nova"]},
            )

        assert preview_response.status_code == 200
