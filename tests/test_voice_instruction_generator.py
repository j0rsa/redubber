"""
Tests for VoiceInstructionGenerator service.
Tests LLM-based voice instruction generation with OpenAI GPT-4.
"""

import json
from unittest.mock import Mock, patch

import pytest

from app.services.voice_instruction_generator import (
    VoiceInstructionGenerator,
    get_voice_instruction_generator,
)


class TestVoiceInstructionGeneratorInit:
    """Test initialization and configuration."""

    def test_init_with_explicit_api_key(self):
        """Test initialization with explicitly provided API key."""
        generator = VoiceInstructionGenerator(api_key="test-key-123")

        assert generator.api_key == "test-key-123"
        assert generator.model == "gpt-4o"
        assert generator.client is not None

    def test_init_with_env_api_key(self, monkeypatch):
        """Test initialization with API key from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")

        generator = VoiceInstructionGenerator()

        assert generator.api_key == "env-test-key"

    def test_init_without_api_key_raises_error(self, monkeypatch):
        """Test initialization without API key raises ValueError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OpenAI API key not found"):
            VoiceInstructionGenerator()


class TestGenerateInstructions:
    """Test voice instruction generation."""

    @pytest.fixture
    def generator(self):
        """Provide generator instance with test API key."""
        return VoiceInstructionGenerator(api_key="test-key")

    @pytest.fixture
    def mock_openai_response(self):
        """Provide mock OpenAI API response."""
        return {
            "voice_instructions": "Speak with a warm, professional tone at a moderate pace. "
            "Use clear enunciation and emphasize key technical terms. "
            "Maintain an engaging, conversational style while conveying authority on the subject matter.",
            "detected_characteristics": {
                "tone": "warm, professional",
                "pace": "moderate",
                "emotion": "engaged, enthusiastic",
                "style": "conversational, authoritative",
            },
        }

    def test_generate_instructions_success(self, generator, mock_openai_response):
        """Test successful instruction generation with valid inputs."""
        original_text = "This is a sample lecture on machine learning."
        translated_text = "Dies ist eine Beispielvorlesung über maschinelles Lernen."

        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_openai_response)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response):
            result = generator.generate_instructions(
                original_text=original_text,
                translated_text=translated_text,
                context={"content_type": "educational", "speaker_gender": "male"},
            )

        assert result["voice_instructions"] == mock_openai_response["voice_instructions"]
        assert result["detected_characteristics"] == mock_openai_response["detected_characteristics"]
        assert result["llm_model"] == "gpt-4o"
        assert "error" not in result

    def test_generate_instructions_with_minimal_context(self, generator, mock_openai_response):
        """Test generation with minimal context (defaults applied)."""
        original_text = "Hello world"
        translated_text = "Hola mundo"

        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_openai_response)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response) as mock_create:
            result = generator.generate_instructions(
                original_text=original_text,
                translated_text=translated_text,
            )

            # Verify default context values are used
            call_args = mock_create.call_args
            prompt = call_args.kwargs["messages"][1]["content"]
            assert "Content Type: general" in prompt
            assert "Speaker Gender: unknown" in prompt
            assert "Speaker Age: adult" in prompt

        assert "voice_instructions" in result
        assert "detected_characteristics" in result

    def test_generate_instructions_with_full_context(self, generator, mock_openai_response):
        """Test generation with full context provided."""
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_openai_response)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        context = {
            "content_type": "news",
            "speaker_gender": "female",
            "speaker_age": "senior",
        }

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response) as mock_create:
            result = generator.generate_instructions(
                original_text="Breaking news today",
                translated_text="Última hora hoy",
                context=context,
            )

            # Verify context is passed to prompt
            call_args = mock_create.call_args
            prompt = call_args.kwargs["messages"][1]["content"]
            assert "Content Type: news" in prompt
            assert "Speaker Gender: female" in prompt
            assert "Speaker Age: senior" in prompt

        assert result["voice_instructions"] is not None

    def test_generate_instructions_api_error_returns_fallback(self, generator):
        """Test that API errors return fallback instructions."""
        with patch.object(
            generator.client.chat.completions,
            "create",
            side_effect=Exception("API connection failed"),
        ):
            result = generator.generate_instructions(
                original_text="Test text",
                translated_text="Texto de prueba",
            )

        # Should return fallback instructions
        assert "Speak naturally with clear enunciation" in result["voice_instructions"]
        assert result["detected_characteristics"]["tone"] == "neutral"
        assert result["detected_characteristics"]["pace"] == "moderate"
        assert result["error"] == "API connection failed"
        assert result["llm_model"] == "gpt-4o"

    def test_generate_instructions_empty_response_raises_error(self, generator):
        """Test handling of empty response from LLM."""
        mock_choice = Mock()
        mock_choice.message.content = None

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response):
            result = generator.generate_instructions(
                original_text="Test",
                translated_text="Prueba",
            )

        # Should return fallback due to empty response
        assert "error" in result
        assert "Empty response from LLM" in result["error"]

    def test_generate_instructions_invalid_json_returns_fallback(self, generator):
        """Test handling of malformed JSON response."""
        mock_choice = Mock()
        mock_choice.message.content = "This is not valid JSON"

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response):
            result = generator.generate_instructions(
                original_text="Test",
                translated_text="Prueba",
            )

        # Should return fallback due to JSON parse error
        assert "error" in result
        assert result["voice_instructions"] is not None

    def test_prompt_building(self, generator):
        """Test that prompt is built correctly with all parameters."""
        prompt = generator._build_prompt(
            original_text="Original sample text",
            translated_text="Translated sample text",
            content_type="educational",
            speaker_gender="female",
            speaker_age="young_adult",
        )

        # Verify prompt contains all key elements
        assert "Original sample text" in prompt
        assert "Translated sample text" in prompt
        assert "educational" in prompt
        assert "female" in prompt
        assert "young_adult" in prompt
        assert "Tone" in prompt
        assert "Pace" in prompt
        assert "Emotion" in prompt
        assert "Style" in prompt
        assert "JSON" in prompt

    def test_api_call_parameters(self, generator, mock_openai_response):
        """Test that API is called with correct parameters."""
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_openai_response)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response) as mock_create:
            generator.generate_instructions(
                original_text="Test",
                translated_text="Prueba",
            )

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs

            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 500
            assert call_kwargs["response_format"] == {"type": "json_object"}
            assert len(call_kwargs["messages"]) == 2
            assert call_kwargs["messages"][0]["role"] == "system"
            assert call_kwargs["messages"][1]["role"] == "user"


class TestRegenerateWithFeedback:
    """Test instruction regeneration with user feedback."""

    @pytest.fixture
    def generator(self):
        """Provide generator instance with test API key."""
        return VoiceInstructionGenerator(api_key="test-key")

    def test_regenerate_with_feedback_success(self, generator):
        """Test successful regeneration with user feedback."""
        mock_response_data = {
            "voice_instructions": "Speak with MORE energy and enthusiasm! Use a faster pace "
            "and dynamic intonation to convey excitement.",
            "detected_characteristics": {
                "tone": "energetic, enthusiastic",
                "pace": "fast",
                "emotion": "excited, passionate",
                "style": "dynamic, expressive",
            },
            "improvements_made": "Increased energy level, faster pace, more dynamic delivery",
        }

        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_response_data)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response):
            result = generator.regenerate_with_feedback(
                original_text="This is exciting news!",
                translated_text="¡Estas son noticias emocionantes!",
                previous_instructions="Speak calmly and slowly.",
                user_feedback="Make it more energetic and exciting!",
            )

        assert "MORE energy" in result["voice_instructions"]
        assert result["detected_characteristics"]["pace"] == "fast"
        assert result["improvements_made"] == "Increased energy level, faster pace, more dynamic delivery"
        assert result["llm_model"] == "gpt-4o"
        assert "error" not in result

    def test_regenerate_with_context(self, generator):
        """Test regeneration with additional context."""
        mock_response_data = {
            "voice_instructions": "Updated instructions with context",
            "detected_characteristics": {
                "tone": "professional",
                "pace": "moderate",
                "emotion": "balanced",
                "style": "authoritative",
            },
            "improvements_made": "Applied feedback",
        }

        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_response_data)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        context = {"content_type": "news", "speaker_gender": "male"}

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response):
            result = generator.regenerate_with_feedback(
                original_text="Breaking news",
                translated_text="Noticias de última hora",
                previous_instructions="Previous instructions",
                user_feedback="Make it more authoritative",
                context=context,
            )

        assert result["voice_instructions"] == "Updated instructions with context"

    def test_regenerate_api_error_returns_previous(self, generator):
        """Test that API errors return previous instructions."""
        previous_instructions = "Original voice instructions"

        with patch.object(
            generator.client.chat.completions,
            "create",
            side_effect=Exception("API timeout"),
        ):
            result = generator.regenerate_with_feedback(
                original_text="Test",
                translated_text="Prueba",
                previous_instructions=previous_instructions,
                user_feedback="Make it better",
            )

        # Should return previous instructions on error
        assert result["voice_instructions"] == previous_instructions
        assert result["error"] == "API timeout"
        assert result["detected_characteristics"] == {}

    def test_regenerate_empty_response(self, generator):
        """Test handling of empty response during regeneration."""
        previous_instructions = "Previous instructions"

        mock_choice = Mock()
        mock_choice.message.content = None

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response):
            result = generator.regenerate_with_feedback(
                original_text="Test",
                translated_text="Prueba",
                previous_instructions=previous_instructions,
                user_feedback="Improve this",
            )

        # Should return previous instructions
        assert result["voice_instructions"] == previous_instructions
        assert "error" in result

    def test_regenerate_prompt_includes_all_elements(self, generator):
        """Test that regeneration prompt includes all required elements."""
        mock_choice = Mock()
        mock_choice.message.content = json.dumps({
            "voice_instructions": "Test",
            "detected_characteristics": {},
        })

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response) as mock_create:
            generator.regenerate_with_feedback(
                original_text="Original text here",
                translated_text="Translated text here",
                previous_instructions="Previous instructions here",
                user_feedback="User feedback here",
            )

            call_args = mock_create.call_args
            prompt = call_args.kwargs["messages"][1]["content"]

            # Verify all elements are in prompt
            assert "Original text here" in prompt
            assert "Translated text here" in prompt
            assert "Previous instructions here" in prompt
            assert "User feedback here" in prompt

    def test_regenerate_api_parameters(self, generator):
        """Test that regeneration API call uses correct parameters."""
        mock_choice = Mock()
        mock_choice.message.content = json.dumps({
            "voice_instructions": "Test",
            "detected_characteristics": {},
        })

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response) as mock_create:
            generator.regenerate_with_feedback(
                original_text="Test",
                translated_text="Prueba",
                previous_instructions="Previous",
                user_feedback="Feedback",
            )

            call_kwargs = mock_create.call_args.kwargs

            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 600  # Higher than initial generation
            assert call_kwargs["response_format"] == {"type": "json_object"}


class TestSingletonPattern:
    """Test singleton pattern for generator instance."""

    def test_get_voice_instruction_generator_creates_instance(self, monkeypatch):
        """Test that get function creates instance on first call."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Clear singleton
        import app.services.voice_instruction_generator as module
        module._generator_instance = None

        generator = get_voice_instruction_generator()

        assert generator is not None
        assert isinstance(generator, VoiceInstructionGenerator)

    def test_get_voice_instruction_generator_returns_same_instance(self, monkeypatch):
        """Test that get function returns same instance on subsequent calls."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Clear singleton
        import app.services.voice_instruction_generator as module
        module._generator_instance = None

        generator1 = get_voice_instruction_generator()
        generator2 = get_voice_instruction_generator()

        assert generator1 is generator2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def generator(self):
        """Provide generator instance."""
        return VoiceInstructionGenerator(api_key="test-key")

    def test_very_long_text(self, generator):
        """Test handling of very long input text."""
        long_text = "This is a very long text. " * 500  # ~3000+ words

        mock_response_data = {
            "voice_instructions": "Handle long text appropriately",
            "detected_characteristics": {
                "tone": "neutral",
                "pace": "moderate",
                "emotion": "balanced",
                "style": "natural",
            },
        }

        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_response_data)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response):
            result = generator.generate_instructions(
                original_text=long_text,
                translated_text=long_text,
            )

        assert result["voice_instructions"] is not None

    def test_empty_strings(self, generator):
        """Test handling of empty input strings."""
        mock_response_data = {
            "voice_instructions": "Default instructions",
            "detected_characteristics": {
                "tone": "neutral",
                "pace": "moderate",
                "emotion": "balanced",
                "style": "natural",
            },
        }

        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_response_data)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response):
            result = generator.generate_instructions(
                original_text="",
                translated_text="",
            )

        assert result["voice_instructions"] is not None

    def test_special_characters_in_text(self, generator):
        """Test handling of special characters and unicode."""
        special_text = "Test with émojis 😀 and spëcial çharacters: <>&\"'"

        mock_response_data = {
            "voice_instructions": "Handle special characters",
            "detected_characteristics": {
                "tone": "neutral",
                "pace": "moderate",
                "emotion": "balanced",
                "style": "natural",
            },
        }

        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_response_data)

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch.object(generator.client.chat.completions, "create", return_value=mock_response):
            result = generator.generate_instructions(
                original_text=special_text,
                translated_text=special_text,
            )

        assert result["voice_instructions"] is not None
