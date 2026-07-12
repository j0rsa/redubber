"""
LLM-based voice instruction generation service.
Uses OpenAI GPT-4 to analyze transcription segments and generate
detailed voice instructions for TTS.
"""

import base64
import json
import os
from typing import Dict, Optional

from openai import OpenAI


class VoiceInstructionGenerator:
    """Generate voice instructions using LLM analysis."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"  # Use latest GPT-4 model

    def generate_instructions(
        self,
        original_text: str,
        translated_text: str,
        context: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        Generate voice instructions by analyzing the transcription segment.

        Args:
            original_text: Original transcription text
            translated_text: Translated text (target language)
            context: Optional context information (content_type, speaker_gender, etc.)

        Returns:
            Dictionary with voice_instructions and detected_characteristics
        """
        context = context or {}
        content_type = context.get("content_type", "general")
        speaker_gender = context.get("speaker_gender", "unknown")
        speaker_age = context.get("speaker_age", "adult")
        source_language = context.get("source_language", "")

        prompt = self._build_prompt(
            original_text, translated_text, content_type, speaker_gender, speaker_age,
            source_language=source_language,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional voice dubbing director. You write reusable TTS speaker-profile instructions that capture how a person sounds — their pitch, energy, rhythm, and delivery style — so a TTS engine can consistently portray them across an entire dubbed video.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            result = json.loads(content)

            return {
                "voice_instructions": result.get(
                    "voice_instructions", "Speak naturally with clear enunciation."
                ),
                "detected_characteristics": result.get("detected_characteristics", {}),
                "llm_model": self.model,
            }

        except Exception as e:
            # Fallback to generic instructions on error
            return {
                "voice_instructions": "Speak naturally with clear enunciation and appropriate pacing. Match the tone of the original content.",
                "detected_characteristics": {
                    "tone": "neutral",
                    "pace": "moderate",
                    "emotion": "balanced",
                    "style": "natural",
                },
                "llm_model": self.model,
                "error": str(e),
            }

    def generate_instructions_from_audio(
        self,
        audio_bytes: bytes,
        original_text: str,
        translated_text: str,
        context: Optional[Dict[str, str]] = None,
        audio_model: str = "gpt-4o-audio-preview",
    ) -> Dict:
        """Generate voice instructions by listening to the actual audio clip.

        Uses gpt-4o-audio-preview to analyse pitch, gender, accent, and delivery
        directly from the audio signal — far more reliable than text-only inference.
        Falls back to text-only on any error.

        Args:
            audio_bytes: Raw MP3 audio bytes for the segment clip.
            original_text: Transcription text (for context).
            translated_text: Translation text (for context).
            context: Optional extra context dict.

        Returns:
            Same dict shape as generate_instructions.
        """
        context = context or {}
        content_type = context.get("content_type", "general")
        speaker_gender = context.get("speaker_gender", "unknown")
        speaker_age = context.get("speaker_age", "adult")
        source_language = context.get("source_language", "")

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        prompt = self._build_prompt(
            original_text, translated_text, content_type, speaker_gender, speaker_age,
            source_language=source_language,
        )

        try:
            response = self.client.chat.completions.create(
                model=audio_model,
                modalities=["text"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional voice dubbing director. "
                            "You write reusable TTS speaker-profile instructions that capture how a person sounds — "
                            "their pitch, gender, rhythm, and delivery style — so a TTS engine can consistently "
                            "portray them across an entire dubbed video. "
                            "You have access to the actual audio, so infer gender, pitch, and accent from what you hear. "
                            "Always respond with valid JSON only — no markdown, no code fences."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_b64, "format": "mp3"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                temperature=0.7,
                max_tokens=600,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response")

            # Strip markdown code fences if present
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```", 2)[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.rsplit("```", 1)[0].strip()

            result = json.loads(cleaned)
            return {
                "voice_instructions": result.get("voice_instructions", "Speak naturally with clear enunciation."),
                "detected_characteristics": result.get("detected_characteristics", {}),
                "llm_model": "gpt-4o-audio-preview",
            }

        except Exception as e:
            import logging as _logging
            _log = _logging.getLogger(__name__)
            _log.warning(
                "Audio-model voice analysis failed (%s: %s) — falling back to text-only. "
                "If this is a 404, the model '%s' may not be enabled on your OpenAI account.",
                type(e).__name__, e, audio_model,
            )
            return self.generate_instructions(original_text, translated_text, context)

    def regenerate_with_feedback(
        self,
        original_text: str,
        translated_text: str,
        previous_instructions: str,
        user_feedback: str,
        context: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        Regenerate instructions incorporating user feedback.

        Args:
            original_text: Original transcription text
            translated_text: Translated text
            previous_instructions: Previously generated instructions
            user_feedback: User's feedback on what to improve
            context: Optional context information

        Returns:
            Dictionary with improved voice_instructions
        """
        context = context or {}

        prompt = f"""
You are a professional voice dubbing director refining reusable TTS speaker-profile instructions based on user feedback.

These instructions describe the speaker's persistent vocal identity (pitch, energy, pace, accent cues, delivery style) and will be applied to every TTS segment in the dubbed video — not just this sample.

Reference sample transcription:
{original_text}

Translation (context only):
{translated_text}

Current instructions:
{previous_instructions}

User feedback:
{user_feedback}

Revise the instructions to incorporate the feedback. Keep them reusable and speaker-profile focused — do not make them specific to this sample's content. Preserve the accent sentence and speaker gender unless the feedback explicitly changes them.

Format your response as JSON:
{{
  "voice_instructions": "Revised reusable speaker-profile instructions (80-160 words)...",
  "detected_characteristics": {{
    "tone": "...",
    "pace": "...",
    "energy": "...",
    "style": "...",
    "speaker_gender": "male | female | unknown"
  }},
  "improvements_made": "One sentence describing what changed"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional voice dubbing director. You refine reusable TTS speaker-profile instructions so they consistently portray a speaker's vocal identity across an entire dubbed video.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=600,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            result = json.loads(content)

            return {
                "voice_instructions": result.get("voice_instructions", previous_instructions),
                "detected_characteristics": result.get("detected_characteristics", {}),
                "improvements_made": result.get("improvements_made", ""),
                "llm_model": self.model,
            }

        except Exception as e:
            return {
                "voice_instructions": previous_instructions,
                "detected_characteristics": {},
                "error": str(e),
            }

    def _build_prompt(
        self,
        original_text: str,
        translated_text: str,
        content_type: str,
        speaker_gender: str,
        speaker_age: str,
        source_language: str = "",
    ) -> str:
        """Build the LLM prompt for voice instruction generation."""
        accent_instruction = (
            f"The speaker's native language is **{source_language}**. "
            f"The instructions MUST include a concrete accent sentence describing exactly how a native {source_language} speaker sounds when speaking the target language — "
            f"specific phonetic traits such as vowel quality, consonant softening, intonation flatness, rhythm patterns, and any characteristic stress patterns. "
            f'Example format: "Speak with a {source_language} accent — [specific phonetic details]." '
            f"Do not be vague — name the actual sounds that differ from native target-language speech."
        ) if source_language else (
            "The instructions MUST include a sentence about accent: infer the speaker's likely native language from the transcription text and describe the non-native accent flavour they would carry in the target language with specific phonetic details."
        )

        return f"""
You are a professional voice dubbing director. Your task is to write reusable TTS voice instructions that capture how the original speaker *sounds as a person* — not what they are saying in this particular segment.

The instructions will be applied to every TTS segment in the dubbed video. Describe the speaker's persistent vocal identity: pitch register, energy level, speaking rhythm, accent flavour, and delivery style. Do NOT reference the content of this sample.

Original audio transcription ({content_type}, speaker: {speaker_gender}, {speaker_age}):
{original_text}

Translation (context only):
{translated_text}

{accent_instruction}

Also infer the speaker's gender from context clues.

Format your response as JSON:
{{
  "voice_instructions": "Reusable speaker-profile instructions (80-160 words). Lead with the most defining vocal trait. The accent sentence is mandatory.",
  "detected_characteristics": {{
    "tone": "e.g. warm and composed",
    "pace": "e.g. measured with deliberate pauses",
    "energy": "e.g. calm authority",
    "style": "e.g. educational lecturer",
    "speaker_gender": "male | female | unknown"
  }}
}}
"""


# Singleton instance
_generator_instance: Optional[VoiceInstructionGenerator] = None


def get_voice_instruction_generator() -> VoiceInstructionGenerator:
    """Get or create the singleton voice instruction generator."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = VoiceInstructionGenerator()
    return _generator_instance
