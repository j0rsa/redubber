"""
LLM-based voice instruction generation service.
Uses OpenAI models to analyze transcription segments (and optional audio)
and generate authentic, accent-forward TTS speaker instructions.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Dict, Optional

from openai import OpenAI

from app.services.accent_playbooks import (
    accent_intensity_guidance,
    format_playbook_block,
    normalize_language_code,
)


def _resolve_openai_api_key(api_key: Optional[str] = None) -> str:
    """Resolve OpenAI API key from explicit arg, settings DB, or env var."""
    if api_key:
        return api_key
    try:
        from app.services.settings_service import get_openai_api_key

        resolved = get_openai_api_key()
        if resolved:
            return resolved
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")


_SYSTEM_TEXT = (
    "You are a voice-dubbing director specializing in authentic non-native English. "
    "You write TTS speaker-profile instructions so English dialogue sounds like a specific "
    "L1 speaker performing English — concrete phonetics first, then rhythm, pitch, and attitude. "
    "Prefer vivid, usable direction over generic 'warm professional' filler. "
    "Light humour/charm is welcome when it fits; never mock or stereotype cruelly. "
    "Always respond with valid JSON only."
)

_SYSTEM_AUDIO = (
    "You are a voice-dubbing director specializing in authentic non-native English. "
    "You can hear the speaker — infer gender, pitch, energy, and accent from the audio. "
    "Write TTS instructions so English sounds like THIS person speaking English: "
    "concrete L1→English phonetics first, then rhythm, pitch, attitude. "
    "Light humour/charm when the delivery allows; never cruel parody. "
    "Always respond with valid JSON only — no markdown, no code fences."
)

_SYSTEM_REGENERATE = (
    "You are a voice-dubbing director refining TTS speaker-profile instructions. "
    "Keep accent authenticity high: phonetics first, then delivery. "
    "Incorporate user feedback precisely. Always respond with valid JSON only."
)


class VoiceInstructionGenerator:
    """Generate voice instructions using LLM analysis."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key."""
        self.api_key = _resolve_openai_api_key(api_key)
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
        prompt = self._build_prompt(original_text, translated_text, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_TEXT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.75,
                max_tokens=900,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            result = json.loads(content)
            return self._normalize_result(result, llm_model=self.model)

        except Exception as e:
            return self._fallback(str(e))

    def generate_instructions_from_audio(
        self,
        audio_bytes: bytes,
        original_text: str,
        translated_text: str,
        context: Optional[Dict[str, str]] = None,
        audio_model: str = "gpt-4o-audio-preview",
    ) -> Dict:
        """Generate voice instructions by listening to the actual audio clip.

        Uses an audio-capable model to analyse pitch, gender, accent, and delivery
        directly from the audio signal — far more reliable than text-only inference.
        Falls back to text-only on any error.
        """
        context = context or {}
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        prompt = self._build_prompt(original_text, translated_text, context)

        try:
            response = self.client.chat.completions.create(
                model=audio_model,
                modalities=["text"],
                messages=[
                    {"role": "system", "content": _SYSTEM_AUDIO},
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
                temperature=0.75,
                max_tokens=900,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response")

            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```", 2)[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.rsplit("```", 1)[0].strip()

            result = json.loads(cleaned)
            return self._normalize_result(result, llm_model=audio_model)

        except Exception as e:
            import logging as _logging

            _log = _logging.getLogger(__name__)
            _log.warning(
                "Audio-model voice analysis failed (%s: %s) — falling back to text-only. "
                "If this is a 404, the model '%s' may not be enabled on your OpenAI account.",
                type(e).__name__,
                e,
                audio_model,
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
        """Regenerate instructions incorporating user feedback."""
        context = context or {}
        source_language = context.get("source_language", "")
        target_language = context.get("target_language", "eng") or "eng"
        accent_intensity = context.get("accent_intensity", "strong")
        playbook = format_playbook_block(source_language)
        intensity = accent_intensity_guidance(accent_intensity)
        target_label = self._language_label(target_language)
        source_label = self._language_label(source_language) if source_language else "the speaker's L1"

        prompt = f"""
Refine reusable TTS speaker-profile instructions from user feedback.

Goal: English (or target-language) dubbing must sound like a native {source_label} speaker
performing {target_label} — authentic accent, concrete phonetics, lightly charming/funny when it fits.

Target language for TTS: **{target_label}**
{intensity}

{playbook}

Reference sample transcription (evidence of how they speak):
{original_text}

Target-language line (what TTS will say; may equal sample if not separately translated):
{translated_text}

Current instructions:
{previous_instructions}

User feedback:
{user_feedback}

Revise the instructions. Rules:
1. Keep accent authenticity HIGH unless feedback explicitly asks to reduce it.
2. Lead with the accent/phonetics paragraph (TTS models overweight the start).
3. Preserve speaker gender unless feedback changes it.
4. Incorporate the feedback concretely (not a vague restatement).
5. Stay reusable across the whole video — speaker identity + accent, not a script rewrite.
6. 120–280 words. No markdown.

Format your response as JSON:
{{
  "voice_instructions": "Revised instructions. MUST start with accent/phonetics.",
  "detected_characteristics": {{
    "tone": "...",
    "pace": "...",
    "energy": "...",
    "style": "...",
    "accent": "e.g. strong Japanese-accented English",
    "speaker_gender": "male | female | unknown"
  }},
  "improvements_made": "One sentence describing what changed"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_REGENERATE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.75,
                max_tokens=900,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            result = json.loads(content)
            normalized = self._normalize_result(result, llm_model=self.model)
            normalized["improvements_made"] = result.get("improvements_made", "")
            return normalized

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
        context: Dict[str, str],
    ) -> str:
        """Build the LLM prompt for voice instruction generation."""
        content_type = context.get("content_type", "general")
        speaker_gender = context.get("speaker_gender", "unknown")
        speaker_age = context.get("speaker_age", "adult")
        source_language = context.get("source_language", "")
        target_language = context.get("target_language", "eng") or "eng"
        accent_intensity = context.get("accent_intensity", "strong")

        source_label = self._language_label(source_language) if source_language else ""
        target_label = self._language_label(target_language)
        playbook = format_playbook_block(source_language)
        intensity = accent_intensity_guidance(accent_intensity)

        same_text = (original_text or "").strip() == (translated_text or "").strip()
        target_note = (
            "NOTE: The 'target line' text is identical to the sample transcription. "
            f"Treat the sample as evidence of the speaker's L1 delivery; write instructions "
            f"for performing **{target_label}** dialogue in their authentic accent — "
            "do not assume the TTS input is already native-sounding English."
            if same_text
            else f"The target line is the {target_label} text TTS will speak."
        )

        if source_language:
            accent_mandate = (
                f"The speaker's native language is **{source_label}** ({normalize_language_code(source_language) or source_language}). "
                f"Instructions MUST open with a dense accent paragraph describing how a native {source_label} speaker "
                f"sounds when speaking {target_label}: name concrete phonetic traits (vowels, consonants, clusters, "
                f"rhythm, intonation). Use the playbook below as a checklist to adapt — do not stay vague "
                f'("slight accent" is forbidden).'
            )
        else:
            accent_mandate = (
                f"Infer the speaker's likely native language from audio/text. Instructions MUST open with a dense "
                f"accent paragraph for how they sound speaking {target_label}, with concrete phonetic traits. "
                f'"Slight accent" without details is forbidden.'
            )

        return f"""
Write reusable TTS voice instructions so dubbed **{target_label}** dialogue sounds like THIS speaker
performing {target_label} — authentic L1 accent, natural quirks, lightly funny/charming when the delivery allows.

These instructions apply to every TTS segment in the video (speaker identity + accent profile).
You MAY use the sample as evidence of *how* they speak (rhythm, attitude, humour). Do NOT quote or
retell the sample's topical content as the instruction.

Content type: {content_type}
Hinted speaker: {speaker_gender}, {speaker_age}
Target language for TTS: **{target_label}**

{intensity}

{accent_mandate}

{playbook}

Sample transcription (evidence):
{original_text}

Target-language line:
{translated_text}

{target_note}

Also infer speaker gender from audio/text when possible.

Output JSON only:
{{
  "voice_instructions": "120-280 words. REQUIRED structure in this order: (1) Accent/phonetics paragraph FIRST, (2) Rhythm & pitch, (3) Attitude/energy including any light humour, (4) Do/Don't one-liners. No markdown headings — plain prose paragraphs are fine.",
  "detected_characteristics": {{
    "tone": "short phrase",
    "pace": "short phrase",
    "energy": "short phrase",
    "style": "short phrase",
    "accent": "e.g. strong Japanese-accented English",
    "speaker_gender": "male | female | unknown"
  }},
  "accent_phonetics": ["2-6 concrete phonetic/prosodic bullets adapted to this speaker"]
}}
"""

    @staticmethod
    def _language_label(code: str) -> str:
        labels = {
            "eng": "English",
            "en": "English",
            "jpn": "Japanese",
            "ja": "Japanese",
            "kor": "Korean",
            "ko": "Korean",
            "zho": "Mandarin Chinese",
            "zh": "Mandarin Chinese",
            "cmn": "Mandarin Chinese",
            "yue": "Cantonese",
            "vie": "Vietnamese",
            "vi": "Vietnamese",
            "tha": "Thai",
            "th": "Thai",
            "ind": "Indonesian",
            "id": "Indonesian",
            "msa": "Malay",
            "ms": "Malay",
            "rus": "Russian",
            "ru": "Russian",
            "spa": "Spanish",
            "es": "Spanish",
            "fra": "French",
            "fr": "French",
            "deu": "German",
            "de": "German",
        }
        key = (code or "").strip().lower()
        if key in labels:
            return labels[key]
        book_key = normalize_language_code(key)
        from app.services.accent_playbooks import ACCENT_PLAYBOOKS

        if book_key in ACCENT_PLAYBOOKS:
            return ACCENT_PLAYBOOKS[book_key]["name"]
        return code or "the target language"

    def _normalize_result(self, result: dict, llm_model: str) -> Dict:
        instructions = result.get(
            "voice_instructions", "Speak naturally with clear enunciation."
        )
        chars = result.get("detected_characteristics") or {}
        # Surface accent phonetics into characteristics for UI chips when present
        phonetics = result.get("accent_phonetics") or []
        if phonetics and not chars.get("accent"):
            chars = {**chars, "accent": "; ".join(str(p) for p in phonetics[:2])}
        return {
            "voice_instructions": instructions,
            "detected_characteristics": chars,
            "accent_phonetics": phonetics,
            "llm_model": llm_model,
        }

    def _fallback(self, error: str) -> Dict:
        return {
            "voice_instructions": (
                "Speak English with a clear non-native accent matching the speaker's native language: "
                "keep syllable-timed rhythm, concrete L1 consonant/vowel habits, and natural mid energy. "
                "Stay intelligible, lightly expressive, never a parody."
            ),
            "detected_characteristics": {
                "tone": "neutral",
                "pace": "moderate",
                "emotion": "balanced",
                "style": "natural",
                "accent": "non-native English",
            },
            "llm_model": self.model,
            "error": error,
        }


# Singleton instance
_generator_instance: Optional[VoiceInstructionGenerator] = None


def get_voice_instruction_generator() -> VoiceInstructionGenerator:
    """Get or create the singleton voice instruction generator.

    Recreates the instance when the resolved API key changes so keys
    saved via Settings (not only OPENAI_API_KEY env) are picked up.
    """
    global _generator_instance
    api_key = _resolve_openai_api_key()
    if _generator_instance is None or _generator_instance.api_key != api_key:
        _generator_instance = VoiceInstructionGenerator(api_key=api_key or None)
    return _generator_instance
