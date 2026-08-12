"""Language-specific accent playbooks for authentic L1→English TTS instructions.

Each playbook lists concrete phonetic and prosodic traits a native speaker of
that language typically carries into English. The LLM must adapt these to the
actual speaker rather than inventing vague "slight accent" wording.
"""

from __future__ import annotations

# Keys are ISO 639-2/B (and common aliases) as stored on projects.source_language_override.
ACCENT_PLAYBOOKS: dict[str, dict[str, str]] = {
    "jpn": {
        "name": "Japanese",
        "traits": (
            "syllable-timed rhythm (even beat, less stress-timing than native English); "
            "flatter, narrower pitch range with gentle rises at phrase ends; "
            "R/L blending (alveolar tap / light L); "
            "difficulty with consonant clusters — light vowel epenthesis (e.g. 'su-to-ress'); "
            "th → s/z or t/d; "
            "final consonants often softened or lightly released; "
            "vowel length contrasts (short vs long) transferred into English; "
            "slight 'katakana English' cadence — clear syllable boundaries, not a slurred native flow"
        ),
        "attitude": (
            "warm, polite, lightly playful energy is welcome when the delivery allows; "
            "authentic teaching/presenter charm — never a cartoon parody or mockery"
        ),
    },
    "kor": {
        "name": "Korean",
        "traits": (
            "syllable-timed to lightly stress-timed mix; "
            "tense vs lax consonant colouring (fortis feel on p/t/k); "
            "R/L approximated with a flap or clear L; "
            "f/v → p/b tendencies; "
            "th → s/t/d; "
            "final consonants clipped or unreleased; "
            "pitch often starts higher then steps down in short phrases; "
            "English stress sometimes placed evenly rather than on the native-English syllable"
        ),
        "attitude": (
            "clear, earnest, slightly bright delivery; light humour OK; "
            "keep it human and specific — not a generic 'Asian accent' caricature"
        ),
    },
    "zho": {
        "name": "Mandarin Chinese",
        "traits": (
            "syllable-timed rhythm; "
            "tone-language transfer: English stress replaced by pitch contours / level tones; "
            "final consonants often dropped or softened (-t/-d/-s); "
            "r-colouring reduced; "
            "th → s/z/d; "
            "v/w and l/n confusions possible depending on speaker; "
            "shorter vowels, crisp CV syllable shapes; "
            "question rises and emphasis done with pitch height more than lengthening"
        ),
        "attitude": (
            "direct, bright, sometimes playfully emphatic; "
            "authentic regional English flavour without mockery"
        ),
    },
    "cmn": {
        "name": "Mandarin Chinese",
        "traits": (
            "syllable-timed rhythm; "
            "tone-language transfer into English stress; "
            "final consonants softened; th → s/z/d; reduced r-colouring; "
            "crisp CV syllables, shorter vowels"
        ),
        "attitude": "direct and bright; lightly playful when appropriate",
    },
    "yue": {
        "name": "Cantonese",
        "traits": (
            "punchy syllable timing; "
            "final stops more audible than Mandarin; "
            "distinct pitch contours from tone transfer; "
            "th → f/d/t; "
            "r often approximated; "
            "energetic, clipped phrase endings"
        ),
        "attitude": "lively and expressive; humour welcome; avoid stereotype",
    },
    "vie": {
        "name": "Vietnamese",
        "traits": (
            "syllable-timed with strong tone-to-pitch transfer; "
            "final consonants often cut short; "
            "th/t distinctions flattened; "
            "vowel quality shifts; "
            "English stress realized as pitch/level more than length"
        ),
        "attitude": "bright, melodic, lightly playful when the line allows",
    },
    "tha": {
        "name": "Thai",
        "traits": (
            "syllable-timed; "
            "tone transfer into English intonation; "
            "final consonants softened; "
            "r/l variability; "
            "gentle, sing-song phrase shapes"
        ),
        "attitude": "softly expressive and warm; light smile in the voice",
    },
    "ind": {
        "name": "Indonesian",
        "traits": (
            "even syllable timing; "
            "clear vowels, less reduction of unstressed syllables; "
            "th → t/d; "
            "r tapped; "
            "friendly mid pitch with limited contrastive stress"
        ),
        "attitude": "friendly, open, lightly humorous when natural",
    },
    "msa": {
        "name": "Malay",
        "traits": (
            "even syllable timing; clear full vowels; "
            "th → t/d; tapped r; "
            "modest pitch range with friendly cadence"
        ),
        "attitude": "warm and approachable; light humour OK",
    },
    "rus": {
        "name": "Russian",
        "traits": (
            "harder consonants; "
            "darker vowels; "
            "less diphthongization; "
            "rolled/trilled or tapped r; "
            "th → s/z/t/d; "
            "stress-timing present but with Slavic consonant clusters; "
            "flatter or descending intonation in statements"
        ),
        "attitude": "dry wit welcome; composed, slightly formal unless audio says otherwise",
    },
}

# Map common aliases / ISO 639-1 / 639-3 codes onto playbook keys.
_LANGUAGE_ALIASES: dict[str, str] = {
    "ja": "jpn",
    "jp": "jpn",
    "japanese": "jpn",
    "ko": "kor",
    "kr": "kor",
    "korean": "kor",
    "zh": "zho",
    "zh-cn": "zho",
    "zh-tw": "zho",
    "chinese": "zho",
    "mandarin": "zho",
    "cantonese": "yue",
    "vi": "vie",
    "vietnamese": "vie",
    "th": "tha",
    "thai": "tha",
    "id": "ind",
    "indonesian": "ind",
    "ms": "msa",
    "malay": "msa",
    "ru": "rus",
    "russian": "rus",
}


def normalize_language_code(code: str) -> str:
    """Normalize a language code/name to a playbook key when possible."""
    raw = (code or "").strip().lower().replace("_", "-")
    if not raw:
        return ""
    if raw in ACCENT_PLAYBOOKS:
        return raw
    if raw in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[raw]
    # zh-Hans → zho
    base = raw.split("-", 1)[0]
    if base in ACCENT_PLAYBOOKS:
        return base
    if base in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[base]
    return raw


def get_accent_playbook(source_language: str) -> dict[str, str] | None:
    """Return the accent playbook for a source language, or None."""
    key = normalize_language_code(source_language)
    if key in ACCENT_PLAYBOOKS:
        return {"code": key, **ACCENT_PLAYBOOKS[key]}
    return None


def format_playbook_block(source_language: str) -> str:
    """Format a playbook as prompt text, or empty string if unknown."""
    book = get_accent_playbook(source_language)
    if not book:
        return ""
    return (
        f"ACCENT PLAYBOOK for native {book['name']} speakers speaking English "
        f"(adapt to THIS speaker from the audio/text — do not copy verbatim):\n"
        f"- Phonetics / rhythm: {book['traits']}\n"
        f"- Attitude: {book['attitude']}\n"
    )


def accent_intensity_guidance(intensity: str) -> str:
    """Human-readable guidance for accent intensity dial."""
    level = (intensity or "strong").strip().lower()
    if level in {"subtle", "light", "mild"}:
        return (
            "Accent intensity: SUBTLE — keep clear intelligibility; "
            "native-English listeners notice a light L1 flavour, not a heavy accent."
        )
    if level in {"moderate", "medium"}:
        return (
            "Accent intensity: MODERATE — clearly non-native English with identifiable L1 traits, "
            "still easy to understand."
        )
    if level in {"comedic", "funny", "playful"}:
        return (
            "Accent intensity: PLAYFUL/STRONG — lean into distinctive L1→English traits and "
            "a lightly humorous delivery when it fits; never cruel parody or slur caricature."
        )
    # default strong
    return (
        "Accent intensity: STRONG — unmistakably a native L1 speaker performing English, "
        "with concrete phonetic traits listed first; lightly funny/charming when the delivery allows, "
        "never a mean stereotype."
    )
