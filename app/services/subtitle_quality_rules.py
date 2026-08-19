"""Subtitle quality rules for the review screen.

Each heuristic is registered as a rule. Running the engine returns breaches
that can be grouped per cue to show how many rules a subtitle line violated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.schemas.subtitle_review import (
    SubtitleQualityBreach,
    SubtitleQualityRule,
)
from stt_hallucination import (
    MIN_CONSECUTIVE_DUPLICATE_SEGMENTS,
    MIN_PHRASE_REPEAT_COUNT,
    _check_character_spam,
    _check_dominant_word,
    _check_intra_cue_numbered_list,
    _check_known_phrases,
    _check_near_duplicate_run,
    _check_numbered_enumeration_loop,
    _check_phrase_spam_in_cue,
    _check_shared_phrase_run,
    _check_segment_density,
    _check_transcript_density,
    _find_repeated_phrase,
    _normalize_text,
)

RuleScope = Literal["cue", "file"]

# Registry of all subtitle review quality rules.
SUBTITLE_QUALITY_RULES: tuple[SubtitleQualityRule, ...] = (
    SubtitleQualityRule(
        id="known_hallucination_phrase",
        label="Known STT phrase",
        scope="cue",
    ),
    SubtitleQualityRule(
        id="intra_cue_numbered_list",
        label="Numbered list in cue",
        scope="cue",
    ),
    SubtitleQualityRule(
        id="phrase_spam_in_cue",
        label="Repeated phrase in cue",
        scope="cue",
    ),
    SubtitleQualityRule(
        id="character_spam",
        label="Character spam",
        scope="cue",
    ),
    SubtitleQualityRule(
        id="excessive_cps",
        label="Text too dense",
        scope="cue",
    ),
    SubtitleQualityRule(
        id="repeated_phrase_loop",
        label="Phrase loop in cue",
        scope="cue",
    ),
    SubtitleQualityRule(
        id="consecutive_duplicate_segments",
        label="Consecutive duplicates",
        scope="cue",
    ),
    SubtitleQualityRule(
        id="numbered_enumeration_loop",
        label="Numbered line sequence",
        scope="cue",
    ),
    SubtitleQualityRule(
        id="near_duplicate_run",
        label="Near-duplicate lines",
        scope="cue",
    ),
    SubtitleQualityRule(
        id="shared_phrase_run",
        label="Shared phrase run",
        scope="cue",
    ),
    SubtitleQualityRule(
        id="dominant_word_loop",
        label="Dominant word loop",
        scope="file",
    ),
    SubtitleQualityRule(
        id="transcript_too_dense",
        label="Transcript too dense",
        scope="file",
    ),
)

_RULES_BY_ID = {rule.id: rule for rule in SUBTITLE_QUALITY_RULES}


@dataclass(frozen=True)
class SubtitleQualityAnalysis:
    """Result of running all subtitle quality rules."""

    rules: tuple[SubtitleQualityRule, ...]
    breaches: tuple[SubtitleQualityBreach, ...]


def _segments_from_cues(
    cues: list[tuple[float, float, str]],
) -> list | None:
    try:
        from openai.types.audio.transcription_segment import TranscriptionSegment
    except ImportError:
        return None

    return [
        TranscriptionSegment(
            id=index,
            seek=0,
            start=start,
            end=end,
            text=text,
            tokens=[],
            temperature=0.0,
            avg_logprob=-0.3,
            compression_ratio=1.0,
            no_speech_prob=0.0,
        )
        for index, (start, end, text) in enumerate(cues)
    ]


def _breach(
    rule_id: str,
    message: str,
    segment_index: int | None,
) -> SubtitleQualityBreach:
    return SubtitleQualityBreach(
        rule_id=rule_id,
        message=message,
        segment_index=segment_index,
    )


def _duplicate_run_breaches(segments: list) -> list[SubtitleQualityBreach]:
    if len(segments) < MIN_CONSECUTIVE_DUPLICATE_SEGMENTS:
        return []

    breaches: list[SubtitleQualityBreach] = []
    run_text = ""
    run_len = 0
    run_indices: list[int] = []

    def flush_run() -> None:
        nonlocal run_len, run_text, run_indices
        if run_len >= MIN_CONSECUTIVE_DUPLICATE_SEGMENTS and run_text:
            preview = run_text[:60] + ("…" if len(run_text) > 60 else "")
            message = (
                f"{run_len} consecutive segments repeat the same text: {preview!r}"
            )
            for index in run_indices:
                breaches.append(
                    _breach("consecutive_duplicate_segments", message, index)
                )
        run_len = 0
        run_text = ""
        run_indices = []

    for index, segment in enumerate(segments):
        normalized = _normalize_text(segment.text or "")
        if not normalized:
            flush_run()
            continue
        if normalized == run_text:
            run_len += 1
            run_indices.append(index)
        else:
            flush_run()
            run_text = normalized
            run_len = 1
            run_indices = [index]
    flush_run()
    return breaches


def _findings_to_breaches(findings: list, rule_id: str | None = None) -> list[SubtitleQualityBreach]:
    breaches: list[SubtitleQualityBreach] = []
    for finding in findings:
        breaches.append(
            _breach(
                rule_id or finding.code,
                finding.message,
                finding.segment_index,
            )
        )
    return breaches


def analyze_subtitle_quality(
    cues: list[tuple[float, float, str]],
) -> SubtitleQualityAnalysis:
    """Run every registered rule and return all breaches."""
    if not cues:
        return SubtitleQualityAnalysis(
            rules=SUBTITLE_QUALITY_RULES,
            breaches=(),
        )

    segments = _segments_from_cues(cues)
    if segments is None:
        return SubtitleQualityAnalysis(
            rules=SUBTITLE_QUALITY_RULES,
            breaches=(),
        )

    audio_duration = max(end for _, end, _ in cues)
    breaches: list[SubtitleQualityBreach] = []

    for index, segment in enumerate(segments):
        text = segment.text or ""
        breaches.extend(
            _findings_to_breaches(_check_known_phrases(text, index, None))
        )
        breaches.extend(
            _findings_to_breaches(_check_intra_cue_numbered_list(text, index, None))
        )
        breaches.extend(
            _findings_to_breaches(_check_phrase_spam_in_cue(text, index, None))
        )
        breaches.extend(
            _findings_to_breaches(_check_character_spam(text, index, None))
        )
        breaches.extend(
            _findings_to_breaches(_check_segment_density(segment, index, None))
        )

        repeated = _find_repeated_phrase(text)
        if repeated:
            breaches.append(
                _breach(
                    "repeated_phrase_loop",
                    (
                        f"phrase {repeated!r} repeats {MIN_PHRASE_REPEAT_COUNT}+ "
                        f"times in this cue"
                    ),
                    index,
                )
            )

    breaches.extend(_duplicate_run_breaches(segments))
    breaches.extend(
        _findings_to_breaches(_check_numbered_enumeration_loop(segments, None))
    )
    breaches.extend(_findings_to_breaches(_check_near_duplicate_run(segments, None)))
    breaches.extend(_findings_to_breaches(_check_shared_phrase_run(segments, None)))
    breaches.extend(_findings_to_breaches(_check_dominant_word(segments, None)))
    breaches.extend(
        _findings_to_breaches(
            _check_transcript_density(segments, audio_duration, None)
        )
    )

    deduped: list[SubtitleQualityBreach] = []
    seen: set[tuple[str, int | None, str]] = set()
    for breach in breaches:
        if breach.rule_id not in _RULES_BY_ID:
            continue
        key = (breach.rule_id, breach.segment_index, breach.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(breach)

    return SubtitleQualityAnalysis(
        rules=SUBTITLE_QUALITY_RULES,
        breaches=tuple(deduped),
    )


def breaches_for_segment(
    breaches: tuple[SubtitleQualityBreach, ...] | list[SubtitleQualityBreach],
    segment_index: int,
) -> list[SubtitleQualityBreach]:
    return [b for b in breaches if b.segment_index == segment_index]


def unique_rule_ids(breaches: list[SubtitleQualityBreach]) -> list[str]:
    return sorted({breach.rule_id for breach in breaches})
