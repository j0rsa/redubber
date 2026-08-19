"""Detect likely STT hallucinations before TTS and downstream pipeline stages.

Whisper and other STT models often emit repetitive loops, boilerplate phrases on
silence, or impossibly dense text. This module flags those patterns so redubbing
can fail fast at transcription instead of burning translation/TTS cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from openai.types.audio.transcription_segment import TranscriptionSegment

# Whisper-style segment metrics (see OpenAI verbose_json transcription objects)
MAX_COMPRESSION_RATIO = 2.4
MIN_AVG_LOGPROB = -1.0
MAX_NO_SPEECH_WITH_TEXT = 0.55

# Text density / repetition
MAX_CHARS_PER_SECOND = 40.0
MIN_CONSECUTIVE_DUPLICATE_SEGMENTS = 3
MIN_PHRASE_REPEAT_COUNT = 3
MIN_PHRASE_WORDS = 2
MAX_PHRASE_WORDS = 10
DOMINANT_WORD_RATIO = 0.38
DOMINANT_WORD_MIN_TOKENS = 24
CHAR_RUN_MIN = 8
MIN_NEAR_DUPLICATE_SIMILARITY = 0.45
MIN_SHARED_PHRASE_RUN = 5
MIN_SHARED_PHRASE_WORDS = 3
_NUMBERED_LINE_RE = re.compile(r"^\d+\.\s*")

# Common silence / tail hallucinations (case-insensitive substring match)
KNOWN_HALLUCINATION_PHRASES: tuple[str, ...] = (
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "subtitles by",
    "subtitles by the amara",
    "amara.org",
    "transcribed by",
    "copyright",
    "all rights reserved",
    "mbc news",
    "bbc news",
    "continued on",
    "to be continued",
    "продолжение следует",
    "请不吝点赞",
    "字幕",
    "thanks for listening",
    "see you next time",
    "don't forget to subscribe",
)

PUNCTUATION = ".,?!:;()[]{}'\"-"


class STTHallucinationError(Exception):
    """Raised when transcription output looks untrustworthy."""

    def __init__(self, report: HallucinationReport) -> None:
        self.report = report
        super().__init__(report.summary())


@dataclass(frozen=True)
class HallucinationFinding:
    """One detected quality problem in STT output."""

    code: str
    message: str
    severity: Literal["error"] = "error"
    segment_index: int | None = None
    chunk_label: str | None = None


@dataclass
class HallucinationReport:
    """Aggregated STT quality analysis."""

    findings: list[HallucinationFinding] = field(default_factory=list)
    segment_count: int = 0
    total_duration: float = 0.0
    source_label: str = ""

    @property
    def passed(self) -> bool:
        return len(self.findings) == 0

    def summary(self) -> str:
        if self.passed:
            return "STT quality check passed"
        header = "STT quality check failed"
        if self.source_label:
            header += f" for {self.source_label}"
        lines = [header + ":"]
        for item in self.findings[:8]:
            loc = ""
            if item.chunk_label:
                loc = f" [{item.chunk_label}]"
            elif item.segment_index is not None:
                loc = f" [segment {item.segment_index}]"
            lines.append(f"- {item.code}{loc}: {item.message}")
        if len(self.findings) > 8:
            lines.append(f"- … and {len(self.findings) - 8} more issue(s)")
        return "\n".join(lines)


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    for ch in PUNCTUATION:
        lowered = lowered.replace(ch, " ")
    return " ".join(lowered.split())


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return [tok for tok in normalized.split() if tok]


def _strip_leading_number(text: str) -> str:
    return _NUMBERED_LINE_RE.sub("", text.strip(), count=1)


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return overlap / union


def _segments_similar(left: str, right: str) -> bool:
    left_body = _strip_leading_number(left)
    right_body = _strip_leading_number(right)
    return _token_jaccard(left_body, right_body) >= MIN_NEAR_DUPLICATE_SIMILARITY


def _contains_word_sequence(text: str, phrase: list[str]) -> bool:
    words = _tokenize(_strip_leading_number(text))
    if len(phrase) > len(words):
        return False
    for start in range(len(words) - len(phrase) + 1):
        if words[start : start + len(phrase)] == phrase:
            return True
    return False


def _find_longest_shared_phrase(
    texts: list[str],
    min_words: int,
) -> list[str] | None:
    if not texts:
        return None

    best: list[str] | None = None
    seen: set[tuple[str, ...]] = set()
    for text in texts:
        words = _tokenize(_strip_leading_number(text))
        max_size = min(len(words), MAX_PHRASE_WORDS)
        for size in range(max_size, min_words - 1, -1):
            for start in range(len(words) - size + 1):
                phrase = tuple(words[start : start + size])
                if phrase in seen:
                    continue
                seen.add(phrase)
                if all(_contains_word_sequence(item, list(phrase)) for item in texts):
                    if best is None or len(phrase) > len(best):
                        best = list(phrase)
    return best


def _check_shared_phrase_run(
    segments: list[TranscriptionSegment], chunk_label: str | None
) -> list[HallucinationFinding]:
    """Flag long runs of lines that share the same core phrase (template loops)."""
    findings: list[HallucinationFinding] = []
    index = 0
    total = len(segments)

    while index < total:
        run = [index]
        next_index = index + 1
        while next_index < total:
            texts = [(segments[i].text or "") for i in run + [next_index]]
            if _find_longest_shared_phrase(texts, MIN_SHARED_PHRASE_WORDS):
                run.append(next_index)
                next_index += 1
            else:
                break

        if len(run) >= MIN_SHARED_PHRASE_RUN:
            texts = [(segments[i].text or "") for i in run]
            phrase = _find_longest_shared_phrase(texts, MIN_SHARED_PHRASE_WORDS)
            phrase_text = " ".join(phrase) if phrase else "…"
            count = len(run)
            message = (
                f"{count} consecutive lines share repeated wording "
                f"({phrase_text!r}) — likely STT template hallucination"
            )
            for run_index in run:
                findings.append(
                    HallucinationFinding(
                        code="shared_phrase_run",
                        message=message,
                        segment_index=run_index,
                        chunk_label=chunk_label,
                    )
                )
            index = run[-1] + 1
        else:
            index += 1

    return findings


def _check_numbered_enumeration_loop(
    segments: list[TranscriptionSegment], chunk_label: str | None
) -> list[HallucinationFinding]:
    """Flag runs of consecutive lines like ``4. …``, ``5. …`` (STT list loops)."""
    findings: list[HallucinationFinding] = []
    run_indices: list[int] = []

    def flush_run() -> None:
        if len(run_indices) < MIN_CONSECUTIVE_DUPLICATE_SEGMENTS:
            return
        count = len(run_indices)
        message = (
            f"{count} consecutive numbered lines — likely STT enumeration hallucination"
        )
        for index in run_indices:
            findings.append(
                HallucinationFinding(
                    code="numbered_enumeration_loop",
                    message=message,
                    segment_index=index,
                    chunk_label=chunk_label,
                )
            )

    for index, segment in enumerate(segments):
        text = (segment.text or "").strip()
        if _NUMBERED_LINE_RE.match(text):
            run_indices.append(index)
        else:
            flush_run()
            run_indices = []
    flush_run()
    return findings


def _check_near_duplicate_run(
    segments: list[TranscriptionSegment], chunk_label: str | None
) -> list[HallucinationFinding]:
    """Flag consecutive segments with very similar wording (not exact duplicates)."""
    if len(segments) < MIN_CONSECUTIVE_DUPLICATE_SEGMENTS:
        return []

    findings: list[HallucinationFinding] = []
    run_indices = [0]

    def flush_run() -> None:
        if len(run_indices) < MIN_CONSECUTIVE_DUPLICATE_SEGMENTS:
            return
        preview = _strip_leading_number(segments[run_indices[0]].text or "")
        preview = preview[:60] + ("…" if len(preview) > 60 else "")
        count = len(run_indices)
        message = (
            f"{count} consecutive near-duplicate lines with similar wording: {preview!r}"
        )
        for index in run_indices:
            findings.append(
                HallucinationFinding(
                    code="near_duplicate_run",
                    message=message,
                    segment_index=index,
                    chunk_label=chunk_label,
                )
            )

    for index in range(1, len(segments)):
        previous = segments[index - 1].text or ""
        current = segments[index].text or ""
        if _segments_similar(previous, current):
            run_indices.append(index)
        else:
            flush_run()
            run_indices = [index]
    flush_run()
    return findings


def _chars_per_second(start: float, end: float, text: str) -> float:
    duration = max(end - start, 0.001)
    compact = _normalize_text(text).replace(" ", "")
    return len(compact) / duration


def _has_whisper_metrics(segment: TranscriptionSegment) -> bool:
    return (
        getattr(segment, "compression_ratio", None) is not None
        or getattr(segment, "avg_logprob", None) is not None
        or getattr(segment, "no_speech_prob", None) is not None
    )


def _check_whisper_metrics(
    segment: TranscriptionSegment, index: int, chunk_label: str | None
) -> list[HallucinationFinding]:
    if not _has_whisper_metrics(segment):
        return []

    findings: list[HallucinationFinding] = []
    text = (segment.text or "").strip()
    compression = float(getattr(segment, "compression_ratio", 0.0) or 0.0)
    avg_logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
    no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)

    if compression > MAX_COMPRESSION_RATIO:
        findings.append(
            HallucinationFinding(
                code="high_compression_ratio",
                message=(
                    f"compression_ratio={compression:.2f} "
                    f"(>{MAX_COMPRESSION_RATIO}) — typical Whisper hallucination signal"
                ),
                segment_index=index,
                chunk_label=chunk_label,
            )
        )
    if text and avg_logprob < MIN_AVG_LOGPROB:
        findings.append(
            HallucinationFinding(
                code="low_avg_logprob",
                message=(
                    f"avg_logprob={avg_logprob:.2f} "
                    f"(<{MIN_AVG_LOGPROB}) — low-confidence transcription"
                ),
                segment_index=index,
                chunk_label=chunk_label,
            )
        )
    if text and no_speech > MAX_NO_SPEECH_WITH_TEXT:
        findings.append(
            HallucinationFinding(
                code="speech_on_silence",
                message=(
                    f"no_speech_prob={no_speech:.2f} with non-empty text "
                    f"(>{MAX_NO_SPEECH_WITH_TEXT})"
                ),
                segment_index=index,
                chunk_label=chunk_label,
            )
        )
    return findings


def _check_segment_density(
    segment: TranscriptionSegment, index: int, chunk_label: str | None
) -> list[HallucinationFinding]:
    text = (segment.text or "").strip()
    if not text:
        return []
    cps = _chars_per_second(segment.start, segment.end, text)
    if cps > MAX_CHARS_PER_SECOND:
        preview = text[:80] + ("…" if len(text) > 80 else "")
        return [
            HallucinationFinding(
                code="excessive_cps",
                message=(
                    f"{cps:.1f} chars/s (>{MAX_CHARS_PER_SECOND}) — "
                    f"text too dense for duration: {preview!r}"
                ),
                segment_index=index,
                chunk_label=chunk_label,
            )
        ]
    return []


def _check_character_spam(text: str, index: int, chunk_label: str | None) -> list[HallucinationFinding]:
    if re.search(rf"(.)\1{{{CHAR_RUN_MIN - 1},}}", text):
        return [
            HallucinationFinding(
                code="character_spam",
                message="repeated single-character run detected",
                segment_index=index,
                chunk_label=chunk_label,
            )
        ]
    return []


def _check_known_phrases(text: str, index: int | None, chunk_label: str | None) -> list[HallucinationFinding]:
    lowered = text.lower()
    hits = [phrase for phrase in KNOWN_HALLUCINATION_PHRASES if phrase in lowered]
    if not hits:
        return []
    shown = ", ".join(repr(h) for h in hits[:3])
    return [
        HallucinationFinding(
            code="known_hallucination_phrase",
            message=f"contains common STT hallucination phrase(s): {shown}",
            segment_index=index,
            chunk_label=chunk_label,
        )
    ]


def _check_consecutive_duplicate_segments(
    segments: list[TranscriptionSegment], chunk_label: str | None
) -> list[HallucinationFinding]:
    if len(segments) < MIN_CONSECUTIVE_DUPLICATE_SEGMENTS:
        return []

    run_text = ""
    run_start = 0
    run_len = 0
    findings: list[HallucinationFinding] = []

    def flush_run() -> None:
        nonlocal run_len, run_text, run_start
        if run_len >= MIN_CONSECUTIVE_DUPLICATE_SEGMENTS and run_text:
            preview = run_text[:60] + ("…" if len(run_text) > 60 else "")
            findings.append(
                HallucinationFinding(
                    code="consecutive_duplicate_segments",
                    message=(
                        f"{run_len} consecutive segments repeat the same text: {preview!r}"
                    ),
                    segment_index=run_start,
                    chunk_label=chunk_label,
                )
            )
        run_len = 0
        run_text = ""

    for index, segment in enumerate(segments):
        normalized = _normalize_text(segment.text or "")
        if not normalized:
            flush_run()
            continue
        if normalized == run_text:
            run_len += 1
        else:
            flush_run()
            run_text = normalized
            run_start = index
            run_len = 1
    flush_run()
    return findings


def _find_repeated_phrase(text: str) -> str | None:
    words = _tokenize(text)
    if len(words) < MIN_PHRASE_WORDS * MIN_PHRASE_REPEAT_COUNT:
        return None

    max_n = min(MAX_PHRASE_WORDS, len(words) // MIN_PHRASE_REPEAT_COUNT)
    for size in range(max_n, MIN_PHRASE_WORDS - 1, -1):
        for start in range(0, len(words) - size * MIN_PHRASE_REPEAT_COUNT + 1):
            phrase = words[start : start + size]
            repeats = 1
            pos = start + size
            while pos + size <= len(words) and words[pos : pos + size] == phrase:
                repeats += 1
                pos += size
            if repeats >= MIN_PHRASE_REPEAT_COUNT:
                return " ".join(phrase)
    return None


def _check_phrase_loops(
    segments: list[TranscriptionSegment], chunk_label: str | None
) -> list[HallucinationFinding]:
    combined = " ".join(_normalize_text(seg.text or "") for seg in segments if seg.text)
    if not combined:
        return []

    repeated = _find_repeated_phrase(combined)
    if repeated:
        return [
            HallucinationFinding(
                code="repeated_phrase_loop",
                message=f"phrase {repeated!r} repeats {MIN_PHRASE_REPEAT_COUNT}+ times in sequence",
                chunk_label=chunk_label,
            )
        ]
    return []


def _check_dominant_word(
    segments: list[TranscriptionSegment], chunk_label: str | None
) -> list[HallucinationFinding]:
    tokens = _tokenize(" ".join(seg.text or "" for seg in segments))
    if len(tokens) < DOMINANT_WORD_MIN_TOKENS:
        return []

    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    word, count = max(counts.items(), key=lambda item: item[1])
    ratio = count / len(tokens)
    if ratio >= DOMINANT_WORD_RATIO:
        return [
            HallucinationFinding(
                code="dominant_word_loop",
                message=(
                    f"word {word!r} is {ratio:.0%} of transcript "
                    f"({count}/{len(tokens)} tokens)"
                ),
                chunk_label=chunk_label,
            )
        ]
    return []


def _check_transcript_density(
    segments: list[TranscriptionSegment],
    audio_duration: float | None,
    chunk_label: str | None,
) -> list[HallucinationFinding]:
    if not segments:
        return [
            HallucinationFinding(
                code="empty_transcript",
                message="no speech segments after transcription",
                chunk_label=chunk_label,
            )
        ]

    total_duration = audio_duration
    if total_duration is None:
        total_duration = max((seg.end for seg in segments), default=0.0) - min(
            (seg.start for seg in segments), default=0.0
        )
    if total_duration <= 0:
        return []

    combined = "".join(_normalize_text(seg.text or "") for seg in segments)
    cps = len(combined.replace(" ", "")) / max(total_duration, 0.001)
    if cps > MAX_CHARS_PER_SECOND:
        return [
            HallucinationFinding(
                code="transcript_too_dense",
                message=(
                    f"overall {cps:.1f} chars/s (>{MAX_CHARS_PER_SECOND}) "
                    f"for {total_duration:.1f}s of audio"
                ),
                chunk_label=chunk_label,
            )
        ]
    return []


def analyze_segments(
    segments: list[TranscriptionSegment],
    *,
    audio_duration: float | None = None,
    source_label: str = "",
    chunk_label: str | None = None,
) -> HallucinationReport:
    """Run all hallucination heuristics on ``segments``."""
    report = HallucinationReport(
        segment_count=len(segments),
        total_duration=audio_duration or 0.0,
        source_label=source_label,
    )

    for index, segment in enumerate(segments):
        text = segment.text or ""
        report.findings.extend(
            _check_whisper_metrics(segment, index, chunk_label)
        )
        report.findings.extend(
            _check_segment_density(segment, index, chunk_label)
        )
        report.findings.extend(
            _check_character_spam(text, index, chunk_label)
        )
        report.findings.extend(
            _check_known_phrases(text, index, chunk_label)
        )

    report.findings.extend(_check_consecutive_duplicate_segments(segments, chunk_label))
    report.findings.extend(_check_numbered_enumeration_loop(segments, chunk_label))
    report.findings.extend(_check_near_duplicate_run(segments, chunk_label))
    report.findings.extend(_check_shared_phrase_run(segments, chunk_label))
    report.findings.extend(_check_phrase_loops(segments, chunk_label))
    report.findings.extend(_check_dominant_word(segments, chunk_label))
    report.findings.extend(
        _check_transcript_density(segments, audio_duration, chunk_label)
    )

    return report


def assert_segments_acceptable(
    segments: list[TranscriptionSegment],
    *,
    audio_duration: float | None = None,
    source_label: str = "",
    chunk_label: str | None = None,
) -> None:
    """Raise ``STTHallucinationError`` when transcription looks hallucinated."""
    report = analyze_segments(
        segments,
        audio_duration=audio_duration,
        source_label=source_label,
        chunk_label=chunk_label,
    )
    if not report.passed:
        raise STTHallucinationError(report)
