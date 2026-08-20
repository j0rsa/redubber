"""Detect likely STT hallucinations before TTS and downstream pipeline stages.

Whisper and other STT models often emit repetitive loops, boilerplate phrases on
silence, or impossibly dense text. This module flags those patterns so redubbing
can fail fast at transcription instead of burning translation/TTS cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Mapping

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
MIN_SHARED_PHRASE_RUN = 3
MIN_SHARED_PHRASE_WORDS = 2
MIN_INTRA_CUE_NUMBERED_ITEMS = 3
MIN_PHRASE_SPAM_COUNT = 4
_NUMBERED_LINE_RE = re.compile(r"^\d+\.\s*")
_INTRA_NUMBERED_MARKER_RE = re.compile(r"\b\d+\.\s+")
INSIGNIFICANT_SHARED_PHRASES = frozenset(
    {
        "thank you",
        "in the",
        "of the",
        "to the",
        "and the",
        "for the",
        "on the",
        "at the",
        "is a",
        "was a",
        "it is",
        "i am",
        "you are",
        "we are",
    }
)

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

RuleComparison = Literal["gt", "lt", "gte", "min_count"]


@dataclass(frozen=True)
class HallucinationRuleSpec:
    """Catalog entry for one hallucination heuristic (defaults + UI metadata)."""

    id: str
    label: str
    description: str
    default_threshold: float | None = None
    threshold_min: float | None = None
    threshold_max: float | None = None
    threshold_step: float | None = None
    unit: str | None = None
    comparison: RuleComparison | None = None


HALLUCINATION_RULE_SPECS: tuple[HallucinationRuleSpec, ...] = (
    HallucinationRuleSpec(
        id="known_hallucination_phrase",
        label="Known STT phrase",
        description="Flag cues that contain common Whisper boilerplate such as “thanks for watching”.",
    ),
    HallucinationRuleSpec(
        id="intra_cue_numbered_list",
        label="Numbered list in cue",
        description="Flag a single cue that packs many numbered menu-style items.",
        default_threshold=float(MIN_INTRA_CUE_NUMBERED_ITEMS),
        threshold_min=2,
        threshold_max=20,
        threshold_step=1,
        unit="items",
        comparison="min_count",
    ),
    HallucinationRuleSpec(
        id="phrase_spam_in_cue",
        label="Repeated phrase in cue",
        description="Flag a cue where the same two-word phrase repeats many times.",
        default_threshold=float(MIN_PHRASE_SPAM_COUNT),
        threshold_min=2,
        threshold_max=20,
        threshold_step=1,
        unit="repeats",
        comparison="min_count",
    ),
    HallucinationRuleSpec(
        id="character_spam",
        label="Character spam",
        description="Flag a run of the same character (aaaaaaa…) typical of STT loops.",
        default_threshold=float(CHAR_RUN_MIN),
        threshold_min=3,
        threshold_max=50,
        threshold_step=1,
        unit="chars",
        comparison="min_count",
    ),
    HallucinationRuleSpec(
        id="excessive_cps",
        label="Text too dense",
        description="Flag a cue whose text is too dense for its duration.",
        default_threshold=MAX_CHARS_PER_SECOND,
        threshold_min=5,
        threshold_max=200,
        threshold_step=1,
        unit="chars/s",
        comparison="gt",
    ),
    HallucinationRuleSpec(
        id="repeated_phrase_loop",
        label="Phrase loop",
        description="Flag a phrase that repeats back-to-back in sequence.",
        default_threshold=float(MIN_PHRASE_REPEAT_COUNT),
        threshold_min=2,
        threshold_max=20,
        threshold_step=1,
        unit="repeats",
        comparison="min_count",
    ),
    HallucinationRuleSpec(
        id="consecutive_duplicate_segments",
        label="Consecutive duplicates",
        description="Flag consecutive cues that repeat the exact same text.",
        default_threshold=float(MIN_CONSECUTIVE_DUPLICATE_SEGMENTS),
        threshold_min=2,
        threshold_max=20,
        threshold_step=1,
        unit="cues",
        comparison="min_count",
    ),
    HallucinationRuleSpec(
        id="numbered_enumeration_loop",
        label="Numbered line sequence",
        description="Flag a run of consecutive numbered lines (4. …, 5. …).",
        default_threshold=float(MIN_CONSECUTIVE_DUPLICATE_SEGMENTS),
        threshold_min=2,
        threshold_max=20,
        threshold_step=1,
        unit="lines",
        comparison="min_count",
    ),
    HallucinationRuleSpec(
        id="near_duplicate_run",
        label="Near-duplicate lines",
        description="Flag consecutive cues with very similar wording (not exact copies).",
        default_threshold=float(MIN_CONSECUTIVE_DUPLICATE_SEGMENTS),
        threshold_min=2,
        threshold_max=20,
        threshold_step=1,
        unit="cues",
        comparison="min_count",
    ),
    HallucinationRuleSpec(
        id="shared_phrase_run",
        label="Shared phrase run",
        description="Flag consecutive cues that share the same core phrase (template loops).",
        default_threshold=float(MIN_SHARED_PHRASE_RUN),
        threshold_min=2,
        threshold_max=20,
        threshold_step=1,
        unit="cues",
        comparison="min_count",
    ),
    HallucinationRuleSpec(
        id="dominant_word_loop",
        label="Dominant word loop",
        description="Flag a transcript where one word makes up too large a share of all tokens.",
        default_threshold=DOMINANT_WORD_RATIO,
        threshold_min=0.1,
        threshold_max=1.0,
        threshold_step=0.01,
        unit="ratio",
        comparison="gte",
    ),
    HallucinationRuleSpec(
        id="transcript_too_dense",
        label="Transcript too dense",
        description="Flag a whole file whose overall characters-per-second is impossibly high.",
        default_threshold=MAX_CHARS_PER_SECOND,
        threshold_min=5,
        threshold_max=200,
        threshold_step=1,
        unit="chars/s",
        comparison="gt",
    ),
    HallucinationRuleSpec(
        id="high_compression_ratio",
        label="High compression ratio",
        description="Whisper metric: high compression_ratio is a typical hallucination signal.",
        default_threshold=MAX_COMPRESSION_RATIO,
        threshold_min=1.0,
        threshold_max=10.0,
        threshold_step=0.1,
        unit="ratio",
        comparison="gt",
    ),
    HallucinationRuleSpec(
        id="low_avg_logprob",
        label="Low average log-probability",
        description="Whisper metric: very low avg_logprob means the model is not confident.",
        default_threshold=MIN_AVG_LOGPROB,
        threshold_min=-5.0,
        threshold_max=0.0,
        threshold_step=0.05,
        unit="logprob",
        comparison="lt",
    ),
    HallucinationRuleSpec(
        id="speech_on_silence",
        label="Speech on silence",
        description="Whisper metric: non-empty text with a high no_speech_prob.",
        default_threshold=MAX_NO_SPEECH_WITH_TEXT,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_step=0.05,
        unit="probability",
        comparison="gt",
    ),
    HallucinationRuleSpec(
        id="empty_transcript",
        label="Empty transcript",
        description="Flag transcription that produced no speech segments at all.",
    ),
)

HALLUCINATION_RULE_SPECS_BY_ID: dict[str, HallucinationRuleSpec] = {
    spec.id: spec for spec in HALLUCINATION_RULE_SPECS
}


@dataclass(frozen=True)
class HallucinationRuleState:
    """Persisted enable flag and threshold for one rule."""

    enabled: bool = True
    threshold: float | None = None


@dataclass(frozen=True)
class HallucinationConfig:
    """Runtime hallucination-detector settings, usually loaded from the database."""

    rules: Mapping[str, HallucinationRuleState] = field(default_factory=dict)

    def is_enabled(self, rule_id: str) -> bool:
        state = self.rules.get(rule_id)
        return True if state is None else state.enabled

    def threshold(self, rule_id: str) -> float:
        spec = HALLUCINATION_RULE_SPECS_BY_ID[rule_id]
        if spec.default_threshold is None:
            raise ValueError(f"Rule {rule_id} has no numeric threshold")
        state = self.rules.get(rule_id)
        if state is None or state.threshold is None:
            return spec.default_threshold
        return float(state.threshold)

    def int_threshold(self, rule_id: str) -> int:
        return int(round(self.threshold(rule_id)))


def default_hallucination_config() -> HallucinationConfig:
    """Return the hardcoded factory defaults (used before DB rows exist)."""
    return HallucinationConfig(
        rules={
            spec.id: HallucinationRuleState(
                enabled=True, threshold=spec.default_threshold
            )
            for spec in HALLUCINATION_RULE_SPECS
        }
    )


DEFAULT_HALLUCINATION_CONFIG = default_hallucination_config()


def hallucination_config_from_rows(rows: list[dict]) -> HallucinationConfig:
    """Merge stored rows with the catalog, filling in any newly added rules."""
    by_id = {row["rule_id"]: row for row in rows}
    states: dict[str, HallucinationRuleState] = {}
    for spec in HALLUCINATION_RULE_SPECS:
        row = by_id.get(spec.id)
        if row is None:
            states[spec.id] = HallucinationRuleState(
                enabled=True, threshold=spec.default_threshold
            )
            continue
        threshold = row.get("threshold")
        if threshold is None:
            threshold = spec.default_threshold
        states[spec.id] = HallucinationRuleState(
            enabled=bool(row.get("enabled", True)),
            threshold=threshold if threshold is None else float(threshold),
        )
    return HallucinationConfig(rules=states)


def resolve_hallucination_config(
    config: HallucinationConfig | None = None,
) -> HallucinationConfig:
    """Use an explicit config, otherwise load saved DB values, otherwise defaults."""
    if config is not None:
        return config
    try:
        from app.services.hallucination_rules import get_hallucination_config

        return get_hallucination_config()
    except Exception:
        return DEFAULT_HALLUCINATION_CONFIG


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


def _phrase_is_significant(phrase: list[str]) -> bool:
    if len(phrase) >= 3:
        return True
    if len(phrase) == 2:
        key = " ".join(phrase)
        if key in INSIGNIFICANT_SHARED_PHRASES:
            return False
        return len(key) >= 8
    return False


def _labels_from_numbered_cue(text: str) -> list[str]:
    markers = list(_INTRA_NUMBERED_MARKER_RE.finditer(text))
    labels: list[str] = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        label = _normalize_text(text[start:end])
        if label:
            labels.append(label)
    return labels


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
                if not _phrase_is_significant(list(phrase)):
                    continue
                if len(phrase) < min_words:
                    continue
                if all(_contains_word_sequence(item, list(phrase)) for item in texts):
                    if best is None or len(phrase) > len(best):
                        best = list(phrase)
    return best


def _check_shared_phrase_run(
    segments: list[TranscriptionSegment],
    chunk_label: str | None,
    config: HallucinationConfig,
) -> list[HallucinationFinding]:
    """Flag long runs of lines that share the same core phrase (template loops)."""
    if not config.is_enabled("shared_phrase_run"):
        return []

    min_run = config.int_threshold("shared_phrase_run")
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

        if len(run) >= min_run:
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


def _check_intra_cue_numbered_list(
    text: str,
    index: int,
    chunk_label: str | None,
    config: HallucinationConfig,
) -> list[HallucinationFinding]:
    """Flag one subtitle cue that packs many numbered list items (STT menu loops)."""
    if not config.is_enabled("intra_cue_numbered_list"):
        return []

    min_items = config.int_threshold("intra_cue_numbered_list")
    labels = _labels_from_numbered_cue(text)
    if len(labels) < min_items:
        return []

    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    top_label, top_count = max(counts.items(), key=lambda item: item[1])
    if top_count >= min_items:
        preview = top_label[:50] + ("…" if len(top_label) > 50 else "")
        return [
            HallucinationFinding(
                code="intra_cue_numbered_list",
                message=(
                    f"{len(labels)} numbered list items in one cue, "
                    f"mostly repeating {preview!r}"
                ),
                segment_index=index,
                chunk_label=chunk_label,
            )
        ]

    shared = _find_longest_shared_phrase(labels, 3)
    if shared and len(labels) >= min_items:
        return [
            HallucinationFinding(
                code="intra_cue_numbered_list",
                message=(
                    f"{len(labels)} numbered list items in one cue share wording "
                    f"({' '.join(shared)!r})"
                ),
                segment_index=index,
                chunk_label=chunk_label,
            )
        ]
    return []


def _check_phrase_spam_in_cue(
    text: str,
    index: int,
    chunk_label: str | None,
    config: HallucinationConfig,
) -> list[HallucinationFinding]:
    """Flag a cue where the same word pair repeats many times (e.g. insert belly)."""
    if not config.is_enabled("phrase_spam_in_cue"):
        return []

    min_spam = config.int_threshold("phrase_spam_in_cue")
    words = _tokenize(text)
    if len(words) < min_spam:
        return []

    bigram_counts: dict[tuple[str, str], int] = {}
    for pos in range(len(words) - 1):
        bigram = (words[pos], words[pos + 1])
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    best_bigram = max(bigram_counts, key=lambda key: bigram_counts[key])
    best_count = bigram_counts[best_bigram]
    if best_count < min_spam:
        return []
    if not _phrase_is_significant(list(best_bigram)):
        return []

    phrase = " ".join(best_bigram)
    return [
        HallucinationFinding(
            code="phrase_spam_in_cue",
            message=(
                f"phrase {phrase!r} repeats {best_count} times in this cue "
                f"(>{min_spam - 1})"
            ),
            segment_index=index,
            chunk_label=chunk_label,
        )
    ]


def _check_numbered_enumeration_loop(
    segments: list[TranscriptionSegment],
    chunk_label: str | None,
    config: HallucinationConfig,
) -> list[HallucinationFinding]:
    """Flag runs of consecutive lines like ``4. …``, ``5. …`` (STT list loops)."""
    if not config.is_enabled("numbered_enumeration_loop"):
        return []

    min_run = config.int_threshold("numbered_enumeration_loop")
    findings: list[HallucinationFinding] = []
    run_indices: list[int] = []

    def flush_run() -> None:
        if len(run_indices) < min_run:
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
    segments: list[TranscriptionSegment],
    chunk_label: str | None,
    config: HallucinationConfig,
) -> list[HallucinationFinding]:
    """Flag consecutive segments with very similar wording (not exact duplicates)."""
    if not config.is_enabled("near_duplicate_run"):
        return []

    min_run = config.int_threshold("near_duplicate_run")
    if len(segments) < min_run:
        return []

    findings: list[HallucinationFinding] = []
    run_indices = [0]

    def flush_run() -> None:
        if len(run_indices) < min_run:
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
    segment: TranscriptionSegment,
    index: int,
    chunk_label: str | None,
    config: HallucinationConfig,
) -> list[HallucinationFinding]:
    if not _has_whisper_metrics(segment):
        return []

    findings: list[HallucinationFinding] = []
    text = (segment.text or "").strip()
    compression = float(getattr(segment, "compression_ratio", 0.0) or 0.0)
    avg_logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
    no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)

    max_compression = config.threshold("high_compression_ratio")
    min_logprob = config.threshold("low_avg_logprob")
    max_no_speech = config.threshold("speech_on_silence")

    if config.is_enabled("high_compression_ratio") and compression > max_compression:
        findings.append(
            HallucinationFinding(
                code="high_compression_ratio",
                message=(
                    f"compression_ratio={compression:.2f} "
                    f"(>{max_compression}) — typical Whisper hallucination signal"
                ),
                segment_index=index,
                chunk_label=chunk_label,
            )
        )
    if config.is_enabled("low_avg_logprob") and text and avg_logprob < min_logprob:
        findings.append(
            HallucinationFinding(
                code="low_avg_logprob",
                message=(
                    f"avg_logprob={avg_logprob:.2f} "
                    f"(<{min_logprob}) — low-confidence transcription"
                ),
                segment_index=index,
                chunk_label=chunk_label,
            )
        )
    if config.is_enabled("speech_on_silence") and text and no_speech > max_no_speech:
        findings.append(
            HallucinationFinding(
                code="speech_on_silence",
                message=(
                    f"no_speech_prob={no_speech:.2f} with non-empty text "
                    f"(>{max_no_speech})"
                ),
                segment_index=index,
                chunk_label=chunk_label,
            )
        )
    return findings


def _check_segment_density(
    segment: TranscriptionSegment,
    index: int,
    chunk_label: str | None,
    config: HallucinationConfig,
) -> list[HallucinationFinding]:
    if not config.is_enabled("excessive_cps"):
        return []
    text = (segment.text or "").strip()
    if not text:
        return []
    max_cps = config.threshold("excessive_cps")
    cps = _chars_per_second(segment.start, segment.end, text)
    if cps > max_cps:
        preview = text[:80] + ("…" if len(text) > 80 else "")
        return [
            HallucinationFinding(
                code="excessive_cps",
                message=(
                    f"{cps:.1f} chars/s (>{max_cps}) — "
                    f"text too dense for duration: {preview!r}"
                ),
                segment_index=index,
                chunk_label=chunk_label,
            )
        ]
    return []


def _check_character_spam(
    text: str,
    index: int,
    chunk_label: str | None,
    config: HallucinationConfig,
) -> list[HallucinationFinding]:
    if not config.is_enabled("character_spam"):
        return []
    min_run = config.int_threshold("character_spam")
    if re.search(rf"(.)\1{{{min_run - 1},}}", text):
        return [
            HallucinationFinding(
                code="character_spam",
                message="repeated single-character run detected",
                segment_index=index,
                chunk_label=chunk_label,
            )
        ]
    return []


def _check_known_phrases(
    text: str,
    index: int | None,
    chunk_label: str | None,
    config: HallucinationConfig,
) -> list[HallucinationFinding]:
    if not config.is_enabled("known_hallucination_phrase"):
        return []
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
    segments: list[TranscriptionSegment],
    chunk_label: str | None,
    config: HallucinationConfig,
) -> list[HallucinationFinding]:
    if not config.is_enabled("consecutive_duplicate_segments"):
        return []

    min_run = config.int_threshold("consecutive_duplicate_segments")
    if len(segments) < min_run:
        return []

    run_text = ""
    run_start = 0
    run_len = 0
    findings: list[HallucinationFinding] = []

    def flush_run() -> None:
        nonlocal run_len, run_text, run_start
        if run_len >= min_run and run_text:
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


def _find_repeated_phrase(
    text: str,
    *,
    min_repeat: int = MIN_PHRASE_REPEAT_COUNT,
    min_words: int = MIN_PHRASE_WORDS,
    max_words: int = MAX_PHRASE_WORDS,
) -> str | None:
    words = _tokenize(text)
    if len(words) < min_words * min_repeat:
        return None

    max_n = min(max_words, len(words) // min_repeat)
    for size in range(max_n, min_words - 1, -1):
        for start in range(0, len(words) - size * min_repeat + 1):
            phrase = words[start : start + size]
            repeats = 1
            pos = start + size
            while pos + size <= len(words) and words[pos : pos + size] == phrase:
                repeats += 1
                pos += size
            if repeats >= min_repeat:
                return " ".join(phrase)
    return None


def _check_phrase_loops(
    segments: list[TranscriptionSegment],
    chunk_label: str | None,
    config: HallucinationConfig,
) -> list[HallucinationFinding]:
    if not config.is_enabled("repeated_phrase_loop"):
        return []

    min_repeat = config.int_threshold("repeated_phrase_loop")
    combined = " ".join(_normalize_text(seg.text or "") for seg in segments if seg.text)
    if not combined:
        return []

    repeated = _find_repeated_phrase(combined, min_repeat=min_repeat)
    if repeated:
        return [
            HallucinationFinding(
                code="repeated_phrase_loop",
                message=f"phrase {repeated!r} repeats {min_repeat}+ times in sequence",
                chunk_label=chunk_label,
            )
        ]
    return []


def _check_dominant_word(
    segments: list[TranscriptionSegment],
    chunk_label: str | None,
    config: HallucinationConfig,
) -> list[HallucinationFinding]:
    if not config.is_enabled("dominant_word_loop"):
        return []

    min_ratio = config.threshold("dominant_word_loop")
    tokens = _tokenize(" ".join(seg.text or "" for seg in segments))
    if len(tokens) < DOMINANT_WORD_MIN_TOKENS:
        return []

    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    word, count = max(counts.items(), key=lambda item: item[1])
    ratio = count / len(tokens)
    if ratio >= min_ratio:
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
    config: HallucinationConfig,
) -> list[HallucinationFinding]:
    if not segments:
        if not config.is_enabled("empty_transcript"):
            return []
        return [
            HallucinationFinding(
                code="empty_transcript",
                message="no speech segments after transcription",
                chunk_label=chunk_label,
            )
        ]

    if not config.is_enabled("transcript_too_dense"):
        return []

    max_cps = config.threshold("transcript_too_dense")
    total_duration = audio_duration
    if total_duration is None:
        total_duration = max((seg.end for seg in segments), default=0.0) - min(
            (seg.start for seg in segments), default=0.0
        )
    if total_duration <= 0:
        return []

    combined = "".join(_normalize_text(seg.text or "") for seg in segments)
    cps = len(combined.replace(" ", "")) / max(total_duration, 0.001)
    if cps > max_cps:
        return [
            HallucinationFinding(
                code="transcript_too_dense",
                message=(
                    f"overall {cps:.1f} chars/s (>{max_cps}) "
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
    config: HallucinationConfig | None = None,
) -> HallucinationReport:
    """Run all hallucination heuristics on ``segments``."""
    resolved = resolve_hallucination_config(config)
    report = HallucinationReport(
        segment_count=len(segments),
        total_duration=audio_duration or 0.0,
        source_label=source_label,
    )

    for index, segment in enumerate(segments):
        text = segment.text or ""
        report.findings.extend(
            _check_whisper_metrics(segment, index, chunk_label, resolved)
        )
        report.findings.extend(
            _check_segment_density(segment, index, chunk_label, resolved)
        )
        report.findings.extend(
            _check_character_spam(text, index, chunk_label, resolved)
        )
        report.findings.extend(
            _check_known_phrases(text, index, chunk_label, resolved)
        )
        report.findings.extend(
            _check_intra_cue_numbered_list(text, index, chunk_label, resolved)
        )
        report.findings.extend(
            _check_phrase_spam_in_cue(text, index, chunk_label, resolved)
        )

    report.findings.extend(
        _check_consecutive_duplicate_segments(segments, chunk_label, resolved)
    )
    report.findings.extend(
        _check_numbered_enumeration_loop(segments, chunk_label, resolved)
    )
    report.findings.extend(_check_near_duplicate_run(segments, chunk_label, resolved))
    report.findings.extend(_check_shared_phrase_run(segments, chunk_label, resolved))
    report.findings.extend(_check_phrase_loops(segments, chunk_label, resolved))
    report.findings.extend(_check_dominant_word(segments, chunk_label, resolved))
    report.findings.extend(
        _check_transcript_density(segments, audio_duration, chunk_label, resolved)
    )

    return report


def assert_segments_acceptable(
    segments: list[TranscriptionSegment],
    *,
    audio_duration: float | None = None,
    source_label: str = "",
    chunk_label: str | None = None,
    config: HallucinationConfig | None = None,
) -> None:
    """Raise ``STTHallucinationError`` when transcription looks hallucinated."""
    report = analyze_segments(
        segments,
        audio_duration=audio_duration,
        source_label=source_label,
        chunk_label=chunk_label,
        config=config,
    )
    if not report.passed:
        raise STTHallucinationError(report)
