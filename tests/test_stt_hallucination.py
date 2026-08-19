"""Tests for STT hallucination detection."""

from __future__ import annotations

import pytest
from openai.types.audio.transcription_segment import TranscriptionSegment

from stt_hallucination import (
    STTHallucinationError,
    analyze_segments,
    assert_segments_acceptable,
)


def _seg(
    text: str,
    *,
    start: float = 0.0,
    end: float = 5.0,
    avg_logprob: float = -0.3,
    compression_ratio: float = 1.2,
    no_speech_prob: float = 0.05,
) -> TranscriptionSegment:
    return TranscriptionSegment(
        id=0,
        seek=0,
        start=start,
        end=end,
        text=text,
        tokens=[],
        temperature=0.0,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
        no_speech_prob=no_speech_prob,
    )


class TestAcceptableTranscripts:
    def test_normal_dialogue_passes(self) -> None:
        segments = [
            _seg("Welcome to today's lesson on drawing basics.", start=0, end=4),
            _seg("We will start with simple shapes.", start=4.2, end=8),
            _seg("Pay attention to proportions.", start=8.5, end=12),
        ]
        report = analyze_segments(segments, audio_duration=12.0)
        assert report.passed
        assert_segments_acceptable(segments, audio_duration=12.0)


class TestWhisperMetrics:
    def test_high_compression_ratio_fails(self) -> None:
        segments = [_seg("hello world", compression_ratio=3.1)]
        report = analyze_segments(segments, audio_duration=5.0)
        assert not report.passed
        assert any(f.code == "high_compression_ratio" for f in report.findings)

    def test_low_logprob_fails(self) -> None:
        segments = [_seg("maybe this is noise", avg_logprob=-1.4)]
        report = analyze_segments(segments, audio_duration=5.0)
        assert any(f.code == "low_avg_logprob" for f in report.findings)

    def test_speech_on_silence_fails(self) -> None:
        segments = [_seg("ghost speech", no_speech_prob=0.9)]
        report = analyze_segments(segments, audio_duration=5.0)
        assert any(f.code == "speech_on_silence" for f in report.findings)


class TestRepetitionAndLoops:
    def test_consecutive_duplicate_segments(self) -> None:
        repeated = "Thanks for watching and please subscribe now"
        segments = [
            _seg(repeated, start=0, end=3),
            _seg(repeated, start=3, end=6),
            _seg(repeated, start=6, end=9),
        ]
        report = analyze_segments(segments, audio_duration=9.0)
        assert any(f.code == "consecutive_duplicate_segments" for f in report.findings)

    def test_numbered_enumeration_loop(self) -> None:
        lines = [
            "4. Not with the water like a fool.",
            "5. Not with the water as a fool.",
            "6. Stop with the water.",
            "7. Not with the water as a fill.",
        ]
        segments = [_seg(text, start=i * 5, end=(i + 1) * 5) for i, text in enumerate(lines)]
        report = analyze_segments(segments, audio_duration=20.0)
        numbered = [f for f in report.findings if f.code == "numbered_enumeration_loop"]
        assert len(numbered) == 4
        assert {f.segment_index for f in numbered} == {0, 1, 2, 3}

    def test_near_duplicate_run(self) -> None:
        lines = [
            "Not with the water like a fool.",
            "Not with the water as a fool.",
            "Not with the water as a fill.",
        ]
        segments = [_seg(text, start=i * 4, end=(i + 1) * 4) for i, text in enumerate(lines)]
        report = analyze_segments(segments, audio_duration=12.0)
        assert any(f.code == "near_duplicate_run" for f in report.findings)

    def test_phrase_loop_in_one_segment(self) -> None:
        text = " ".join(["hello there friend"] * 4)
        segments = [_seg(text, end=20.0)]
        report = analyze_segments(segments, audio_duration=20.0)
        assert any(f.code == "repeated_phrase_loop" for f in report.findings)

    def test_dominant_word_loop(self) -> None:
        tokens = ["music"] * 20 + ["and", "the", "a", "is"]
        text = " ".join(tokens)
        segments = [_seg(text, end=30.0)]
        report = analyze_segments(segments, audio_duration=30.0)
        assert any(f.code == "dominant_word_loop" for f in report.findings)


class TestKnownPhrasesAndDensity:
    def test_known_hallucination_phrase(self) -> None:
        segments = [_seg("Thank you for watching this video tutorial.")]
        report = analyze_segments(segments, audio_duration=5.0)
        assert any(f.code == "known_hallucination_phrase" for f in report.findings)

    def test_excessive_chars_per_second(self) -> None:
        dense = "word " * 80
        segments = [_seg(dense.strip(), end=2.0)]
        report = analyze_segments(segments, audio_duration=2.0)
        assert any(
            f.code in {"excessive_cps", "transcript_too_dense"} for f in report.findings
        )

    def test_normal_density_passes(self) -> None:
        # ~35 chars/s should not trigger density warnings
        text = "a" * 35
        segments = [_seg(text, end=1.0)]
        report = analyze_segments(segments, audio_duration=1.0)
        assert not any(
            f.code in {"excessive_cps", "transcript_too_dense"} for f in report.findings
        )

    def test_empty_transcript_fails(self) -> None:
        report = analyze_segments([], audio_duration=10.0)
        assert any(f.code == "empty_transcript" for f in report.findings)


class TestAssertHelper:
    def test_raises_with_summary(self) -> None:
        segments = [_seg("Thank you for watching")]
        with pytest.raises(STTHallucinationError) as exc:
            assert_segments_acceptable(segments, audio_duration=5.0)
        assert "STT quality check failed" in str(exc.value)
        assert exc.value.report.findings
