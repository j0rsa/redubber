"""Guard that hallucination rules are driven by saved DB values, not constants."""

from __future__ import annotations

import inspect

import pytest
from fastapi.testclient import TestClient
from openai.types.audio.transcription_segment import TranscriptionSegment

from app.services.subtitle_quality_rules import (
    analyze_subtitle_file,
    analyze_subtitle_quality,
    unique_rule_ids,
)
from stt_hallucination import (
    HALLUCINATION_RULE_SPECS,
    STTHallucinationError,
    analyze_segments,
    assert_segments_acceptable,
    _check_character_spam,
    _check_consecutive_duplicate_segments,
    _check_dominant_word,
    _check_intra_cue_numbered_list,
    _check_known_phrases,
    _check_near_duplicate_run,
    _check_numbered_enumeration_loop,
    _check_phrase_loops,
    _check_phrase_spam_in_cue,
    _check_segment_density,
    _check_shared_phrase_run,
    _check_transcript_density,
    _check_whisper_metrics,
    _find_repeated_phrase,
)

# Names of factory defaults. Runtime checks must not read these.
_FACTORY_THRESHOLD_NAMES = (
    "MAX_COMPRESSION_RATIO",
    "MIN_AVG_LOGPROB",
    "MAX_NO_SPEECH_WITH_TEXT",
    "MAX_CHARS_PER_SECOND",
    "MIN_CONSECUTIVE_DUPLICATE_SEGMENTS",
    "MIN_PHRASE_REPEAT_COUNT",
    "DOMINANT_WORD_RATIO",
    "CHAR_RUN_MIN",
    "MIN_SHARED_PHRASE_RUN",
    "MIN_INTRA_CUE_NUMBERED_ITEMS",
    "MIN_PHRASE_SPAM_COUNT",
)

_CHECK_FUNCTIONS = (
    _check_whisper_metrics,
    _check_segment_density,
    _check_character_spam,
    _check_known_phrases,
    _check_intra_cue_numbered_list,
    _check_phrase_spam_in_cue,
    _check_consecutive_duplicate_segments,
    _check_numbered_enumeration_loop,
    _check_near_duplicate_run,
    _check_shared_phrase_run,
    _check_phrase_loops,
    _check_dominant_word,
    _check_transcript_density,
    _find_repeated_phrase,
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


def test_catalog_defaults_match_former_hardcoded_values() -> None:
    by_id = {spec.id: spec for spec in HALLUCINATION_RULE_SPECS}
    assert by_id["excessive_cps"].default_threshold == 40.0
    assert by_id["transcript_too_dense"].default_threshold == 40.0
    assert by_id["high_compression_ratio"].default_threshold == 2.4
    assert by_id["low_avg_logprob"].default_threshold == -1.0
    assert by_id["speech_on_silence"].default_threshold == 0.55
    assert by_id["dominant_word_loop"].default_threshold == 0.38
    assert by_id["character_spam"].default_threshold == 8.0
    assert by_id["consecutive_duplicate_segments"].default_threshold == 3.0
    assert by_id["known_hallucination_phrase"].default_threshold is None


def test_check_functions_do_not_read_factory_threshold_constants() -> None:
    for func in _CHECK_FUNCTIONS:
        source = inspect.getsource(func)
        for name in _FACTORY_THRESHOLD_NAMES:
            assert name not in source, f"{func.__name__} still references {name}"


class TestPublicApisReadSavedRules:
    """Call sites that omit config must still honor values persisted via Settings."""

    def test_analyze_segments_uses_saved_threshold(self, client: TestClient) -> None:
        dense = "word " * 80
        segments = [_seg(dense.strip(), end=2.0)]
        default_report = analyze_segments(segments, audio_duration=2.0)
        assert any(
            f.code in {"excessive_cps", "transcript_too_dense"}
            for f in default_report.findings
        )

        response = client.put(
            "/api/settings",
            json={
                "hallucination_rules": [
                    {"id": "excessive_cps", "threshold": 200},
                    {"id": "transcript_too_dense", "threshold": 200},
                ]
            },
        )
        assert response.status_code == 200

        tuned = analyze_segments(segments, audio_duration=2.0)
        assert not any(
            f.code in {"excessive_cps", "transcript_too_dense"} for f in tuned.findings
        )

    def test_analyze_subtitle_quality_uses_saved_enable_flag(
        self, client: TestClient
    ) -> None:
        cues = [(0.0, 5.0, "Thank you for watching this video.")]
        assert "known_hallucination_phrase" in unique_rule_ids(
            list(analyze_subtitle_quality(cues).breaches)
        )

        response = client.put(
            "/api/settings",
            json={
                "hallucination_rules": [
                    {"id": "known_hallucination_phrase", "enabled": False}
                ]
            },
        )
        assert response.status_code == 200

        analysis = analyze_subtitle_quality(cues)
        assert "known_hallucination_phrase" not in unique_rule_ids(
            list(analysis.breaches)
        )

    def test_analyze_subtitle_file_uses_saved_enable_flag(
        self, client: TestClient, tmp_path
    ) -> None:
        path = tmp_path / "clip.en.srt"
        path.write_text(
            "1\n00:00:00,000 --> 00:00:05,000\nThank you for watching this video.\n",
            encoding="utf-8",
        )
        assert any(
            b.rule_id == "known_hallucination_phrase"
            for b in analyze_subtitle_file(path).breaches
        )

        client.put(
            "/api/settings",
            json={
                "hallucination_rules": [
                    {"id": "known_hallucination_phrase", "enabled": False}
                ]
            },
        )
        assert all(
            b.rule_id != "known_hallucination_phrase"
            for b in analyze_subtitle_file(path).breaches
        )

    def test_assert_segments_acceptable_uses_saved_enable_flag(
        self, client: TestClient
    ) -> None:
        segments = [_seg("Thank you for watching")]
        with pytest.raises(STTHallucinationError):
            assert_segments_acceptable(segments, audio_duration=5.0)

        client.put(
            "/api/settings",
            json={
                "hallucination_rules": [
                    {"id": "known_hallucination_phrase", "enabled": False}
                ]
            },
        )
        assert_segments_acceptable(segments, audio_duration=5.0)
