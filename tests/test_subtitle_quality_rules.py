"""Tests for the subtitle quality rule registry."""

from __future__ import annotations

from app.services.subtitle_quality_rules import (
    SUBTITLE_QUALITY_RULES,
    analyze_subtitle_quality,
    breaches_for_segment,
    unique_rule_ids,
)
from stt_hallucination import (
    HallucinationConfig,
    HallucinationRuleState,
    default_hallucination_config,
)


def test_rule_registry_has_unique_ids() -> None:
    ids = [rule.id for rule in SUBTITLE_QUALITY_RULES]
    assert len(ids) == len(set(ids))


def test_breached_rule_count_per_cue() -> None:
    cues = [(0.0, 5.0, "Thank you for watching this video.")]
    analysis = analyze_subtitle_quality(cues)
    segment_breaches = breaches_for_segment(analysis.breaches, 0)
    assert unique_rule_ids(segment_breaches) == ["known_hallucination_phrase"]
    assert len(unique_rule_ids(segment_breaches)) == 1


def test_disabled_rule_is_not_reported() -> None:
    rules = dict(default_hallucination_config().rules)
    previous = rules["known_hallucination_phrase"]
    rules["known_hallucination_phrase"] = HallucinationRuleState(
        enabled=False, threshold=previous.threshold
    )
    cues = [(0.0, 5.0, "Thank you for watching this video.")]
    analysis = analyze_subtitle_quality(
        cues, config=HallucinationConfig(rules=rules)
    )
    assert unique_rule_ids(list(analysis.breaches)) == []
