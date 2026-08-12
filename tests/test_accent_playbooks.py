"""Tests for accent playbooks used in voice instruction generation."""

from app.services.accent_playbooks import (
    accent_intensity_guidance,
    format_playbook_block,
    get_accent_playbook,
    normalize_language_code,
)


def test_normalize_language_aliases():
    assert normalize_language_code("ja") == "jpn"
    assert normalize_language_code("Japanese") == "jpn"
    assert normalize_language_code("ko") == "kor"
    assert normalize_language_code("zh-CN") == "zho"
    assert normalize_language_code("jpn") == "jpn"


def test_get_japanese_playbook():
    book = get_accent_playbook("jpn")
    assert book is not None
    assert book["name"] == "Japanese"
    assert "syllable-timed" in book["traits"]


def test_format_playbook_block_includes_checklist():
    block = format_playbook_block("kor")
    assert "ACCENT PLAYBOOK" in block
    assert "Korean" in block
    assert "Phonetics" in block


def test_unknown_language_has_no_playbook():
    assert get_accent_playbook("xyz") is None
    assert format_playbook_block("xyz") == ""


def test_accent_intensity_guidance():
    assert "SUBTLE" in accent_intensity_guidance("subtle")
    assert "STRONG" in accent_intensity_guidance("strong")
    assert "PLAYFUL" in accent_intensity_guidance("funny")
