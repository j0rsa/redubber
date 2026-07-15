"""Tests for target-state detection used for replaced video counts."""

from __future__ import annotations

from utils import count_videos_in_target_state, is_video_in_target_state


class TestIsVideoInTargetState:
    """Tests for is_video_in_target_state."""

    def test_requires_target_language(self) -> None:
        assert is_video_in_target_state(
            [{"language": "eng"}, {"language": "jpn"}],
            [{"language": "eng"}],
            "",
        ) is False

    def test_requires_two_audio_streams(self) -> None:
        assert is_video_in_target_state(
            [{"language": "eng"}],
            [{"language": "eng"}],
            "eng",
        ) is False

    def test_requires_target_language_audio_track(self) -> None:
        assert is_video_in_target_state(
            [{"language": "jpn"}, {"language": "fra"}],
            [{"language": "eng"}],
            "eng",
        ) is False

    def test_requires_target_language_subtitle(self) -> None:
        assert is_video_in_target_state(
            [{"language": "eng"}, {"language": "jpn"}],
            [{"language": "jpn"}],
            "eng",
        ) is False

    def test_true_when_audio_and_subtitle_match_target(self) -> None:
        assert is_video_in_target_state(
            [{"language": "eng"}, {"language": "jpn"}],
            [{"language": "eng"}],
            "eng",
        ) is True

    def test_accepts_pydantic_like_objects(self) -> None:
        class Stream:
            def __init__(self, language: str) -> None:
                self.language = language

        assert is_video_in_target_state(
            [Stream("eng"), Stream("jpn")],
            [Stream("eng")],
            "eng",
        ) is True


class TestCountVideosInTargetState:
    """Tests for count_videos_in_target_state."""

    def test_counts_matching_records_only(self) -> None:
        records = [
            {
                "audio_streams": [{"language": "eng"}, {"language": "jpn"}],
                "subtitle_matches": [{"language": "eng"}],
            },
            {
                "audio_streams": [{"language": "jpn"}],
                "subtitle_matches": [{"language": "jpn"}],
            },
            {
                "audio_streams": [{"language": "eng"}, {"language": "spa"}],
                "subtitle_matches": [{"language": "eng"}],
            },
        ]

        assert count_videos_in_target_state(records, "eng") == 2
