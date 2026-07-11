"""
Tests for automatic source language detection during project creation.
"""

from video_analyzer import detect_dominant_language


class TestDominantLanguageDetection:
    """Test cases for detecting the most common language from audio streams."""

    def test_single_video_single_language(self):
        """Test detection with one video having one audio stream."""
        audio_streams = [[{"language": "eng", "codec": "aac"}]]

        result = detect_dominant_language(audio_streams)

        assert result == "eng"

    def test_single_video_multiple_streams_same_language(self):
        """Test detection with one video having multiple streams of the same language."""
        audio_streams = [
            [
                {"language": "rus", "codec": "aac"},
                {"language": "rus", "codec": "ac3"},
            ]
        ]

        result = detect_dominant_language(audio_streams)

        assert result == "rus"

    def test_multiple_videos_same_language(self):
        """Test detection with multiple videos all having the same language."""
        audio_streams = [
            [{"language": "zho", "codec": "aac"}],
            [{"language": "zho", "codec": "aac"}],
            [{"language": "zho", "codec": "aac"}],
        ]

        result = detect_dominant_language(audio_streams)

        assert result == "zho"

    def test_multiple_videos_mixed_languages_clear_majority(self):
        """Test detection with multiple videos, one language is clearly dominant."""
        audio_streams = [
            [{"language": "eng", "codec": "aac"}],  # English
            [{"language": "eng", "codec": "aac"}],  # English
            [{"language": "eng", "codec": "aac"}],  # English
            [{"language": "spa", "codec": "aac"}],  # Spanish (minority)
        ]

        result = detect_dominant_language(audio_streams)

        assert result == "eng"

    def test_two_char_codes_normalized_to_three(self):
        """Test that 2-char language codes are normalized to 3-char codes."""
        audio_streams = [
            [{"language": "en", "codec": "aac"}],  # 2-char
            [{"language": "en", "codec": "aac"}],  # 2-char
        ]

        result = detect_dominant_language(audio_streams)

        assert result == "eng"  # Should be normalized to 3-char

    def test_mixed_two_and_three_char_codes(self):
        """Test that mixed 2-char and 3-char codes are counted together."""
        audio_streams = [
            [{"language": "ru", "codec": "aac"}],  # 2-char
            [{"language": "rus", "codec": "aac"}],  # 3-char
            [{"language": "ru", "codec": "ac3"}],  # 2-char
        ]

        result = detect_dominant_language(audio_streams)

        assert result == "rus"  # Should recognize both as same language

    def test_unknown_languages_ignored(self):
        """Test that unknown/invalid languages are ignored."""
        audio_streams = [
            [{"language": "unknown", "codec": "aac"}],
            [{"language": "und", "codec": "aac"}],
            [{"language": "", "codec": "aac"}],
            [{"language": "eng", "codec": "aac"}],  # Only valid one
        ]

        result = detect_dominant_language(audio_streams)

        assert result == "eng"

    def test_all_unknown_languages_returns_empty(self):
        """Test that when all languages are unknown, empty string is returned."""
        audio_streams = [
            [{"language": "unknown", "codec": "aac"}],
            [{"language": "und", "codec": "aac"}],
            [{"language": "", "codec": "aac"}],
        ]

        result = detect_dominant_language(audio_streams)

        assert result == ""

    def test_empty_audio_streams_list(self):
        """Test that empty input returns empty string."""
        audio_streams = []

        result = detect_dominant_language(audio_streams)

        assert result == ""

    def test_videos_with_no_audio_streams(self):
        """Test videos with empty audio stream lists."""
        audio_streams = [[], [], []]

        result = detect_dominant_language(audio_streams)

        assert result == ""

    def test_case_insensitive_language_detection(self):
        """Test that language codes are case-insensitive."""
        audio_streams = [
            [{"language": "ENG", "codec": "aac"}],
            [{"language": "Eng", "codec": "aac"}],
            [{"language": "eng", "codec": "aac"}],
        ]

        result = detect_dominant_language(audio_streams)

        assert result == "eng"

    def test_real_world_mixed_content(self):
        """Test realistic scenario with mixed valid/invalid languages."""
        audio_streams = [
            # Video 1: Russian with commentary
            [{"language": "rus", "codec": "aac"}, {"language": "eng", "codec": "ac3"}],
            # Video 2: Russian only
            [{"language": "rus", "codec": "aac"}],
            # Video 3: Unknown + Russian
            [{"language": "unknown", "codec": "aac"}, {"language": "rus", "codec": "ac3"}],
            # Video 4: Russian with music track (no language)
            [{"language": "rus", "codec": "aac"}, {"language": "", "codec": "aac"}],
        ]

        result = detect_dominant_language(audio_streams)

        # Russian should be dominant (4 occurrences vs 1 English)
        assert result == "rus"

    def test_tie_breaker_first_in_counter(self):
        """Test behavior when two languages have equal occurrences."""
        # Counter.most_common() returns in arbitrary order for ties,
        # but the first one encountered will be returned
        audio_streams = [
            [{"language": "eng", "codec": "aac"}],
            [{"language": "spa", "codec": "aac"}],
        ]

        result = detect_dominant_language(audio_streams)

        # Either is valid, but should be one of them
        assert result in ["eng", "spa"]

    def test_multi_stream_videos_realistic(self):
        """Test videos with multiple audio tracks (original + dubs)."""
        audio_streams = [
            # Movie with original Chinese + English dub
            [{"language": "zho", "codec": "aac"}, {"language": "eng", "codec": "ac3"}],
            # Another Chinese movie with English dub
            [{"language": "zho", "codec": "aac"}, {"language": "eng", "codec": "ac3"}],
            # Chinese movie without dub
            [{"language": "zho", "codec": "aac"}],
        ]

        result = detect_dominant_language(audio_streams)

        # Chinese should be detected as dominant (3 vs 2)
        assert result == "zho"


class TestLanguageDetectionIntegration:
    """Integration tests for language detection in project workflow."""

    def test_language_detection_workflow(self):
        """Test the complete workflow of language detection."""
        # Simulate scanning multiple videos in a project
        video_analyses = [
            {
                "filename": "lecture_01.mp4",
                "audio_streams": [{"language": "rus", "codec": "aac"}],
            },
            {
                "filename": "lecture_02.mp4",
                "audio_streams": [{"language": "rus", "codec": "aac"}],
            },
            {
                "filename": "lecture_03.mp4",
                "audio_streams": [
                    {"language": "rus", "codec": "aac"},
                    {"language": "eng", "codec": "ac3"},  # Commentary track
                ],
            },
        ]

        # Extract audio streams
        all_audio_streams = [v["audio_streams"] for v in video_analyses]

        # Detect dominant language
        detected_language = detect_dominant_language(all_audio_streams)

        # Should detect Russian as the source language
        assert detected_language == "rus"

    def test_empty_project_no_videos(self):
        """Test handling of a project with no video files."""
        all_audio_streams = []

        detected_language = detect_dominant_language(all_audio_streams)

        # Should return empty string, not crash
        assert detected_language == ""
