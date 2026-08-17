"""Tests for ffprobe duration lookup used at the start of extract."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from redubber import Redubber


def _redubber() -> Redubber:
    return Redubber(openai_token="test", interactive=False)


class TestGetMediaDuration:
    def test_uses_format_duration(self) -> None:
        payload = {"format": {"duration": "12.5"}, "streams": []}
        result = MagicMock(returncode=0, stdout=json.dumps(payload), stderr="")
        with patch("redubber.subprocess.run", return_value=result):
            assert _redubber().get_media_duration("/videos/a.mp4") == 12.5

    def test_falls_back_to_stream_duration(self) -> None:
        payload = {"format": {}, "streams": [{"duration": "8.0"}]}
        result = MagicMock(returncode=0, stdout=json.dumps(payload), stderr="")
        with patch("redubber.subprocess.run", return_value=result):
            assert _redubber().get_media_duration("/videos/a.mp4") == 8.0

    def test_includes_ffprobe_stderr_on_failure(self) -> None:
        result = MagicMock(
            returncode=1,
            stdout="",
            stderr="moov atom not found",
        )
        with (
            patch("redubber.subprocess.run", return_value=result),
            pytest.raises(RuntimeError, match="moov atom not found"),
        ):
            _redubber().get_media_duration("/videos/broken.mp4")
