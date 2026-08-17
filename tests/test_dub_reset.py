"""Tests for dub-reset (remove generated sub + first dubbed audio track)."""

from __future__ import annotations

from pathlib import Path
from subprocess import run as run_cmd
from unittest.mock import MagicMock, patch

import pytest

from app.services.dub_reset import (
    DubResetError,
    find_generated_subtitle_paths,
    reset_dubbed_video,
    strip_first_audio_track,
    working_dir_subtitle_path,
)
from database import DatabaseManager
from utils import convert_to_two_char_lang_code, normalize_lang_code


def _write(path: Path, content: str = "sub") -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestLanguageHelpers:
    def test_two_char_from_three(self) -> None:
        assert convert_to_two_char_lang_code("eng") == "en"
        assert convert_to_two_char_lang_code("jpn") == "ja"
        assert convert_to_two_char_lang_code("en") == "en"

    def test_normalize_lang_code(self) -> None:
        assert normalize_lang_code("en") == "eng"
        assert normalize_lang_code("ENG") == "eng"
        assert normalize_lang_code("") == ""
        assert normalize_lang_code(None) == ""


class TestFindGeneratedSubtitlePaths:
    def test_matches_target_language_sidecar_only(self, tmp_path: Path) -> None:
        video = tmp_path / "lesson.mp4"
        video.write_bytes(b"fake")
        target = _write(tmp_path / "lesson.en.srt")
        _write(tmp_path / "lesson.ru.srt")
        _write(tmp_path / "lesson.srt")
        _write(tmp_path / "other.en.srt")

        found = find_generated_subtitle_paths(video, "eng")

        assert found == [target]

    def test_matches_three_letter_suffix(self, tmp_path: Path) -> None:
        video = tmp_path / "lesson.mp4"
        video.write_bytes(b"fake")
        target = _write(tmp_path / "lesson.eng.srt")

        assert find_generated_subtitle_paths(video, "en") == [target]

    def test_empty_when_video_missing(self, tmp_path: Path) -> None:
        assert find_generated_subtitle_paths(tmp_path / "missing.mp4", "eng") == []

    def test_empty_when_no_target_language(self, tmp_path: Path) -> None:
        video = tmp_path / "lesson.mp4"
        video.write_bytes(b"fake")
        _write(tmp_path / "lesson.en.srt")
        assert find_generated_subtitle_paths(video, "") == []


class TestWorkingDirSubtitlePath:
    def test_layout_matches_reproj_subtitles_section(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"fake")

        path = working_dir_subtitle_path(video, str(project), "project")

        assert (
            path
            == project / ".redubber" / "lesson.mp4" / "03_subtitles" / "lesson.en.srt"
        )


class TestStripFirstAudioTrack:
    def test_raises_when_ffmpeg_fails(self, tmp_path: Path) -> None:
        video = tmp_path / "lesson.mp4"
        video.write_bytes(b"fake")

        failed = MagicMock(returncode=1, stderr="boom", stdout="")
        with (
            patch("app.services.dub_reset.subprocess.run", return_value=failed),
            pytest.raises(DubResetError, match="ffmpeg failed"),
        ):
            strip_first_audio_track(video)

        assert video.exists()
        assert not list(tmp_path.glob(".lesson.undub*"))

    def test_replaces_original_on_success(self, tmp_path: Path) -> None:
        video = tmp_path / "lesson.mp4"
        video.write_bytes(b"original")

        def fake_run(cmd, **_kwargs):
            out = Path(cmd[-1])
            out.write_bytes(b"stripped")
            return MagicMock(returncode=0, stderr="", stdout="")

        with patch("app.services.dub_reset.subprocess.run", side_effect=fake_run):
            strip_first_audio_track(video)

        assert video.read_bytes() == b"stripped"

    def test_ffmpeg_drops_first_audio_stream(self, tmp_path: Path) -> None:
        video = tmp_path / "two_tracks.mp4"
        create = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:d=0.4",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=8000:cl=mono:d=0.4",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=8000:cl=mono:d=0.4",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "2:a",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ]
        built = run_cmd(create, capture_output=True, text=True)
        if built.returncode != 0:
            pytest.skip(f"could not build test video: {built.stderr[-400:]}")

        from video_analyzer import get_video_info_with_duration

        before = get_video_info_with_duration(video)
        assert len(before["audio_streams"]) == 2

        strip_first_audio_track(video)

        after = get_video_info_with_duration(video)
        assert len(after["audio_streams"]) == 1


def _seed_final_video(
    tmp_path: Path, *, in_target_state: bool = True
) -> tuple[DatabaseManager, int, dict, Path, Path]:
    project_dir = tmp_path / "videos"
    project_dir.mkdir()
    video = project_dir / "lesson.mp4"
    video.write_bytes(b"fake-video")
    srt = _write(
        project_dir / "lesson.en.srt", "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
    )

    db = DatabaseManager(str(tmp_path / "test.db"))
    project_id = db.add_project(str(project_dir), "Demo")
    db.set_target_language(project_id, "eng")
    db.add_subtitle_file(project_id, str(srt), srt.name, "eng")

    audio_streams = (
        [
            {"index": 0, "language": "eng", "codec": "aac"},
            {"index": 1, "language": "rus", "codec": "aac"},
        ]
        if in_target_state
        else [{"index": 0, "language": "rus", "codec": "aac"}]
    )
    subtitles = [
        {"language": "eng", "embedded": False, "path": str(srt), "filename": srt.name}
    ]
    db.save_video_analysis(
        project_id,
        {
            "filename": "lesson.mp4",
            "path": str(video),
            "size_mb": 1.0,
            "duration_seconds": 10,
            "audio_streams": audio_streams,
            "subtitles": subtitles,
        },
    )
    record = db.get_video_analysis(project_id)[0]
    return db, project_id, record, video, srt


class TestResetDubbedVideo:
    def test_rejects_video_not_in_target_state(self, tmp_path: Path) -> None:
        db, project_id, record, _video, _srt = _seed_final_video(
            tmp_path, in_target_state=False
        )

        with pytest.raises(DubResetError, match="final redubbed state"):
            reset_dubbed_video(
                db=db,
                project_id=project_id,
                video_record=record,
                project_path=str(_video.parent),
                project_name="Demo",
                target_language="eng",
            )

    def test_deletes_generated_sub_and_strips_audio(self, tmp_path: Path) -> None:
        db, project_id, record, video, srt = _seed_final_video(tmp_path)
        working_srt = (
            video.parent / ".redubber" / "lesson.mp4" / "03_subtitles" / "lesson.en.srt"
        )
        working_srt.parent.mkdir(parents=True)
        working_srt.write_text("generated", encoding="utf-8")

        with (
            patch("app.services.dub_reset.strip_first_audio_track") as strip,
            patch("redubber.sync_video_metadata") as sync,
        ):
            result = reset_dubbed_video(
                db=db,
                project_id=project_id,
                video_record=record,
                project_path=str(video.parent),
                project_name="Demo",
                target_language="eng",
            )

        strip.assert_called_once_with(video)
        sync.assert_called_once()
        assert not srt.exists()
        assert not working_srt.exists()
        assert result["removed_audio_track"] is True
        assert str(srt) in result["deleted_subtitles"]
        assert str(working_srt) in result["deleted_subtitles"]
        remaining = db.get_subtitle_files_for_video(project_id, "lesson.mp4")
        assert remaining == []
