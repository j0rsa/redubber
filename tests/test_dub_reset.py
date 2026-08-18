"""Tests for dub-reset (remove generated sub + language-verified dubbed audio track)."""

from __future__ import annotations

from pathlib import Path
from shutil import which
from subprocess import run as run_cmd
from unittest.mock import MagicMock, patch

import pytest

from app.services.dub_reset import (
    DubResetError,
    backup_video_before_reset,
    clear_finalization_artifacts,
    find_generated_subtitle_paths,
    identify_dubbed_stream_index,
    reconcile_video_with_disk,
    reset_dubbed_video,
    strip_dubbed_audio_track,
    strip_first_audio_track,
    working_dir_subtitle_path,
)
from database import DatabaseManager
from utils import convert_to_two_char_lang_code, normalize_lang_code

TWO_TRACK_STREAMS = [
    {"index": 1, "language": "eng", "codec": "aac"},
    {"index": 2, "language": "rus", "codec": "aac"},
]
ONE_TRACK_STREAMS = [
    {"index": 2, "language": "rus", "codec": "aac"},
]


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


class TestIdentifyDubbedStreamIndex:
    def test_returns_target_language_track(self) -> None:
        index = identify_dubbed_stream_index(TWO_TRACK_STREAMS, "eng", "rus")
        assert index == 1

    def test_rejects_single_track(self) -> None:
        with pytest.raises(DubResetError, match="at least two audio tracks"):
            identify_dubbed_stream_index(ONE_TRACK_STREAMS, "eng")

    def test_rejects_untagged_tracks(self) -> None:
        streams = [
            {"index": 1, "language": "eng", "codec": "aac"},
            {"index": 2, "language": "unknown", "codec": "aac"},
        ]
        with pytest.raises(DubResetError, match="missing language tags"):
            identify_dubbed_stream_index(streams, "eng")

    def test_rejects_when_no_target_language_track(self) -> None:
        streams = [
            {"index": 1, "language": "rus", "codec": "aac"},
            {"index": 2, "language": "deu", "codec": "aac"},
        ]
        with pytest.raises(DubResetError, match="No audio track is tagged"):
            identify_dubbed_stream_index(streams, "eng")

    def test_rejects_when_source_would_be_removed(self) -> None:
        streams = [
            {"index": 1, "language": "eng", "codec": "aac"},
            {"index": 2, "language": "eng", "codec": "aac"},
        ]
        with pytest.raises(DubResetError, match="Multiple audio tracks"):
            identify_dubbed_stream_index(streams, "eng", "rus")


class TestStripFirstAudioTrack:
    def test_deprecated_alias_is_disabled(self, tmp_path: Path) -> None:
        video = tmp_path / "lesson.mp4"
        video.write_bytes(b"fake")
        with pytest.raises(DubResetError, match="Unsafe track removal"):
            strip_first_audio_track(video)


class TestStripDubbedAudioTrack:
    def test_raises_when_ffmpeg_fails(self, tmp_path: Path) -> None:
        video = tmp_path / "lesson.mp4"
        video.write_bytes(b"fake")

        failed = MagicMock(returncode=1, stderr="boom", stdout="")
        with (
            patch(
                "app.services.dub_reset.probe_audio_streams",
                return_value=TWO_TRACK_STREAMS,
            ),
            patch("app.services.dub_reset.subprocess.run", return_value=failed),
            pytest.raises(DubResetError, match="ffmpeg failed"),
        ):
            strip_dubbed_audio_track(video, "eng", "rus")

        assert video.exists()
        assert not list(tmp_path.glob(".lesson.undub*"))

    def test_replaces_original_on_success(self, tmp_path: Path) -> None:
        video = tmp_path / "lesson.mp4"
        video.write_bytes(b"original")

        def fake_run(cmd, **_kwargs):
            out = Path(cmd[-1])
            out.write_bytes(b"stripped")
            return MagicMock(returncode=0, stderr="", stdout="")

        with (
            patch(
                "app.services.dub_reset.probe_audio_streams",
                side_effect=[TWO_TRACK_STREAMS, ONE_TRACK_STREAMS],
            ),
            patch("app.services.dub_reset.subprocess.run", side_effect=fake_run),
        ):
            removed = strip_dubbed_audio_track(video, "eng", "rus")

        assert removed == 1
        assert video.read_bytes() == b"stripped"

    @pytest.mark.skipif(which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_ffmpeg_drops_target_language_stream(self, tmp_path: Path) -> None:
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
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=rus",
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

        strip_dubbed_audio_track(video, "eng", "rus")

        after = get_video_info_with_duration(video)
        assert len(after["audio_streams"]) == 1
        assert normalize_lang_code(after["audio_streams"][0]["language"]) == "rus"


class TestBackupVideoBeforeReset:
    def test_creates_timestamped_backup(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        video = project / "lesson.mp4"
        video.write_bytes(b"original")

        backup = backup_video_before_reset(video, str(project), "project")

        assert backup.exists()
        assert backup.read_bytes() == b"original"
        assert backup.parent == project / ".redubber" / "backups"
        assert "pre-undub" in backup.name


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
    db.set_source_language_override(project_id, "rus")
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


class TestClearFinalizationArtifacts:
    def test_removes_dubbed_file_and_backup(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        video = project_dir / "lesson.mp4"
        video.write_bytes(b"fake")

        working_root = project_dir / ".redubber"
        rel_dir = working_root / "lesson.mp4"
        rel_dir.mkdir(parents=True)
        dubbed = rel_dir / "lesson.dubbed.mp4"
        dubbed.write_bytes(b"dubbed")
        backup_dir = working_root / "backups"
        backup_dir.mkdir()
        backup = backup_dir / "lesson.20250101.mp4"
        backup.write_bytes(b"backup")

        removed = clear_finalization_artifacts(video, str(project_dir), "project")

        assert not dubbed.exists()
        assert not backup.exists()
        assert str(dubbed) in removed
        assert str(backup) in removed


class TestReconcileVideoWithDisk:
    def test_clears_stale_backup_when_file_not_in_target_state(
        self, tmp_path: Path
    ) -> None:
        db, project_id, record, video, _srt = _seed_final_video(
            tmp_path, in_target_state=False
        )
        backup_dir = video.parent / ".redubber" / "backups"
        backup_dir.mkdir(parents=True)
        backup = backup_dir / "lesson.20250101.mp4"
        backup.write_bytes(b"backup")

        with patch("app.services.dub_reset.sync_video_metadata") as sync:
            result = reconcile_video_with_disk(
                db=db,
                project_id=project_id,
                video_record=record,
                project_path=str(video.parent),
                project_name="Demo",
                target_language="eng",
            )

        sync.assert_called_once()
        assert not backup.exists()
        assert result["reconciled"] is True
        assert any("backups" in fix or "backup" in fix for fix in result["fixes"])

    def test_removes_stale_target_subtitle_db_row(self, tmp_path: Path) -> None:
        db, project_id, record, video, srt = _seed_final_video(
            tmp_path, in_target_state=False
        )
        srt.unlink()
        db.add_subtitle_file(project_id, str(srt), srt.name, "eng")

        with patch("app.services.dub_reset.sync_video_metadata"):
            reconcile_video_with_disk(
                db=db,
                project_id=project_id,
                video_record=record,
                project_path=str(video.parent),
                project_name="Demo",
                target_language="eng",
            )

        assert db.get_subtitle_files_for_video(project_id, "lesson.mp4") == []


class TestResetDubbedVideo:
    def test_rejects_video_not_in_target_state(self, tmp_path: Path) -> None:
        db, project_id, record, video, _srt = _seed_final_video(
            tmp_path, in_target_state=False
        )
        backup_dir = video.parent / ".redubber" / "backups"
        backup_dir.mkdir(parents=True)
        backup = backup_dir / "lesson.20250101.mp4"
        backup.write_bytes(b"backup")

        with (
            patch(
                "app.services.dub_reset.probe_audio_streams",
                return_value=TWO_TRACK_STREAMS,
            ),
            patch("app.services.dub_reset.sync_video_metadata"),
            pytest.raises(DubResetError, match="final redubbed state"),
        ):
            reset_dubbed_video(
                db=db,
                project_id=project_id,
                video_record=record,
                project_path=str(video.parent),
                project_name="Demo",
                target_language="eng",
                source_language="rus",
            )

        assert not backup.exists()

    def test_deletes_generated_sub_and_strips_audio(self, tmp_path: Path) -> None:
        db, project_id, record, video, srt = _seed_final_video(tmp_path)
        working_srt = (
            video.parent / ".redubber" / "lesson.mp4" / "03_subtitles" / "lesson.en.srt"
        )
        working_srt.parent.mkdir(parents=True)
        working_srt.write_text("generated", encoding="utf-8")

        with (
            patch(
                "app.services.dub_reset.probe_audio_streams",
                return_value=TWO_TRACK_STREAMS,
            ),
            patch(
                "app.services.dub_reset.backup_video_before_reset",
                return_value=video.parent / ".redubber" / "backups" / "lesson.pre-undub.mp4",
            ),
            patch(
                "app.services.dub_reset.strip_dubbed_audio_track",
                return_value=1,
            ) as strip,
            patch("app.services.dub_reset.sync_video_metadata") as sync,
        ):
            result = reset_dubbed_video(
                db=db,
                project_id=project_id,
                video_record=record,
                project_path=str(video.parent),
                project_name="Demo",
                target_language="eng",
                source_language="rus",
            )

        strip.assert_called_once_with(video, "eng", "rus")
        assert sync.call_count == 2
        assert not srt.exists()
        assert not working_srt.exists()
        assert result["removed_audio_track"] is True
        assert result["removed_stream_index"] == 1
        assert str(srt) in result["deleted_subtitles"]
        assert str(working_srt) in result["deleted_subtitles"]
        remaining = db.get_subtitle_files_for_video(project_id, "lesson.mp4")
        assert remaining == []

    def test_clears_finalize_backup_artifacts(self, tmp_path: Path) -> None:
        db, project_id, record, video, _srt = _seed_final_video(tmp_path)
        backup_dir = video.parent / ".redubber" / "backups"
        backup_dir.mkdir(parents=True)
        backup = backup_dir / "lesson.20250101.mp4"
        backup.write_bytes(b"backup")

        with (
            patch(
                "app.services.dub_reset.probe_audio_streams",
                return_value=TWO_TRACK_STREAMS,
            ),
            patch(
                "app.services.dub_reset.backup_video_before_reset",
                return_value=backup_dir / "lesson.pre-undub.mp4",
            ),
            patch("app.services.dub_reset.strip_dubbed_audio_track", return_value=1),
            patch("app.services.dub_reset.sync_video_metadata"),
        ):
            reset_dubbed_video(
                db=db,
                project_id=project_id,
                video_record=record,
                project_path=str(video.parent),
                project_name="Demo",
                target_language="eng",
                source_language="rus",
            )

        assert not backup.exists()

    def test_rejects_when_disk_has_fewer_than_two_audio_streams(
        self, tmp_path: Path
    ) -> None:
        db, project_id, record, video, _srt = _seed_final_video(tmp_path)

        with (
            patch(
                "app.services.dub_reset.probe_audio_streams",
                return_value=ONE_TRACK_STREAMS,
            ),
            patch("app.services.dub_reset.sync_video_metadata"),
            pytest.raises(DubResetError, match="does not have two audio tracks"),
        ):
            reset_dubbed_video(
                db=db,
                project_id=project_id,
                video_record=record,
                project_path=str(video.parent),
                project_name="Demo",
                target_language="eng",
                source_language="rus",
            )
