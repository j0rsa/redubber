"""Tests for app.core.project_paths helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.core.project_paths import (
    get_project_working_dir,
    get_tts_previews_dir,
    sanitise_project_name,
)


# ---------------------------------------------------------------------------
# sanitise_project_name
# ---------------------------------------------------------------------------


class TestSanitiseProjectName:
    """Tests for sanitise_project_name."""

    def test_spaces_become_underscores(self) -> None:
        """Spaces in the basename should be replaced with underscores."""
        result = sanitise_project_name("/Users/john/My Cool Project")
        assert result == "my_cool_project"

    def test_lowercased(self) -> None:
        """Result must be lowercase."""
        result = sanitise_project_name("/storage/DocReview")
        assert result == "docreview"

    def test_special_characters_stripped(self) -> None:
        """Non-alphanumeric, non-underscore, non-hyphen chars are stripped."""
        result = sanitise_project_name("/storage/Cool Project!")
        assert result == "cool_project"

    def test_slashes_in_basename_not_present(self) -> None:
        """Only the basename is used — path separators in the final segment don't affect result."""
        result = sanitise_project_name("/some/path/doc-review")
        assert result == "doc-review"

    def test_hyphens_preserved(self) -> None:
        """Hyphens should be kept."""
        result = sanitise_project_name("/projects/my-project")
        assert result == "my-project"

    def test_empty_basename_falls_back_to_project(self) -> None:
        """When sanitised result is empty, fallback to 'project'."""
        # A basename of only special characters that are all stripped
        result = sanitise_project_name("/storage/!!!###")
        assert result == "project"

    def test_mixed_unicode_stripped(self) -> None:
        """Non-ASCII characters are stripped."""
        result = sanitise_project_name("/storage/über_cool")
        assert result == "ber_cool"

    def test_multiple_spaces_collapsed(self) -> None:
        """Multiple consecutive spaces are collapsed into a single underscore."""
        result = sanitise_project_name("/storage/my   project")
        assert result == "my_project"  # \s+ collapses runs of whitespace to one underscore


# ---------------------------------------------------------------------------
# get_project_working_dir
# ---------------------------------------------------------------------------


class TestGetProjectWorkingDir:
    """Tests for get_project_working_dir."""

    def test_uses_settings_working_directory_when_set(self, tmp_path: Path) -> None:
        """When settings.working_directory is non-empty, return <working_directory>/<sanitised_name>."""
        mock_settings = MagicMock()
        mock_settings.working_directory = str(tmp_path / "global_workdir")

        with patch(
            "app.core.project_paths.get_settings", return_value=mock_settings
        ):
            result = get_project_working_dir(
                "/Users/john/projects/My Project", "My Project"
            )

        expected = tmp_path / "global_workdir" / "my_project"
        assert result == expected

    def test_falls_back_to_dotredubber_when_working_directory_empty(
        self, tmp_path: Path
    ) -> None:
        """When settings.working_directory is empty, fall back to <project_path>/.redubber."""
        mock_settings = MagicMock()
        mock_settings.working_directory = ""

        with patch(
            "app.core.project_paths.get_settings", return_value=mock_settings
        ):
            result = get_project_working_dir(
                str(tmp_path / "my_video_project"), "My Video Project"
            )

        expected = tmp_path / "my_video_project" / ".redubber"
        assert result == expected

    def test_falls_back_to_dotredubber_when_settings_service_unavailable(
        self, tmp_path: Path
    ) -> None:
        """When the settings service import fails, fall back to .redubber gracefully."""
        with patch(
            "app.core.project_paths.get_settings",
            side_effect=ImportError("settings_service not built yet"),
        ):
            result = get_project_working_dir(
                str(tmp_path / "video_project"), "Video Project"
            )

        expected = tmp_path / "video_project" / ".redubber"
        assert result == expected

    def test_sanitises_project_path_basename_for_folder_name(
        self, tmp_path: Path
    ) -> None:
        """The folder inside working_directory uses the sanitised basename of project_path."""
        mock_settings = MagicMock()
        mock_settings.working_directory = str(tmp_path / "workdir")

        with patch(
            "app.core.project_paths.get_settings", return_value=mock_settings
        ):
            result = get_project_working_dir(
                "/videos/Awesome Series!", "Awesome Series!"
            )

        # sanitise_project_name("/videos/Awesome Series!") → "awesome_series"
        assert result.name == "awesome_series"


# ---------------------------------------------------------------------------
# get_tts_previews_dir
# ---------------------------------------------------------------------------


class TestGetTtsPreviewsDir:
    """Tests for get_tts_previews_dir."""

    def test_returns_tts_previews_subdirectory(self, tmp_path: Path) -> None:
        """Result should be tts_previews/ under the working dir."""
        mock_settings = MagicMock()
        mock_settings.working_directory = str(tmp_path / "workdir")

        with patch(
            "app.core.project_paths.get_settings", return_value=mock_settings
        ):
            result = get_tts_previews_dir(str(tmp_path / "my_project"), "my_project")

        assert result.name == "tts_previews"

    def test_creates_directory_on_disk(self, tmp_path: Path) -> None:
        """get_tts_previews_dir must create the directory if it doesn't exist."""
        mock_settings = MagicMock()
        mock_settings.working_directory = str(tmp_path / "workdir")

        with patch(
            "app.core.project_paths.get_settings", return_value=mock_settings
        ):
            result = get_tts_previews_dir(str(tmp_path / "my_project"), "my_project")

        assert result.is_dir(), f"Directory was not created at {result}"

    def test_fallback_path_also_creates_directory(self, tmp_path: Path) -> None:
        """Even in fallback mode the directory is created."""
        mock_settings = MagicMock()
        mock_settings.working_directory = ""

        project_path = tmp_path / "video_project"
        project_path.mkdir()

        with patch(
            "app.core.project_paths.get_settings", return_value=mock_settings
        ):
            result = get_tts_previews_dir(str(project_path), "video_project")

        assert result == project_path / ".redubber" / "tts_previews"
        assert result.is_dir()
