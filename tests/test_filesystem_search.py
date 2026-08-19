"""Tests for fuzzy filesystem directory search."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.services.filesystem_search import fuzzy_score, search_directories


class TestFuzzyScore:
    def test_empty_query_returns_zero(self) -> None:
        assert fuzzy_score("", "Videos") == 0.0
        assert fuzzy_score("  ", "Videos") == 0.0

    def test_substring_match_scores_high(self) -> None:
        assert fuzzy_score("vid", "Videos") > fuzzy_score("vds", "Meetings")

    def test_fuzzy_subsequence_match(self) -> None:
        assert fuzzy_score("tut", "Tutorials") > 0
        assert fuzzy_score("xyz", "Tutorials") == 0.0

    def test_case_insensitive(self) -> None:
        assert fuzzy_score("MEET", "meetings") > 0


class TestSearchDirectories:
    def test_finds_matching_directories(self, tmp_path: Path) -> None:
        (tmp_path / "Videos").mkdir()
        (tmp_path / "Documents").mkdir()
        (tmp_path / "Videos" / "Tutorials").mkdir(parents=True)
        (tmp_path / "Videos" / "Meetings").mkdir(parents=True)

        hits = search_directories(tmp_path, "tut")

        assert len(hits) == 1
        assert hits[0].name == "Tutorials"
        assert hits[0].path.endswith("Videos/Tutorials")

    def test_respects_limit(self, tmp_path: Path) -> None:
        for index in range(5):
            (tmp_path / f"Folder_{index}").mkdir()

        hits = search_directories(tmp_path, "folder", limit=2)

        assert len(hits) == 2

    def test_empty_query_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "Videos").mkdir()
        assert search_directories(tmp_path, "") == []
        assert search_directories(tmp_path, "   ") == []

    def test_skips_hidden_directories(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "Visible").mkdir()

        hits = search_directories(tmp_path, "hid")

        assert hits == []


class TestFilesystemSearchEndpoint:
    def test_search_returns_matching_directories(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "Videos").mkdir()
        (tmp_path / "Videos" / "Tutorials").mkdir(parents=True)
        (tmp_path / "Videos" / "Meetings").mkdir(parents=True)

        response = client.get(
            "/api/filesystem/search",
            params={"q": "meet", "root": str(tmp_path)},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "meet"
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["name"] == "Meetings"
        assert data["nodes"][0]["type"] == "directory"

    def test_search_requires_query(self, client: TestClient, tmp_path: Path) -> None:
        (tmp_path / "Videos").mkdir()

        response = client.get(
            "/api/filesystem/search",
            params={"root": str(tmp_path)},
        )

        assert response.status_code == 422

    def test_search_root_not_found(self, client: TestClient) -> None:
        response = client.get(
            "/api/filesystem/search",
            params={"q": "test", "root": "/nonexistent/path/for/search"},
        )

        assert response.status_code == 404
