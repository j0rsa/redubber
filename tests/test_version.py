"""Tests for package version resolution (single source of truth)."""

from __future__ import annotations

import tomllib
from pathlib import Path

from app.core.config import get_package_version, settings


def test_package_version_matches_pyproject() -> None:
    """Runtime version must match pyproject.toml — the only place to bump."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    expected = data["tool"]["poetry"]["version"]

    assert get_package_version() == expected
    assert settings.api_version == expected


def test_health_reports_pyproject_version(client) -> None:
    """UI badge reads /health — must report the pyproject version."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    expected = data["tool"]["poetry"]["version"]

    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["version"] == expected
