"""Project path resolution helpers using the settings system.

Provides filesystem-safe working directories for projects, falling back
to a legacy ``<project_path>/.redubber/`` layout when no global working
directory is configured.
"""

from __future__ import annotations

import re
from pathlib import Path


def get_settings():  # type: ignore[return]
    """Import and return the global settings instance.

    Thin wrapper that defers the import so that the settings service
    module is not required at import time (it may be built by a parallel
    agent).  Tests patch *this* function via
    ``app.core.project_paths.get_settings``.

    Returns:
        The application settings object from ``app.services.settings_service``.

    Raises:
        ImportError: If the settings service module is not yet available.
        Any error raised by ``get_settings`` from the service module.
    """
    from app.services.settings_service import get_settings as _get_settings

    return _get_settings()


def sanitise_project_name(project_path: str) -> str:
    """Derive a filesystem-safe folder name from the project path basename.

    Takes the last component of *project_path*, lowercases it, replaces
    whitespace with underscores, and strips any character that is not
    alphanumeric, an underscore, or a hyphen.

    Args:
        project_path: Absolute or relative path to the project directory.

    Returns:
        A non-empty, lowercase string suitable for use as a directory name.
        Falls back to ``"project"`` if the sanitised result is empty.

    Examples:
        >>> sanitise_project_name("/Users/john/My Videos/Cool Project!")
        'cool_project'
        >>> sanitise_project_name("/storage/projects/doc-review")
        'doc-review'
    """
    basename = Path(project_path).name
    lowered = basename.lower()
    # Replace all whitespace with underscores
    underscored = re.sub(r"\s+", "_", lowered)
    # Strip characters that are not alphanumeric, underscore, or hyphen
    cleaned = re.sub(r"[^a-z0-9_\-]", "", underscored)
    return cleaned or "project"


def get_project_working_dir(project_path: str, project_name: str) -> Path:
    """Return the working directory for a project.

    Resolution order:
    1. If ``settings.working_directory`` is set (non-empty), returns
       ``<working_directory>/<slug>/`` where slug is derived from the
       project's display name (not the folder name).
    2. Otherwise falls back to ``<project_path>/.redubber/``.

    The returned path is **not** created by this function; callers that need
    the directory to exist should call ``.mkdir(parents=True, exist_ok=True)``
    on the result.

    Args:
        project_path: Absolute path to the project's video directory.
        project_name: Display name of the project used to derive the slug.

    Returns:
        Resolved ``Path`` for the project's working directory.
    """
    working_directory = ""
    try:
        s = get_settings()
        working_directory = s.working_directory or ""
    except Exception:
        pass

    if working_directory:
        # Slug from display name, fall back to path basename if name is empty
        slug_source = project_name if project_name.strip() else project_path
        folder_name = sanitise_project_name(slug_source)
        return Path(working_directory) / folder_name

    # Backward-compatible fallback
    return Path(project_path) / ".redubber"


def get_tts_previews_dir(project_path: str, project_name: str) -> Path:
    """Return the TTS previews directory for a project, creating it if needed.

    The directory is always a ``tts_previews/`` subdirectory of the project's
    working directory as returned by :func:`get_project_working_dir`.

    Args:
        project_path: Absolute path to the project's video directory.
        project_name: Display name of the project.

    Returns:
        ``Path`` pointing to the ``tts_previews/`` directory.  The directory
        (and any missing parents) is created before returning.
    """
    previews_dir = get_project_working_dir(project_path, project_name) / "tts_previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    return previews_dir
