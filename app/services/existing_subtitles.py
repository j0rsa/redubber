"""Reuse existing subtitle files instead of running STT / subtitle generation."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from openai.types.audio.transcription_segment import TranscriptionSegment

from app.core.project_paths import get_project_working_dir
from app.services.subtitle_review import parse_srt
from utils import detect_subtitle_language, normalize_lang_code

log = logging.getLogger(__name__)

SUBTITLE_EXTS = {
    ".srt",
    ".vtt",
    ".ass",
    ".ssa",
    ".sub",
    ".sbv",
    ".ttml",
    ".dfxp",
    ".stl",
    ".scc",
}

# Pipeline convention: generated / staged cues live at 03_subtitles/<stem>.en.srt
WORKDIR_SUBTITLE_NAME_SUFFIX = ".en.srt"

# Live-task progress after STT + subtitle generation (see docs/redubbing.md).
SUBTITLES_READY_PROGRESS = 38


def subtitle_filename_belongs_to_video(sub_filename: str, video_filename: str) -> bool:
    """True when a subtitle filename is a sidecar of a video filename.

    ``01.mp4`` matches ``01.srt`` / ``01.eng.srt``, but not ``010.eng.srt``
    or ``02.eng.srt``.
    """
    video_stem = Path(video_filename).stem
    sub_stem = Path(sub_filename).stem
    return sub_stem == video_stem or sub_stem.startswith(video_stem + ".")


def subtitle_belongs_to_video(sub_path: Path, video_path: Path) -> bool:
    """True when ``sub_path`` is a same-directory sidecar of ``video_path``."""
    try:
        same_dir = sub_path.parent.resolve() == video_path.parent.resolve()
    except OSError:
        same_dir = sub_path.parent == video_path.parent
    if not same_dir:
        return False
    return subtitle_filename_belongs_to_video(sub_path.name, video_path.name)


def iter_sidecar_subtitles(video_path: Path) -> list[Path]:
    """Return subtitle files sitting next to ``video_path`` that belong to it."""
    if not video_path.exists():
        return []
    parent = video_path.parent
    if not parent.is_dir():
        return []

    found: list[Path] = []
    for candidate in sorted(parent.iterdir()):
        if not candidate.is_file() or candidate.suffix.lower() not in SUBTITLE_EXTS:
            continue
        if not subtitle_belongs_to_video(candidate, video_path):
            continue
        found.append(candidate)
    return found


def external_subtitle_records(video_path: Path) -> list[dict]:
    """Sidecar subtitle dicts for ``video_analysis.subtitle_matches``."""
    records: list[dict] = []
    for sub in iter_sidecar_subtitles(video_path):
        records.append(
            {
                "language": detect_subtitle_language(sub) or "",
                "embedded": False,
                "path": str(sub),
                "filename": sub.name,
            }
        )
    return records


def subtitle_matches_language(
    sub_path: Path,
    language: str | None,
    *,
    include_unsuffixed: bool = True,
) -> bool:
    """Return True if ``sub_path`` matches ``language``.

    ``language=None`` matches every subtitle. Files with no language suffix are
    treated as the target language when ``include_unsuffixed`` is True (see
    :func:`utils.detect_subtitle_language`).
    """
    if language is None:
        return True
    target = normalize_lang_code(language)
    if not target:
        return False
    detected = detect_subtitle_language(sub_path)
    if not detected:
        return include_unsuffixed
    return normalize_lang_code(detected) == target


def find_sidecar_subtitles(
    video_path: Path,
    language: str | None = None,
    *,
    include_unsuffixed: bool = True,
) -> list[Path]:
    """Sidecar subtitle files for ``video_path``, optionally filtered by language."""
    return [
        path
        for path in iter_sidecar_subtitles(video_path)
        if subtitle_matches_language(
            path, language, include_unsuffixed=include_unsuffixed
        )
    ]


def workdir_subtitle_dest(
    video_path: Path, project_path: str, project_name: str
) -> Path:
    """Path where the pipeline expects the generated SRT for this video."""
    working_root = get_project_working_dir(project_path, project_name)
    rel = os.path.relpath(str(video_path), project_path)
    return (
        working_root
        / rel
        / "03_subtitles"
        / f"{video_path.stem}{WORKDIR_SUBTITLE_NAME_SUFFIX}"
    )


def find_reusable_subtitle(
    video_path: Path,
    *,
    project_path: str,
    project_name: str,
    language: str | None = None,
    include_unsuffixed: bool = True,
) -> Path | None:
    """Return the best existing subtitle to reuse, preferring the workdir copy."""
    dest = workdir_subtitle_dest(video_path, project_path, project_name)
    if dest.is_file():
        return dest

    subtitles_dir = dest.parent
    if subtitles_dir.is_dir():
        staged = sorted(
            p
            for p in subtitles_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".srt", ".vtt"}
        )
        if staged:
            return staged[0]

    sidecars = find_sidecar_subtitles(
        video_path, language, include_unsuffixed=include_unsuffixed
    )
    return sidecars[0] if sidecars else None


def copy_subtitle_to_workdir(src: Path, dest: Path) -> Path:
    """Copy ``src`` to the pipeline workdir path, creating parents as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            if dest.resolve() == src.resolve():
                return dest
        except OSError:
            return dest
        return dest
    shutil.copy2(src, dest)
    log.info("Staged existing subtitle %s → %s", src, dest)
    return dest


def stage_existing_subtitle(
    video_path: Path,
    *,
    project_path: str,
    project_name: str,
    language: str | None = None,
    include_unsuffixed: bool = True,
) -> Path | None:
    """Copy a matching existing subtitle into ``03_subtitles/`` if needed.

    Returns the workdir SRT path when a reusable subtitle was found, else None.
    """
    dest = workdir_subtitle_dest(video_path, project_path, project_name)
    if dest.is_file():
        return dest

    found = find_reusable_subtitle(
        video_path,
        project_path=project_path,
        project_name=project_name,
        language=language,
        include_unsuffixed=include_unsuffixed,
    )
    if found is None:
        return None
    if found.resolve() == dest.resolve():
        return dest
    return copy_subtitle_to_workdir(found, dest)


def segments_from_subtitle_file(path: Path) -> list[TranscriptionSegment]:
    """Parse an SRT/VTT file into OpenAI ``TranscriptionSegment`` objects for TTS."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log.warning("Could not read subtitle file %s", path)
        return []

    cues = parse_srt(content)
    segments: list[TranscriptionSegment] = []
    for index, (start, end, text) in enumerate(cues):
        segments.append(
            TranscriptionSegment(
                id=index,
                seek=int(start * 100),
                start=start,
                end=end,
                text=text,
                tokens=[],
                temperature=0.0,
                avg_logprob=0.0,
                compression_ratio=1.0,
                no_speech_prob=0.0,
            )
        )
    return segments


def stage_target_subtitles_for_videos(
    video_paths: list[Path],
    *,
    project_path: str,
    project_name: str,
    target_language: str | None,
) -> list[Path]:
    """Stage target-language sidecars into each video's ``03_subtitles/`` dir.

    Used by project file scans so disk-based pipeline progress reflects existing
    subs immediately.
    """
    staged: list[Path] = []
    for video_path in video_paths:
        dest = stage_existing_subtitle(
            video_path,
            project_path=project_path,
            project_name=project_name,
            language=target_language,
            include_unsuffixed=True,
        )
        if dest is not None:
            staged.append(dest)
    return staged
