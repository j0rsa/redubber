"""Load generated subtitles and map them to source chunks and TTS files."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from app.core.project_paths import get_project_working_dir
from app.schemas.subtitle_review import (
    SubtitleReviewOriginalAudio,
    SubtitleReviewResponse,
    SubtitleReviewSegment,
)
from utils import convert_to_three_char_lang_code, detect_subtitle_language

SUBTITLE_EXTS = {".srt", ".vtt"}
AUDIO_EXTS = {".m4a", ".mp3", ".aac"}
_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


class SubtitleReviewError(Exception):
    """Raised when a generated subtitle cannot be loaded for review."""


def _hms_to_seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    ms = int((millis + "000")[:3])
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + ms / 1000.0


def parse_srt(content: str) -> list[tuple[float, float, str]]:
    """Parse SRT (or simple VTT-like) cues into (start, end, text) tuples."""
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip())
    cues: list[tuple[float, float, str]] = []
    for block in blocks:
        lines = [line for line in block.strip().split("\n") if line.strip() != "WEBVTT"]
        if len(lines) < 2:
            continue
        time_line = lines[0]
        text_from = 1
        if _TIME_RE.search(lines[0]) is None:
            time_line = lines[1]
            text_from = 2
        match = _TIME_RE.search(time_line)
        if not match:
            continue
        start = _hms_to_seconds(*match.group(1, 2, 3, 4))
        end = _hms_to_seconds(*match.group(5, 6, 7, 8))
        text = " ".join(line.strip() for line in lines[text_from:]).strip()
        if text and end > start:
            cues.append((start, end, text))
    return cues


def probe_duration(path: Path) -> float:
    """Return media duration in seconds, or 0 if ffprobe fails."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def _norm_lang(code: str | None) -> str:
    if not code:
        return ""
    converted = convert_to_three_char_lang_code(code.strip().lower())
    return (converted or "").lower()


def artefact_dirs(
    video_path: Path, project_path: str, project_name: str
) -> dict[str, Path]:
    """Return working-dir folders for this video (may not exist)."""
    working_dir = get_project_working_dir(project_path, project_name)
    rel = os.path.relpath(str(video_path), project_path)
    root = working_dir / rel
    return {
        "root": root,
        "chunks": root / "01_source_audio_chunks",
        "stt": root / "02_stt",
        "subtitles": root / "03_subtitles",
        "tts": root / "04_tts",
    }


def find_review_srt(
    video_path: Path,
    project_path: str,
    project_name: str,
    target_language: str,
) -> Path | None:
    """Locate the generated subtitle: working-dir copy first, then sidecar."""
    dirs = artefact_dirs(video_path, project_path, project_name)
    stem = video_path.stem
    candidates = [
        dirs["subtitles"] / f"{stem}.en.srt",
        video_path.parent / f"{stem}.en.srt",
    ]
    for path in candidates:
        if path.is_file():
            return path

    target = _norm_lang(target_language)
    if not target or not video_path.parent.is_dir():
        return None
    for candidate in sorted(video_path.parent.iterdir()):
        if not candidate.is_file() or candidate.suffix.lower() not in SUBTITLE_EXTS:
            continue
        if candidate.stem != stem and not candidate.stem.startswith(stem + "."):
            continue
        detected = detect_subtitle_language(candidate)
        if detected and _norm_lang(detected) == target:
            return candidate
    return None


def list_audio_files(directory: Path, prefix: str | None = None) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS
    ]
    if prefix:
        files = [path for path in files if path.name.startswith(prefix)]
    return sorted(files, key=lambda path: path.name)


def build_chunk_timeline(
    chunks_dir: Path, stem: str
) -> list[tuple[Path, float, float]]:
    """Map each source chunk to an absolute [start, end) window on the video timeline."""
    files = list_audio_files(chunks_dir, prefix=f"{stem}_")
    timeline: list[tuple[Path, float, float]] = []
    cursor = 0.0
    for path in files:
        duration = probe_duration(path)
        if duration <= 0:
            continue
        timeline.append((path, cursor, cursor + duration))
        cursor += duration
    return timeline


def chunk_for_time(
    timeline: list[tuple[Path, float, float]], at: float
) -> tuple[Path, float, float] | None:
    """Return (chunk, abs_start, abs_end) covering time ``at``."""
    for path, start, end in timeline:
        if start <= at < end or (at == end and path == timeline[-1][0]):
            return path, start, end
    return None


def tts_file_for_index(tts_dir: Path, index: int) -> Path | None:
    for ext in (".m4a", ".mp3", ".aac"):
        path = tts_dir / f"{index:03d}.en{ext}"
        if path.is_file():
            return path
    return None


def is_safe_chunk_name(name: str) -> bool:
    """Reject path traversal; allow spaces that appear in real video stems."""
    if not name or name != Path(name).name or ".." in name:
        return False
    return Path(name).suffix.lower() in AUDIO_EXTS


def build_subtitle_review(
    *,
    project_id: int,
    video_id: int,
    video_path: Path,
    filename: str,
    project_path: str,
    project_name: str,
    target_language: str,
    min_duration: float = 0.0,
    max_duration: float = 0.0,
) -> SubtitleReviewResponse:
    srt_path = find_review_srt(video_path, project_path, project_name, target_language)
    if srt_path is None:
        raise SubtitleReviewError("No generated subtitle file found for this video")

    try:
        cues = parse_srt(srt_path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        raise SubtitleReviewError(f"Could not read subtitle file: {exc}") from exc

    dirs = artefact_dirs(video_path, project_path, project_name)
    timeline = build_chunk_timeline(dirs["chunks"], video_path.stem)

    segments: list[SubtitleReviewSegment] = []
    any_tts = False
    for index, (start, end, text) in enumerate(cues):
        duration = round(end - start, 3)
        if min_duration > 0 and duration < min_duration:
            continue
        if max_duration > 0 and duration > max_duration:
            continue

        original: SubtitleReviewOriginalAudio | None = None
        hit = chunk_for_time(timeline, start)
        if hit is not None:
            chunk_path, chunk_start, chunk_end = hit
            seek_end = min(end - chunk_start, chunk_end - chunk_start)
            original = SubtitleReviewOriginalAudio(
                chunk_url=(
                    f"/api/projects/{project_id}/videos/{video_id}"
                    f"/subtitle-review/original/{chunk_path.name}"
                ),
                chunk_name=chunk_path.name,
                seek_start=round(max(0.0, start - chunk_start), 3),
                seek_end=round(max(0.0, seek_end), 3),
            )

        tts_path = tts_file_for_index(dirs["tts"], index)
        tts_url = None
        if tts_path is not None:
            any_tts = True
            tts_url = (
                f"/api/projects/{project_id}/videos/{video_id}"
                f"/subtitle-review/tts/{index}"
            )

        segments.append(
            SubtitleReviewSegment(
                index=index,
                start=round(start, 3),
                end=round(end, 3),
                duration=duration,
                text=text,
                original=original,
                tts_url=tts_url,
            )
        )

    return SubtitleReviewResponse(
        video_id=video_id,
        filename=filename,
        srt_path=str(srt_path),
        segments=segments,
        total=len(cues),
        returned=len(segments),
        has_chunks=bool(timeline),
        has_tts=any_tts,
    )
