"""Load generated subtitles and map them to source chunks and TTS files."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from app.core.project_paths import get_project_working_dir
from app.schemas.subtitle_review import (
    SubtitleReviewFileOption,
    SubtitleReviewHallucinationWarning,
    SubtitleReviewOriginalAudio,
    SubtitleReviewResponse,
    SubtitleReviewSegment,
)
from stt_hallucination import (
    MIN_CONSECUTIVE_DUPLICATE_SEGMENTS,
    MIN_PHRASE_REPEAT_COUNT,
    _check_character_spam,
    _check_dominant_word,
    _check_known_phrases,
    _check_near_duplicate_run,
    _check_numbered_enumeration_loop,
    _check_segment_density,
    _check_transcript_density,
    _find_repeated_phrase,
    _normalize_text,
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


def _sidecar_matches_video(candidate: Path, stem: str) -> bool:
    if candidate.stem == stem:
        return True
    return candidate.stem.startswith(stem + ".")


def _file_option(path: Path, source: str) -> SubtitleReviewFileOption:
    return SubtitleReviewFileOption(
        path=str(path.resolve()),
        label=path.name,
        source=source,
    )


def list_review_srts(
    video_path: Path,
    project_path: str,
    project_name: str,
    target_language: str,
) -> list[SubtitleReviewFileOption]:
    """Return all subtitle files available for review, preferred order first."""
    dirs = artefact_dirs(video_path, project_path, project_name)
    stem = video_path.stem
    seen: set[str] = set()
    options: list[SubtitleReviewFileOption] = []

    def add(path: Path, source: str) -> None:
        key = str(path.resolve())
        if not path.is_file() or key in seen:
            return
        seen.add(key)
        options.append(_file_option(path, source))

    add(dirs["subtitles"] / f"{stem}.en.srt", "generated")
    add(video_path.parent / f"{stem}.en.srt", "sidecar")

    subtitles_dir = dirs["subtitles"]
    if subtitles_dir.is_dir():
        for candidate in sorted(subtitles_dir.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() in SUBTITLE_EXTS:
                if _sidecar_matches_video(candidate, stem):
                    add(candidate, "working_dir")

    if video_path.parent.is_dir():
        for candidate in sorted(video_path.parent.iterdir()):
            if not candidate.is_file() or candidate.suffix.lower() not in SUBTITLE_EXTS:
                continue
            if not _sidecar_matches_video(candidate, stem):
                continue
            add(candidate, "sidecar")

    target = _norm_lang(target_language)
    if target and video_path.parent.is_dir():
        for candidate in sorted(video_path.parent.iterdir()):
            if not candidate.is_file() or candidate.suffix.lower() not in SUBTITLE_EXTS:
                continue
            if not _sidecar_matches_video(candidate, stem):
                continue
            detected = detect_subtitle_language(candidate)
            if detected and _norm_lang(detected) == target:
                add(candidate, "sidecar")

    return options


def find_review_srt(
    video_path: Path,
    project_path: str,
    project_name: str,
    target_language: str,
) -> Path | None:
    """Locate the default generated subtitle: working-dir copy first, then sidecar."""
    options = list_review_srts(video_path, project_path, project_name, target_language)
    if not options:
        return None
    return Path(options[0].path)


def resolve_review_srt(
    video_path: Path,
    project_path: str,
    project_name: str,
    target_language: str,
    srt_path: str | None = None,
) -> Path:
    """Pick the subtitle file to review, validating an explicit path when given."""
    options = list_review_srts(video_path, project_path, project_name, target_language)
    if not options:
        raise SubtitleReviewError("No generated subtitle file found for this video")

    allowed = {option.path for option in options}
    if srt_path:
        resolved = str(Path(srt_path).resolve())
        if resolved not in allowed:
            raise SubtitleReviewError("Requested subtitle file is not available for this video")
        return Path(resolved)

    return Path(options[0].path)


def _finding_to_warning(
    finding_code: str,
    message: str,
    segment_index: int | None,
) -> SubtitleReviewHallucinationWarning:
    return SubtitleReviewHallucinationWarning(
        code=finding_code,
        message=message,
        segment_index=segment_index,
    )


def _duplicate_run_warnings(
    segments: list,
) -> list[SubtitleReviewHallucinationWarning]:
    """Flag every cue in a run of consecutive identical subtitles."""
    if len(segments) < MIN_CONSECUTIVE_DUPLICATE_SEGMENTS:
        return []

    warnings: list[SubtitleReviewHallucinationWarning] = []
    run_text = ""
    run_len = 0
    run_indices: list[int] = []

    def flush_run() -> None:
        nonlocal run_len, run_text, run_indices
        if run_len >= MIN_CONSECUTIVE_DUPLICATE_SEGMENTS and run_text:
            preview = run_text[:60] + ("…" if len(run_text) > 60 else "")
            message = (
                f"{run_len} consecutive segments repeat the same text: {preview!r}"
            )
            for index in run_indices:
                warnings.append(
                    _finding_to_warning(
                        "consecutive_duplicate_segments",
                        message,
                        index,
                    )
                )
        run_len = 0
        run_text = ""
        run_indices = []

    for index, segment in enumerate(segments):
        normalized = _normalize_text(segment.text or "")
        if not normalized:
            flush_run()
            continue
        if normalized == run_text:
            run_len += 1
            run_indices.append(index)
        else:
            flush_run()
            run_text = normalized
            run_len = 1
            run_indices = [index]
    flush_run()
    return warnings


def analyze_srt_hallucinations(
    cues: list[tuple[float, float, str]],
    *,
    source_label: str = "",
) -> list[SubtitleReviewHallucinationWarning]:
    """Run STT quality heuristics and attach warnings to every affected cue."""
    if not cues:
        return []

    try:
        from openai.types.audio.transcription_segment import TranscriptionSegment
    except ImportError:
        return []

    segments = [
        TranscriptionSegment(
            id=index,
            seek=0,
            start=start,
            end=end,
            text=text,
            tokens=[],
            temperature=0.0,
            avg_logprob=-0.3,
            compression_ratio=1.0,
            no_speech_prob=0.0,
        )
        for index, (start, end, text) in enumerate(cues)
    ]
    audio_duration = max(end for _, end, _ in cues)
    warnings: list[SubtitleReviewHallucinationWarning] = []

    for index, segment in enumerate(segments):
        text = segment.text or ""
        for finding in _check_known_phrases(text, index, None):
            warnings.append(
                _finding_to_warning(finding.code, finding.message, index)
            )
        for finding in _check_character_spam(text, index, None):
            warnings.append(
                _finding_to_warning(finding.code, finding.message, index)
            )
        for finding in _check_segment_density(segment, index, None):
            warnings.append(
                _finding_to_warning(finding.code, finding.message, index)
            )

        repeated = _find_repeated_phrase(text)
        if repeated:
            warnings.append(
                _finding_to_warning(
                    "repeated_phrase_loop",
                    (
                        f"phrase {repeated!r} repeats {MIN_PHRASE_REPEAT_COUNT}+ "
                        f"times in this cue"
                    ),
                    index,
                )
            )

    warnings.extend(_duplicate_run_warnings(segments))

    for finding in _check_numbered_enumeration_loop(segments, None):
        warnings.append(
            _finding_to_warning(finding.code, finding.message, finding.segment_index)
        )
    for finding in _check_near_duplicate_run(segments, None):
        warnings.append(
            _finding_to_warning(finding.code, finding.message, finding.segment_index)
        )

    for finding in _check_dominant_word(segments, None):
        warnings.append(
            _finding_to_warning(finding.code, finding.message, None)
        )
    for finding in _check_transcript_density(segments, audio_duration, None):
        warnings.append(
            _finding_to_warning(finding.code, finding.message, None)
        )

    deduped: list[SubtitleReviewHallucinationWarning] = []
    seen: set[tuple[str, int | None, str]] = set()
    for warning in warnings:
        key = (warning.code, warning.segment_index, warning.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(warning)

    return deduped


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
    srt_path: str | None = None,
) -> SubtitleReviewResponse:
    available_files = list_review_srts(
        video_path, project_path, project_name, target_language
    )
    selected = resolve_review_srt(
        video_path,
        project_path,
        project_name,
        target_language,
        srt_path=srt_path,
    )

    try:
        cues = parse_srt(selected.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        raise SubtitleReviewError(f"Could not read subtitle file: {exc}") from exc

    hallucination_warnings = analyze_srt_hallucinations(
        cues,
        source_label=selected.name,
    )

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
        srt_path=str(selected),
        available_files=available_files,
        segments=segments,
        total=len(cues),
        returned=len(segments),
        has_chunks=bool(timeline),
        has_tts=any_tts,
        hallucination_warnings=hallucination_warnings,
    )
