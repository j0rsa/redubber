"""
Video analysis module for extracting audio stream languages using ffprobe.
"""

import subprocess
import json
import os
from pathlib import Path
from typing import List, Dict, Optional


def get_video_info_with_duration(video_path: Path) -> Dict:
    """
    Extract video file information including duration and audio streams using ffprobe.

    Returns:
        Dictionary containing video info with duration and audio streams
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return {'duration_seconds': 0, 'audio_streams': []}

        data = json.loads(result.stdout)

        # Extract duration from format section
        duration_seconds = 0
        format_info = data.get('format', {})
        if 'duration' in format_info:
            try:
                duration_seconds = float(format_info['duration'])
            except (ValueError, TypeError):
                duration_seconds = 0

        # Extract audio streams
        audio_streams = []
        for i, stream in enumerate(data.get('streams', [])):
            if stream.get('codec_type') == 'audio':
                # Extract language from tags
                tags = stream.get('tags', {})
                language = None

                # Try different tag formats for language
                for lang_key in ['language', 'lang', 'Language', 'LANGUAGE']:
                    if lang_key in tags:
                        language = tags[lang_key]
                        break

                # If no language tag, try to detect from filename
                if not language:
                    language = detect_language_from_filename(video_path)

                audio_streams.append({
                    'index': i,
                    'language': language or 'unknown',
                    'codec': stream.get('codec_name', 'unknown'),
                    'channels': stream.get('channels', 'unknown'),
                    'sample_rate': stream.get('sample_rate', 'unknown')
                })

        return {
            'duration_seconds': duration_seconds,
            'audio_streams': audio_streams
        }

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return {'duration_seconds': 0, 'audio_streams': []}


def detect_language_from_filename(video_path: Path) -> Optional[str]:
    """Fallback language detection from filename patterns."""
    filename = video_path.name.lower()

    # Common language patterns in filenames
    language_patterns = {
        r'\.en\.|_en\.|english|eng': 'en',
        r'\.es\.|_es\.|spanish|esp': 'es',
        r'\.fr\.|_fr\.|french|fra': 'fr',
        r'\.de\.|_de\.|german|ger': 'de',
        r'\.it\.|_it\.|italian|ita': 'it',
        r'\.pt\.|_pt\.|portuguese|por': 'pt',
        r'\.ru\.|_ru\.|russian|rus': 'ru',
        r'\.ja\.|_ja\.|japanese|jpn': 'ja',
        r'\.ko\.|_ko\.|korean|kor': 'ko',
        r'\.zh\.|_zh\.|chinese|chi': 'zh',
        r'\.ar\.|_ar\.|arabic|ara': 'ar',
        r'\.hi\.|_hi\.|hindi|hin': 'hi',
    }

    import re
    for pattern, lang_code in language_patterns.items():
        if re.search(pattern, filename):
            return lang_code

    return None


def analyze_project_files(project_path: str, progress_callback=None) -> Dict:
    """
    Analyze all video and subtitle files in a project.

    Args:
        project_path: Path to project directory
        progress_callback: Function to call with progress updates

    Returns:
        Dictionary with analysis results
    """
    from file_scanner import FileScanner
    from utils import detect_subtitle_language

    scanner = FileScanner()
    project_path = Path(project_path)

    if progress_callback:
        progress_callback("Scanning files...")

    # Get all video and subtitle files
    video_files, subtitle_files = scanner.scan_folder(str(project_path))

    results = {
        'videos': [],
        'subtitles': [],
        'stats': {
            'total_videos': len(video_files),
            'total_subtitles': len(subtitle_files)
        }
    }

    # Analyze video files
    for i, video_file in enumerate(video_files):
        if progress_callback:
            progress_callback(f"Analyzing video {i+1}/{len(video_files)}: {video_file.name}")

        # Get video info including duration and audio streams
        video_info = get_video_info_with_duration(video_file)
        audio_streams = video_info['audio_streams']
        duration_seconds = video_info['duration_seconds']

        # Find matching subtitles
        base_name = video_file.stem
        matching_subtitles = []

        for sub_file in subtitle_files:
            if sub_file.name.startswith(base_name):
                sub_language = detect_subtitle_language(sub_file)
                matching_subtitles.append({
                    'filename': sub_file.name,
                    'path': str(sub_file),
                    'language': sub_language
                })

        results['videos'].append({
            'filename': video_file.name,
            'path': str(video_file),
            'audio_streams': audio_streams,
            'subtitles': matching_subtitles,
            'size_mb': round(video_file.stat().st_size / (1024 * 1024), 1),
            'duration_seconds': duration_seconds
        })

    # Analyze standalone subtitle files
    for subtitle_file in subtitle_files:
        # Check if this subtitle is already matched to a video
        already_matched = False
        for video_result in results['videos']:
            if any(sub['filename'] == subtitle_file.name for sub in video_result['subtitles']):
                already_matched = True
                break

        if not already_matched:
            sub_language = detect_subtitle_language(subtitle_file)
            results['subtitles'].append({
                'filename': subtitle_file.name,
                'path': str(subtitle_file),
                'language': sub_language
            })

    if progress_callback:
        progress_callback("Analysis complete!")

    return results