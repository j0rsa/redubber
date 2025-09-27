"""
Utility functions for language detection and file processing.
"""

import os
import re
from pathlib import Path
from typing import Optional
try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
    # Set seed for consistent results
    DetectorFactory.seed = 0
except ImportError:
    LANGDETECT_AVAILABLE = False


def detect_video_language(video_path: Path) -> Optional[str]:
    """
    Detect the language of a video file based on filename patterns.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Language code if detected, None otherwise
    """
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
    
    for pattern, lang_code in language_patterns.items():
        if re.search(pattern, filename):
            return lang_code
    
    return None


def detect_subtitle_language(subtitle_path: Path) -> Optional[str]:
    """
    Detect the language of a subtitle file.
    First tries filename patterns, then content-based detection if available.
    
    Args:
        subtitle_path: Path to the subtitle file
        
    Returns:
        Language code if detected, None otherwise
    """
    # First try filename pattern detection
    filename = subtitle_path.name.lower()
    
    # Common subtitle language patterns
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
    
    for pattern, lang_code in language_patterns.items():
        if re.search(pattern, filename):
            return lang_code
    
    # If filename pattern detection fails, try content-based detection
    if LANGDETECT_AVAILABLE:
        return detect_subtitle_content_language(subtitle_path)
    
    return None


def detect_subtitle_content_language(subtitle_path: Path) -> Optional[str]:
    """
    Detect language from subtitle file content using langdetect.
    
    Args:
        subtitle_path: Path to the subtitle file
        
    Returns:
        Language code if detected, None otherwise
    """
    if not LANGDETECT_AVAILABLE:
        return None
    
    try:
        with open(subtitle_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract text content from common subtitle formats
        text_content = extract_subtitle_text(content)
        
        if text_content and len(text_content.strip()) > 50:  # Need sufficient text
            detected_lang = detect(text_content)
            return detected_lang
    
    except (LangDetectException, UnicodeDecodeError, OSError):
        pass
    
    return None


def extract_subtitle_text(content: str) -> str:
    """
    Extract plain text from subtitle content, removing timestamps and formatting.
    
    Args:
        content: Raw subtitle file content
        
    Returns:
        Extracted text content
    """
    # Remove SRT timestamps and numbering
    content = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', content)
    
    # Remove VTT timestamps
    content = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', '', content)
    
    # Remove HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    
    # Remove ASS/SSA formatting
    content = re.sub(r'\{[^}]*\}', '', content)
    
    # Remove extra whitespace and line breaks
    content = re.sub(r'\n+', ' ', content)
    content = re.sub(r'\s+', ' ', content)
    
    return content.strip()


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    try:
        return file_path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"