# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Redubber is a comprehensive audio redubbing system with two main interfaces:
1. **Streamlit Web UI** (`app.py`) - Project manager for indexing and managing video/subtitle files
2. **Core Redubbing Engine** (`redubber.py`) - Handles the actual AI-powered video redubbing workflow using OpenAI APIs

## Architecture

### Core Components

**Streamlit Web Application:**
- `app.py` - Main Streamlit application with UI logic and session management
- `components/open.py` - Project opening/loading interface
- `components/project.py` - Current project display and management

**Data Layer:**
- `database.py` - SQLite database operations via `DatabaseManager` class
- `file_scanner.py` - Recursive file system scanning via `FileScanner` class
- `utils.py` - Language detection utilities for video/subtitle files

**Video Processing:**
- `video_analyzer.py` - Video analysis using ffprobe for audio stream detection
- `redubber.py` - Core redubbing engine with OpenAI integration for transcription, translation, and TTS

**Examples:**
- `example.py` - Demonstrates batch redubbing workflow using the core engine

### Database Schema

The SQLite database (`redubber.db`) contains three main tables:
- `projects` - Project metadata with path, name, timestamps
- `video_files` - Video file records linked to projects with language detection
- `subtitle_files` - Subtitle file records linked to projects with language matching

### Redubbing Workflow

The core redubbing process in `redubber.py` follows this workflow:
1. **Extract audio** from video using ffmpeg
2. **Transcribe** audio to text using OpenAI Whisper API
3. **Translate** text using OpenAI GPT models
4. **Generate speech** from translated text using OpenAI TTS
5. **Mix** new audio track back into video with original audio streams

### Language Detection

Language detection works through two mechanisms:
1. Filename pattern matching using regex patterns for common language codes
2. Content-based detection for subtitles using the `langdetect` library (optional dependency)

## Common Commands

**Install dependencies:**
```bash
poetry install
```

**Run the Streamlit web application:**
```bash
poetry run streamlit run app.py
# or use the convenience script
./run.sh
# or use the poetry script
poetry run redubber
```

**Run the batch redubbing example:**
```bash
poetry run python example.py
```

**Add new dependencies:**
```bash
poetry add <package-name>
```

**Run in development shell:**
```bash
poetry shell
```

**Type checking (if mypy is available):**
```bash
poetry run mypy --ignore-missing-imports .
```

## Key Development Patterns

**Database Management:**
- Database operations use context managers (`with sqlite3.connect()`) for proper resource management
- All database queries use parameterized statements to prevent SQL injection
- Session state management in Streamlit maintains database connections and current project context

**File Processing:**
- File scanning is recursive using `Path.rglob()` to handle nested project structures
- Video-subtitle matching uses base filename pattern matching (removes extension and matches with LIKE)
- Supported file extensions are defined in `FileScanner` class constants

**Error Handling & Dependencies:**
- Language detection falls back gracefully when optional dependencies are missing
- Optional dependencies are handled with try/catch imports (e.g., `langdetect` for content-based language detection)
- External tool dependencies (ffmpeg/ffprobe) are called via subprocess with proper error handling

**Audio/Video Processing:**
- Video analysis uses ffprobe subprocess calls with JSON output parsing for audio stream detection
- Audio processing pipeline uses temporary files in `redubber_tmp/` directory
- OpenAI integration requires API token via environment variable `OPENAI_TOKEN` or `openai_config.json`
- Concurrent processing supported via ThreadPoolExecutor for batch operations

**Configuration:**
- OpenAI settings stored in `openai_config.json` (token, model, voice, instructions)
- Streamlit configuration in `.streamlit/config.toml`
- SpaCy language model (`en_core_web_sm`) used for natural language processing

## File Extensions

**Supported video formats:** mp4, avi, mkv, mov, wmv, flv, webm, m4v, mpg, mpeg, 3gp, ogv
**Supported subtitle formats:** srt, vtt, ass, ssa, sub, sbv, ttml, dfxp, stl, scc

## Development Environment

**Database:** SQLite database (`redubber.db`) is created automatically in the working directory

**Core Dependencies:**
- `streamlit` - Web application framework
- `openai` - AI-powered transcription, translation, and text-to-speech
- `spacy` + `en_core_web_sm` - Natural language processing
- `langdetect` - Content-based language detection (optional)

**External Tools:**
- **FFmpeg/ffprobe** - Required for video analysis and audio extraction
- **Python 3.13** - Exact version requirement in pyproject.toml

**Project Structure:**
- Root-level modules for core functionality
- `components/` package for Streamlit UI organization
- `redubber_tmp/` directory for temporary audio processing files
- Configuration files: `openai_config.json`, `.streamlit/config.toml`