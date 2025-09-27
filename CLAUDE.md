# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Redubber is a Streamlit-based audio redub project manager that indexes video and subtitle files in project folders using SQLite for fast access. The application automatically detects languages from filenames and subtitle content.

## Architecture

The application follows a modular architecture with clear separation of concerns:

- `app.py` - Main Streamlit application with UI logic and session management
- `database.py` - SQLite database operations via `DatabaseManager` class
- `file_scanner.py` - File system scanning via `FileScanner` class
- `utils.py` - Language detection utilities for video/subtitle files

### Database Schema

The SQLite database (`redubber.db`) contains three main tables:
- `projects` - Project metadata with path, name, timestamps
- `video_files` - Video file records linked to projects with language detection
- `subtitle_files` - Subtitle file records linked to projects with language matching

### Language Detection

Language detection works through two mechanisms:
1. Filename pattern matching using regex patterns for common language codes
2. Content-based detection for subtitles using the `langdetect` library (optional dependency)

## Common Commands

**Install dependencies:**
```bash
poetry install
```

**Run the application:**
```bash
poetry run streamlit run app.py
```

**Alternative start method:**
```bash
./run.sh
```

**Add new dependencies:**
```bash
poetry add <package-name>
```

**Run in development shell:**
```bash
poetry shell
```

## Key Development Patterns

- Database operations use context managers (`with sqlite3.connect()`) for proper resource management
- File scanning is recursive using `Path.rglob()` to handle nested project structures
- Language detection falls back gracefully when optional dependencies are missing
- Session state management in Streamlit maintains database connections and current project context
- All database queries use parameterized statements to prevent SQL injection
- Video-subtitle matching uses base filename pattern matching (removes extension and matches with LIKE)
- Optional dependencies are handled with try/catch imports (e.g., `langdetect` for content-based language detection)

## File Extensions

**Supported video formats:** mp4, avi, mkv, mov, wmv, flv, webm, m4v, mpg, mpeg, 3gp, ogv
**Supported subtitle formats:** srt, vtt, ass, ssa, sub, sbv, ttml, dfxp, stl, scc

## Development Environment

**Database:** SQLite database (`redubber.db`) is created automatically in the working directory
**Dependencies:** Core dependencies are `streamlit` and `langdetect` (optional for content-based language detection)
**Project structure:** Single-level Python modules without packages - all `.py` files are in the root directory