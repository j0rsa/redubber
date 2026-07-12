# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Redubber is an AI-powered video redubbing system. It transcribes, translates, and re-voices video files using OpenAI APIs. The service is a FastAPI backend + React PWA frontend, deployed via Docker.

## Architecture

### Backend (`app/`)

- `app/main.py` — FastAPI app factory, lifespan, CORS, static file serving
- `app/core/config.py` — Pydantic settings from environment variables
- `app/core/project_paths.py` — Working directory resolution per project
- `app/api/routes/` — Route modules: `projects`, `videos`, `tasks`, `settings`, `voice_refinement`, `filesystem`
- `app/infrastructure/task_queue.py` — Async job queue (asyncio + ThreadPoolExecutor), drives the full redub pipeline
- `app/services/` — `settings_service`, `tts_preview_generator`, `voice_instruction_generator`
- `app/schemas/` — Pydantic request/response models

### Core Engine (root-level, used by the backend)

- `redubber.py` — Full redubbing pipeline: extract audio → transcribe → translate → TTS → mix
- `reproj.py` — Per-video working directory and file layout management
- `database.py` — SQLite `DatabaseManager` (projects, videos, voice refinement tables)
- `file_scanner.py` — Recursive video/subtitle file detection
- `video_analyzer.py` — ffprobe-based audio stream detection
- `pipeline_status.py` — Pipeline step status tracking
- `seg_postprocessor.py` — Segment post-processing after STT
- `utils.py` — Language detection from filenames and content

### Frontend (`frontend/`)

React PWA (Vite, TypeScript, TanStack Query, CSS Modules). Source in `frontend/src/`:
- `pages/` — ProjectHub, ProjectDetail, NewProject, JobMonitor
- `components/` — FileGrid, PipelineStatus, VoiceRefinement, Settings, TasksPanel, etc.
- `hooks/` — useActiveTasks, useVoiceRefinement, useProjects, useVideos, useTasks, useSettings
- `types/` — Shared TypeScript interfaces

### Database Schema

SQLite (`redubber.db`) — created automatically on first run:
- `projects` — path, name, voice, voice_instructions, target/source language
- `video_files` — per-project video records with language detection
- `subtitle_files` — subtitle records matched to videos
- `voice_instruction_generations` — LLM voice analysis history
- `tts_preview_cache` — cached TTS preview audio (hash-keyed)
- `voice_selection_history` — audit trail of voice selections

### Redubbing Pipeline (11 stages)

1. Extract audio chunks (ffmpeg)
2. Transcribe (gpt-4o-transcribe / whisper-1)
3. Translate (GPT-4o)
4. Generate TTS segments (gpt-4o-mini-tts, async, up to 100 concurrent)
5. Assemble audio (ffmpeg)
6. Mix with video (ffmpeg)
7. Validate output
8. Replace original (if `auto_process` enabled)
9. Copy subtitles
10. Sync metadata to DB
11. Clean up temp files

## Common Commands

**Install all dependencies + git hooks:**
```bash
make install
```

**Run backend + frontend in parallel (dev mode):**
```bash
make dev
# or individually:
make dev-backend   # FastAPI on :8000
make dev-frontend  # Vite on :5173
```

**Run tests:**
```bash
make test          # excludes integration tests
poetry run pytest tests/ -m integration  # integration only
```

**Lint + format:**
```bash
make lint          # ruff check
make format        # ruff check --fix + ruff format
```

**Build frontend:**
```bash
make build
```

**Docker:**
```bash
docker-compose up -d
```

## Key Development Patterns

**Task queue:** Jobs are submitted via `POST /api/redub`, picked up by `TaskQueueManager` workers. Blocking pipeline stages run in a `ThreadPoolExecutor`; async TTS runs directly in the event loop.

**Working directories:** Each project gets a `.redubber/` subdirectory inside its folder (or under `REDUBBER_WORKING_DIR` if set). All pipeline artefacts live there.

**Settings:** Two-tier config:
1. `app/core/config.py` — infra settings from env vars (database path, concurrency, API keys)
2. `app/schemas/settings.py` / `app/services/settings_service.py` — user-facing settings persisted to `settings.json` (TTS model, voice, speed, etc.), editable from the UI

**Voice refinement:** Three-step flow: select segment → LLM analyses audio/text to generate voice instructions → parallel TTS previews for all 6 voices. Instructions and selection are stored per-project.

**Database:** All queries use parameterised statements. `DatabaseManager` is instantiated per-request from the `database_url` config value.

**File extensions:**
- Video: mp4, avi, mkv, mov, wmv, flv, webm, m4v, mpg, mpeg, 3gp, ogv
- Subtitles: srt, vtt, ass, ssa, sub, sbv, ttml, dfxp, stl, scc

## Environment Variables

See `README.md` for the full table. Required: `OPENAI_API_KEY`. Key optional vars: `DATABASE_URL`, `MOUNTED_STORAGE`, `REDUBBER_WORKING_DIR`, `MAX_CONCURRENT_REDUBS`.
