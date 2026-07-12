# Redubber

AI-powered video redubbing — transcribes, translates, and re-voices video with OpenAI APIs. v2.0 delivers 5× faster processing via async TTS and ships as a React PWA + FastAPI service.

## Features

- **Full redubbing pipeline** — Whisper STT → GPT translation → TTS → ffmpeg mix-down
- **Async TTS** — up to 100 concurrent API calls, 5× faster than sequential
- **Voice refinement** — AI-guided voice selection with per-project instructions and cached previews
- **React PWA** — installable, works offline, real-time job progress
- **Project management** — multi-project SQLite store, auto file scanning, language detection
- **Docker-first** — single `docker-compose up` gets you running

---

## Quick Start (Docker)

**Prerequisites:** Docker 20.10+, OpenAI API key

```bash
# 1. Create storage directories and env file
mkdir -p storage
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 2. Start
docker-compose up -d

# 3. Open
open http://localhost:8000
```

Videos and the database persist in `./storage`. Logs: `docker-compose logs -f`.

---

## Development Setup

```bash
# Install all deps + git hooks
make install

# Start backend (port 8000) and frontend (port 5173) in parallel
make dev
```

Or individually:

```bash
make dev-backend   # FastAPI + uvicorn --reload
make dev-frontend  # Vite HMR
```

Frontend proxies `/api/*` to `localhost:8000` automatically.

### Other useful commands

```bash
make test          # Run backend tests (excludes integration)
make lint          # ruff check
make format        # ruff check --fix + ruff format
make build         # Production frontend build → frontend/dist/
make story         # Storybook component explorer (port 6006)
```

---

## Environment Variables

All variables are read at startup. Set them in `.env` (local dev) or pass them to the container.

### Required

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (`sk-...`) — can also be set via the UI Settings page and is then persisted to `settings.json` |

### Storage & Paths

| Variable | Default | Description |
|---|---|---|
| `REDUBBER_CONFIG_PATH` | _(empty)_ | Directory where `redubber.db` and `settings.json` are stored. **Set this to a mounted volume path in production** so the database and all UI settings (including your API key) survive container restarts. |

### Performance

| Variable | Default | Description |
|---|---|---|
| `MAX_CONCURRENT_REDUBS` | `1` | Max simultaneous redubbing jobs. Increase only if CPU/RAM allow — each job is already heavily parallelised internally |
| `TASK_QUEUE_MAX_SIZE` | `100` | Max queued jobs before new submissions are rejected |


### Logging & API

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Python log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `API_TITLE` | `Redubber API` | Title shown in `/api/docs` |
| `API_VERSION` | `2.0.0` | Version shown in `/api/docs` |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated allowed CORS origins |

### App-level Settings (UI-configurable)

These are stored in `settings.json` and editable via **Settings → ⚙** in the UI. They can also be seeded at startup via environment:

| Env Var | Default | Description |
|---|---|---|
| `REDUBBER_PROJECTS_ROOT` | _(empty)_ | Starting directory for the project file browser |
| `REDUBBER_WORKING_DIR` | _(empty)_ | Root where `.redubber/` working dirs are created |

---

## Deployment

### Docker Compose (recommended)

The included `docker-compose.yml` is production-ready for single-host deployments.

```bash
# Production run with explicit env
OPENAI_API_KEY=sk-... docker-compose up -d
```

**Persistent data** lives in `./storage` (mounted as `/mounted-storage` in the container). Back this up — it contains the database and dubbed output files.

**Resource limits** are set in `docker-compose.yml` (default: 2 CPU / 4 GB RAM). Tune based on video volume and concurrency needs.

### Docker image (GHCR)

Pre-built images are published on every push to `main`:

```bash
# Latest stable
docker pull ghcr.io/j0rsa/redubber:latest

# Pinned version
docker pull ghcr.io/j0rsa/redubber:v2.0.0
```

Run standalone:

```bash
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key \
  -v $(pwd)/storage:/mounted-storage \
  ghcr.io/j0rsa/redubber:latest
```

### Reverse proxy (nginx / Caddy)

The app serves the React frontend from `/` and the API from `/api`. A minimal nginx config:

```nginx
server {
    listen 443 ssl;
    server_name redubber.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        # Required for large video uploads
        client_max_body_size 4G;
        proxy_read_timeout 600s;
    }
}
```

### Health check

```bash
curl http://localhost:8000/api/health
# {"status":"ok","version":"2.0.0"}
```

---

## Architecture

```
┌──────────────────────────────────────────────┐
│  React PWA (Vite, TanStack Query, CSS Modules)│
│  served from /  by FastAPI StaticFiles        │
└─────────────────────┬────────────────────────┘
                      │ /api/*
┌─────────────────────▼────────────────────────┐
│  FastAPI (uvicorn)                            │
│  ├─ /api/projects    project CRUD             │
│  ├─ /api/redub       submit redub job         │
│  ├─ /api/tasks       job status & cancel      │
│  ├─ /api/settings    tool-level settings      │
│  └─ /api/projects/{id}/voice-*  refinement    │
│                                               │
│  TaskQueueManager (asyncio + ThreadPoolExecutor)
│  └─ Pipeline stages:                          │
│     1. Extract audio  (ffmpeg)                │
│     2. Transcribe     (Whisper / gpt-4o-transcribe)
│     3. Translate      (GPT-4o)                │
│     4. TTS            (gpt-4o-mini-tts, async)│
│     5. Assemble audio (ffmpeg)                │
│     6. Mix with video (ffmpeg)                │
│     7. Finalize       (validate, replace, cleanup)
└─────────────────────┬────────────────────────┘
                      │
┌─────────────────────▼────────────────────────┐
│  SQLite (redubber.db)                         │
│  projects · video_files · subtitle_files      │
│  voice_instruction_generations                │
│  tts_preview_cache · voice_selection_history  │
└──────────────────────────────────────────────┘
```

---

## Voice Refinement

Find the best AI voice for a project before committing to a full redub.

1. Open a project → click **Refine Voice**
2. Pick a representative segment (5–15 s)
3. **Analyse with AI** — GPT analyses tone, pace, accent, energy
4. Edit instructions if needed, or regenerate with feedback
5. **Preview all 6 voices** in ~10 s (parallel TTS, cached)
6. Select and **Save** — applied to all future TTS in this project

### Available voices

| Voice | Character | Best for |
|---|---|---|
| **Alloy** | Neutral, balanced | General purpose |
| **Echo** | Male, clear | Technical, instructional |
| **Fable** | British, expressive | Storytelling |
| **Onyx** | Deep, authoritative | News, reports |
| **Nova** | Warm, engaging | Education, friendly content |
| **Shimmer** | Soft, gentle | Calm, meditative content |

---

## Supported Formats

**Video:** mp4, avi, mkv, mov, wmv, flv, webm, m4v, mpg, mpeg, 3gp, ogv

**Subtitles:** srt, vtt, ass, ssa, sub, sbv, ttml, dfxp, stl, scc

---

## API docs

Interactive Swagger UI available at **`/api/docs`** when the server is running.
