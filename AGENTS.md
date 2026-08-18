# AGENTS.md

## Cursor Cloud specific instructions

Redubber is a FastAPI backend (Python 3.13, Poetry) + React PWA frontend (Vite, npm) for AI video redubbing. See `README.md` and `CLAUDE.md` for architecture and the full command list; `Makefile` has all standard dev/test/lint/build targets.

### Environment notes (already provisioned in the VM snapshot; update script refreshes deps)
- Python 3.13 is required (`pyproject.toml` pins `>=3.13,<3.14`); the system default `python3` is 3.12, so the backend runs under Python 3.13 only. Poetry's virtualenv is already bound to 3.13 (`poetry env use python3.13`).
- Poetry is installed at `~/.local/bin/poetry` and is on `PATH` via `~/.bashrc`. In non-login shells, call it as `~/.local/bin/poetry` if `poetry` is not found.
- `ffmpeg`/`ffprobe` are installed system-wide and are required by the redub pipeline and file scan (duration/audio-stream probing).

### Running the app (dev mode)
- Standard command: `make dev` (runs backend on `:8000` and frontend on `:5173` in parallel). Individually: `make dev-backend`, `make dev-frontend`.
- The Vite dev server proxies `/api/*` to `http://localhost:8000`, so both must run for the full UI. Open `http://localhost:5173`.
- Backend logs `Static files directory not found at .../app/static` in dev — this is expected; the built frontend is only served by FastAPI in production (Docker). Ignore it in dev.

### Configuration gotcha (important)
- Do NOT `cp .env.example .env`. `.env.example` is stale and lists keys the current `Settings` model (`app/core/config.py`) rejects (`extra_forbidden`), which crashes backend startup. The model only accepts: `REDUBBER_CONFIG_PATH`, `OPENAI_API_KEY`, `MAX_CONCURRENT_REDUBS`, `TASK_QUEUE_MAX_SIZE`, `API_TITLE`, `API_VERSION`, `LOG_LEVEL`, `CORS_ORIGINS`.
- The backend runs fine with no `.env` at all (defaults). Settings are read from environment variables too, so prefer setting `OPENAI_API_KEY` as an environment variable/secret rather than an `.env` file.
- `OPENAI_API_KEY` is only needed to actually run the redub pipeline (transcribe/translate/TTS) or voice refinement. Project creation, file scanning, browsing, lint, and the test suites do NOT need it. The key can also be set at runtime via the app's Settings page (persisted to `redubber.db`).

### Testing / lint / build
- Backend tests: `poetry run pytest` (integration + `stale` tests are deselected by default via `pyproject.toml`; run integration with `poetry run pytest -m integration`). Backend lint: `poetry run ruff check .`.
- Frontend (from `frontend/`): typecheck `npx tsc --noEmit`, lint `npm run lint` (oxlint), tests `npx vitest run`, build `npm run build`.
- Frontend `vitest` runs Storybook component tests in a real browser via Playwright chromium. The chromium browser is preinstalled in the snapshot; if you hit "Executable doesn't exist" run `cd frontend && npx playwright install chromium`.
- The `pre-push` git hook (installed via `make install-hooks` / `scripts/hooks/pre-push`) runs the full gate: backend ruff + pytest, then frontend `tsc`, `oxlint`, `vitest`, and `build`. All of these pass in this environment.

### Hello-world sanity check
- Create a directory with a video, then create a project pointing at it (UI "New Project" or `POST /api/projects` with `{"path": "<dir>"}`). The background scan uses ffprobe to populate duration/size/audio-stream metadata. A tiny test clip can be generated with:
  `ffmpeg -f lavfi -i testsrc=duration=3:size=320x240:rate=15 -f lavfi -i sine=frequency=440:duration=3 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest sample_clip.mp4`
