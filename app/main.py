"""FastAPI application factory with lifespan management."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.infrastructure.task_queue import TaskQueueManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class _NoHealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_NoHealthCheckFilter())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan (startup and shutdown)."""
    # Startup
    logger.info("Starting Redubber API v%s", settings.api_version)

    # Initialize TaskQueueManager
    task_manager = TaskQueueManager(
        max_queue_size=settings.task_queue_max_size,
        max_workers=4,
    )
    await task_manager.start_workers(num_workers=settings.max_concurrent_redubs)
    app.state.task_manager = task_manager
    logger.info(
        "TaskQueueManager started with %d workers", settings.max_concurrent_redubs
    )

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down TaskQueueManager")
    await task_manager.stop_workers()
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="""
# Redubber API - AI-Powered Video Redubbing Platform

The Redubber API provides comprehensive video redubbing capabilities powered by OpenAI's
Whisper, GPT-4, and TTS technologies. Transform videos into different languages while
preserving the original speaker's tone, emotion, and delivery style.

## Key Features

- **Intelligent Voice Refinement**: AI-powered voice instruction generation for natural-sounding TTS
- **Multi-Voice Testing**: Generate previews for all available voices simultaneously
- **Smart Caching**: Avoid redundant TTS generations with intelligent caching
- **Async Processing**: High-performance async TTS with 5x throughput improvement
- **Project Management**: Organize videos, track progress, and manage redubbing workflows

## Workflow

1. Create a project and upload videos
2. Start redubbing task (transcribe, translate, generate audio)
3. Use voice refinement to test different voices and instructions
4. Select the best voice and apply to all segments
5. Download the final redubbed video

## Rate Limits

- OpenAI API limits apply to transcription, translation, and TTS operations
- Recommended: 1-3 concurrent redubbing tasks per instance
- TTS generation: Up to 100 concurrent requests (configurable)
        """.strip(),
        contact={
            "name": "Redubber Development Team",
            "email": "support@redubber.io",
            "url": "https://github.com/yourusername/redubber",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        openapi_tags=[
            {
                "name": "projects",
                "description": "Project management operations - create, list, update, and delete projects",
            },
            {
                "name": "videos",
                "description": "Video file operations - upload, analyze, and manage video files within projects",
            },
            {
                "name": "tasks",
                "description": "Redubbing task operations - start, monitor, and manage video redubbing tasks",
            },
            {
                "name": "voice-refinement",
                "description": "Voice refinement operations - generate AI-powered voice instructions, test multiple TTS voices, and select optimal voice settings for projects",
            },
            {
                "name": "settings",
                "description": "Tool-level application settings — AI model selection, workspace configuration, and automation preferences",
            },
            {
                "name": "health",
                "description": "Service health and status monitoring",
            },
        ],
        redirect_slashes=False,
    )

    # CORS middleware for frontend
    cors_origins = [origin.strip() for origin in settings.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers
    from app.api.routes import projects, tasks, videos, voice_refinement, filesystem
    from app.api.routes import settings as settings_routes

    app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
    app.include_router(tasks.router, prefix="/api", tags=["tasks"])
    app.include_router(videos.router, prefix="/api", tags=["videos"])
    app.include_router(
        voice_refinement.router, prefix="/api", tags=["voice-refinement"]
    )
    app.include_router(settings_routes.router, prefix="/api/settings", tags=["settings"])
    app.include_router(filesystem.router, prefix="/api/filesystem", tags=["filesystem"])

    @app.get(
        "/api/health",
        tags=["health"],
        summary="Health check",
        description="Returns service health status and version information",
        response_description="Service health status",
    )
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": settings.api_version}

    # Mount static files for React frontend (production only)
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        logger.info("Mounting static files from %s", static_path)

        # Serve static assets directory (JS bundles, CSS, images)
        assets_path = static_path / "assets"
        if assets_path.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

        # SPA catch-all: serve a real file if it exists, otherwise return index.html
        # so that React Router handles client-side navigation (e.g. /project/1).
        from fastapi.responses import FileResponse

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):
            file = static_path / full_path
            if file.is_file():
                return FileResponse(str(file))
            return FileResponse(str(static_path / "index.html"))

    else:
        logger.warning("Static files directory not found at %s", static_path)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],
        reload_excludes=["*.db", "*.log", "redubber_tmp/*", "storage/*"],
        log_level=settings.log_level.lower(),
    )
