"""OpenAPI schema customization and utilities."""

from typing import Any, Dict

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """
    Generate customized OpenAPI schema with additional metadata.

    This function extends FastAPI's auto-generated OpenAPI schema with:
    - Custom server configurations
    - Security schemes (if authentication is added)
    - Additional examples
    - Custom extensions

    Args:
        app: FastAPI application instance

    Returns:
        Complete OpenAPI 3.0 schema dictionary
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
        servers=app.servers,
        terms_of_service=app.terms_of_service,
        contact=app.contact,
        license_info=app.license_info,
    )

    # Add custom server configurations
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Local development server",
        },
        {
            "url": "http://localhost:8000/api",
            "description": "Local development server (API prefix)",
        },
        {
            "url": "https://api.redubber.io",
            "description": "Production server",
        },
    ]

    # Add security schemes (placeholder for future authentication)
    # Uncomment and customize when authentication is implemented
    # openapi_schema["components"]["securitySchemes"] = {
    #     "BearerAuth": {
    #         "type": "http",
    #         "scheme": "bearer",
    #         "bearerFormat": "JWT",
    #         "description": "JWT token obtained from /auth/login endpoint",
    #     },
    #     "ApiKeyAuth": {
    #         "type": "apiKey",
    #         "in": "header",
    #         "name": "X-API-Key",
    #         "description": "API key for service-to-service authentication",
    #     },
    # }

    # Add global security requirement (placeholder)
    # openapi_schema["security"] = [{"BearerAuth": []}]

    # Add custom extensions
    openapi_schema["x-api-id"] = "redubber-api"
    openapi_schema["x-audience"] = "public"
    openapi_schema["x-logo"] = {
        "url": "https://redubber.io/logo.png",
        "altText": "Redubber Logo",
    }

    # Add additional info
    openapi_schema["info"]["x-api-lifecycle"] = "production"
    openapi_schema["info"]["x-api-stability"] = "stable"

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def get_openapi_tags_metadata() -> list[Dict[str, Any]]:
    """
    Get metadata for OpenAPI tags (endpoint groups).

    Returns:
        List of tag metadata dictionaries with name, description, and external docs
    """
    return [
        {
            "name": "projects",
            "description": "Project management operations - create, list, update, and delete projects",
            "externalDocs": {
                "description": "Project management guide",
                "url": "https://docs.redubber.io/projects",
            },
        },
        {
            "name": "videos",
            "description": "Video file operations - upload, analyze, and manage video files within projects",
            "externalDocs": {
                "description": "Video upload guide",
                "url": "https://docs.redubber.io/videos",
            },
        },
        {
            "name": "tasks",
            "description": "Redubbing task operations - start, monitor, and manage video redubbing tasks",
            "externalDocs": {
                "description": "Task management guide",
                "url": "https://docs.redubber.io/tasks",
            },
        },
        {
            "name": "voice-refinement",
            "description": """
Voice refinement operations - generate AI-powered voice instructions, test multiple TTS voices, and select optimal voice settings.

**Voice Refinement Workflow**:
1. **Analyze**: Generate voice instructions from transcription segment
2. **Refine**: (Optional) Regenerate instructions with user feedback
3. **Preview**: Generate TTS audio for all available voices
4. **Select**: Save the best voice and instructions to project

**Performance**:
- Voice instruction generation: 2-4 seconds (GPT-4o)
- Voice previews (6 voices): 2-5 seconds (async TTS, 5x faster than sequential)
- Smart caching reduces preview regeneration to <100ms for cache hits
            """.strip(),
            "externalDocs": {
                "description": "Voice refinement tutorial",
                "url": "https://docs.redubber.io/voice-refinement",
            },
        },
        {
            "name": "health",
            "description": "Service health and status monitoring",
        },
    ]


def add_custom_openapi_route(app: FastAPI) -> None:
    """
    Add custom OpenAPI schema endpoint to FastAPI app.

    This replaces the default OpenAPI schema generation with our custom version.

    Args:
        app: FastAPI application instance
    """

    def custom_openapi_route():
        return custom_openapi(app)

    app.openapi = custom_openapi_route  # type: ignore
