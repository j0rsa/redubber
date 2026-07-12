"""Settings API endpoints for managing tool-level application configuration."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.schemas.settings import SettingsResponse, SettingsUpdate
from app.services import settings_service

router = APIRouter()


@router.get(
    "",
    response_model=SettingsResponse,
    summary="Get application settings",
    description=(
        "Returns the current tool-level settings. "
        "The OpenAI API key is **always masked** in the response: "
        'an empty string means no key is configured; `"sk-...xxxx"` shows the last 4 characters only.'
    ),
    responses={
        200: {
            "description": "Current settings returned successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "openai_api_key": "sk-...xYzW",
                        "tts_model": "tts-1",
                        "voice_analysis_model": "gpt-4o",
                        "default_voice": "nova",
                        "working_directory": "",
                        "auto_process": False,
                    }
                }
            },
        },
        500: {"description": "Unexpected server error while reading settings."},
    },
)
async def get_settings() -> SettingsResponse:
    """Retrieve current application settings.

    Loads settings from ``settings.json`` on disk. If the file does not
    exist yet, the response contains default values. The OpenAI API key
    field is masked in the response; the full key is never returned.

    Returns:
        Current settings with the API key masked.
    """
    try:
        return settings_service.get_settings()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load settings: {exc}",
        ) from exc


@router.put(
    "",
    response_model=SettingsResponse,
    summary="Update application settings",
    description=(
        "Merges the supplied fields into the current settings and persists them to disk. "
        "Only fields that are present in the request body are updated; "
        "all other settings retain their current values. "
        "The OpenAI API key is **stored in full** but **masked in the response**. "
        "\n\n"
        "**Validation rules:**\n"
        "- `tts_model` must be one of `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`.\n"
        "- `default_voice` must be one of `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`.\n"
        "- `working_directory` must be an existing directory path on disk, or an empty string."
    ),
    responses={
        200: {
            "description": "Settings updated and persisted successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "openai_api_key": "sk-...xYzW",
                        "tts_model": "tts-1-hd",
                        "voice_analysis_model": "gpt-4o",
                        "default_voice": "shimmer",
                        "working_directory": "/Users/alice/videos",
                        "auto_process": False,
                    }
                }
            },
        },
        422: {
            "description": (
                "Validation error — invalid field value. "
                "Common causes: unknown `tts_model`, unknown `default_voice`, "
                "or `working_directory` does not exist on disk."
            ),
        },
        500: {"description": "Unexpected server error while persisting settings."},
    },
)
async def update_settings(update: SettingsUpdate) -> SettingsResponse:
    """Update one or more application settings.

    Merges *update* into the currently persisted settings and writes the
    result back to ``settings.json``. Fields absent from the request body
    are left unchanged. The response contains the full updated settings
    with the API key masked.

    Args:
        update: Partial settings containing only the fields to change.

    Returns:
        Full updated settings with the API key masked.

    Raises:
        HTTPException: 422 if ``working_directory`` is set but does not exist on disk.
        HTTPException: 500 if settings cannot be persisted.
    """
    try:
        return settings_service.update_settings(update)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update settings: {exc}",
        ) from exc
