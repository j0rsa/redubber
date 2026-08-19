"""Project management API endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.core.dependencies import get_db, get_scanner
from app.schemas.models import (
    ProjectCreate,
    ProjectResponse,
    SourceLanguageUpdate,
    TargetLanguageUpdate,
    VoiceSettingsUpdate,
)
from app.services.project_scan import scan_project_files
from database import DatabaseManager
from file_scanner import FileScanner

router = APIRouter()


async def _scan_project_files(
    project_id: int, project_path: str, db: DatabaseManager, scanner: FileScanner
) -> None:
    """Background task to scan project directory and populate database."""
    scan_project_files(
        project_id,
        project_path,
        db,
        scanner,
        detect_source_language=True,
    )


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ProjectResponse)
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=ProjectResponse, include_in_schema=False)
async def create_project(
    project: ProjectCreate,
    background_tasks: BackgroundTasks,
    db: Annotated[DatabaseManager, Depends(get_db)],
    scanner: Annotated[FileScanner, Depends(get_scanner)],
) -> ProjectResponse:
    """Create or open a project and trigger file scan.

    Creates a new project entry in the database if it doesn't exist,
    or updates the timestamp if it already exists. Automatically triggers
    an asynchronous file scan to index video and subtitle files.

    Args:
        project: Project creation request with directory path.
        background_tasks: FastAPI background tasks manager.
        db: DatabaseManager dependency.
        scanner: FileScanner dependency.

    Returns:
        Complete project information including generated ID.

    Raises:
        HTTPException: 400 if path is invalid or inaccessible.
        HTTPException: 422 if path validation fails.
    """
    project_path = Path(project.path)

    # Validate path exists and is a directory
    if not project_path.exists():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Path does not exist: {project.path}",
        )

    if not project_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Path is not a directory: {project.path}",
        )

    project_name = (project.name or "").strip() or project_path.name
    project_id = db.add_project(path=str(project_path), name=project_name)

    # Trigger async file scan in background
    background_tasks.add_task(
        _scan_project_files, project_id, str(project_path), db, scanner
    )

    # Retrieve and return created project
    project_record = db.get_project_by_path(str(project_path))
    if not project_record:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create project",
        )

    # Apply default voice from global settings if the project has none yet
    try:
        from app.services.settings_service import get_settings as _get_settings
        _s = _get_settings()
        if _s.default_voice and not project_record.get("voice"):
            db.set_voice_settings(
                project_id=project_id,
                voice=_s.default_voice,
                voice_instructions="",
            )
            # Re-fetch so the response reflects the applied default
            project_record = db.get_project_by_path(str(project_path)) or project_record
    except Exception:
        pass  # settings service not yet available — skip silently

    return ProjectResponse(**project_record)


@router.get("", response_model=list[ProjectResponse])
@router.get("/", response_model=list[ProjectResponse], include_in_schema=False)
async def list_projects(
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> list[ProjectResponse]:
    """List all projects ordered by last update.

    Args:
        db: DatabaseManager dependency.

    Returns:
        List of all projects, most recently updated first.
    """
    projects = db.get_all_projects()
    return [ProjectResponse(**project) for project in projects]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> ProjectResponse:
    """Get detailed project information by ID.

    Args:
        project_id: Unique project identifier.
        db: DatabaseManager dependency.

    Returns:
        Complete project information.

    Raises:
        HTTPException: 404 if project not found.
    """
    project = db.get_project_by_id(project_id)
    if project:
        return ProjectResponse(**project)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Project {project_id} not found",
    )


@router.put("/{project_id}/voice-settings", response_model=ProjectResponse)
async def update_voice_settings(
    project_id: int,
    settings_update: VoiceSettingsUpdate,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> ProjectResponse:
    """Update TTS voice configuration for a project.

    Updates the voice identifier and custom instructions used for
    text-to-speech generation during redubbing.

    Args:
        project_id: Target project identifier.
        settings_update: New voice settings (voice + instructions fields).
        db: DatabaseManager dependency.

    Returns:
        Updated project information.

    Raises:
        HTTPException: 404 if project not found.
    """
    project = db.get_project_by_id(project_id)

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    instr = settings_update.get_instructions()
    db.set_voice_settings(
        project_id=project_id,
        voice=settings_update.voice,
        voice_instructions=instr,
    )

    if settings_update.segment_used:
        db.save_voice_selection(
            project_id=project_id,
            voice_name=settings_update.voice,
            voice_instructions=instr,
            segment_used=settings_update.segment_used,
        )

    # Return updated project
    project = db.get_project_by_id(project_id)
    if project:
        return ProjectResponse(**project)

    # Should never reach here
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to retrieve updated project",
    )


@router.put("/{project_id}/source-language", response_model=ProjectResponse)
async def update_source_language(
    project_id: int,
    language_update: SourceLanguageUpdate,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> ProjectResponse:
    """Update source language override for a project.

    Sets or clears the source language override. When set, all audio streams
    will be treated as this language during processing. This overrides the
    auto-detected language from audio stream metadata.

    Args:
        project_id: Target project identifier.
        language_update: New source language setting.
        db: DatabaseManager dependency.

    Returns:
        Updated project information.

    Raises:
        HTTPException: 404 if project not found.
    """
    # Verify project exists
    projects = db.get_all_projects()
    project_exists = any(p["id"] == project_id for p in projects)

    if not project_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Update source language
    db.set_source_language_override(
        project_id=project_id, language_override=language_update.source_language
    )

    # Return updated project
    project = db.get_project_by_id(project_id)
    if project:
        return ProjectResponse(**project)

    # Should never reach here
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to retrieve updated project",
    )


@router.put("/{project_id}/target-language", response_model=ProjectResponse)
async def update_target_language(
    project_id: int,
    language_update: TargetLanguageUpdate,
    db: Annotated[DatabaseManager, Depends(get_db)],
) -> ProjectResponse:
    """Update target language for dubbing output of a project.

    Sets the language that audio will be translated into during the redubbing
    pipeline. When set to 'eng', the pipeline uses Whisper's translation API
    directly. For any other language, transcription preserves the source
    language and a subsequent LLM step translates to the target.

    Args:
        project_id: Target project identifier.
        language_update: New target language setting.
        db: DatabaseManager dependency.

    Returns:
        Updated project information.

    Raises:
        HTTPException: 404 if project not found.
    """
    project = db.get_project_by_id(project_id)

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    db.set_target_language(
        project_id=project_id, language=language_update.target_language
    )

    # Return updated project
    project = db.get_project_by_id(project_id)
    if project:
        return ProjectResponse(**project)

    # Should never reach here
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to retrieve updated project",
    )


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: int,
    db: Annotated[DatabaseManager, Depends(get_db)],
):
    """Delete project and all associated metadata.

    Removes project entry, video analysis records, and subtitle references.
    Does not delete actual video files on disk.

    Args:
        project_id: Project to delete.
        db: DatabaseManager dependency.

    Raises:
        HTTPException: 404 if project not found.
    """
    # Find project by ID to get path and name
    projects = db.get_all_projects()
    project_record: dict | None = None

    for project in projects:
        if project["id"] == project_id:
            project_record = project
            break

    if not project_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    project_path = project_record["path"]

    # Remove working directory including TTS preview cache files.
    # Use the settings-aware helper so we clean the right directory regardless
    # of whether a global working_directory is configured.
    import shutil
    from app.core.project_paths import get_project_working_dir
    working_dir = get_project_working_dir(project_path, project_record["name"])
    if working_dir.exists():
        shutil.rmtree(working_dir, ignore_errors=True)

    # Delete project and all associated data (DB cascade removes tts_preview_cache rows)
    db.remove_project_by_path(project_path)
