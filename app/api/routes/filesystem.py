"""Filesystem browsing API endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from app.services.filesystem_search import search_directories

router = APIRouter()


class FileNode(BaseModel):
    name: str
    path: str
    type: str  # "file" | "directory"
    size: int | None = None


class DirectoryListing(BaseModel):
    path: str
    nodes: list[FileNode]


class DirectorySearchResponse(BaseModel):
    root: str
    query: str
    nodes: list[FileNode]


@router.get(
    "/browse",
    response_model=DirectoryListing,
    status_code=status.HTTP_200_OK,
    summary="List directory contents",
    description="Returns immediate children of a directory path for the file browser.",
    tags=["filesystem"],
)
async def browse_directory(
    path: Annotated[
        str,
        Query(description="Absolute path to the directory to list"),
    ] = "/",
) -> DirectoryListing:
    """List the contents of a directory.

    Returns only immediate children — not recursive. Hidden entries (starting
    with '.') are excluded. Symbolic links are resolved before stat.

    Args:
        path: Absolute path to browse. Defaults to filesystem root.

    Raises:
        HTTPException: 404 if path does not exist.
        HTTPException: 422 if path is not a directory.
        HTTPException: 403 if the directory is not readable.
    """
    target = Path(path).resolve()

    if not target.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Path does not exist: {path}",
        )

    if not target.is_dir():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Path is not a directory: {path}",
        )

    try:
        entries = list(target.iterdir())
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {path}",
        )

    nodes: list[FileNode] = []
    for entry in sorted(entries, key=lambda e: (e.is_file(), e.name.lower())):
        if entry.name.startswith("."):
            continue
        try:
            is_dir = entry.is_dir()
            size = None if is_dir else entry.stat().st_size
            nodes.append(
                FileNode(
                    name=entry.name,
                    path=str(entry),
                    type="directory" if is_dir else "file",
                    size=size,
                )
            )
        except (PermissionError, OSError):
            continue

    return DirectoryListing(path=str(target), nodes=nodes)


@router.get(
    "/search",
    response_model=DirectorySearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Fuzzy-search directories under a root path",
    description="Recursively finds folders whose names match the query using fuzzy matching.",
    tags=["filesystem"],
)
async def search_filesystem_directories(
    q: Annotated[str, Query(min_length=1, description="Folder name search query")],
    root: Annotated[str, Query(description="Root directory to search under")] = "/",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    max_depth: Annotated[int, Query(ge=1, le=12)] = 8,
) -> DirectorySearchResponse:
    """Return directories under ``root`` that fuzzy-match ``q``."""
    target = Path(root).resolve()

    if not target.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Path does not exist: {root}",
        )

    if not target.is_dir():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Path is not a directory: {root}",
        )

    try:
        hits = search_directories(
            target,
            q,
            limit=limit,
            max_depth=max_depth,
        )
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {root}",
        ) from None

    nodes = [
        FileNode(name=hit.name, path=hit.path, type="directory", size=None)
        for hit in hits
    ]
    return DirectorySearchResponse(root=str(target), query=q.strip(), nodes=nodes)
