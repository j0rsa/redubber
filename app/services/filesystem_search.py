"""Fuzzy directory search for the project-creation file browser."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DirectorySearchHit:
    name: str
    path: str
    score: float


def fuzzy_score(query: str, text: str) -> float:
    """Score how well ``query`` fuzzy-matches ``text`` (higher is better)."""
    if not query:
        return 0.0

    q = query.lower().strip()
    t = text.lower()
    if not q or not t:
        return 0.0

    if q in t:
        return 100.0 + (len(q) / max(len(t), 1)) * 50.0

    qi = 0
    score = 0.0
    prev_match = -1
    for index, char in enumerate(t):
        if qi < len(q) and char == q[qi]:
            score += 1.0
            if prev_match == index - 1:
                score += 2.0
            if qi == 0 and index == 0:
                score += 5.0
            prev_match = index
            qi += 1

    if qi < len(q):
        return 0.0
    return score


def search_directories(
    root: Path,
    query: str,
    *,
    limit: int = 50,
    max_depth: int = 8,
) -> list[DirectorySearchHit]:
    """Walk ``root`` and return directories whose names fuzzy-match ``query``."""
    root = root.resolve()
    if not root.is_dir():
        return []

    q = query.strip()
    if not q:
        return []

    hits: list[DirectorySearchHit] = []
    root_depth = len(root.parts)

    for path in sorted(root.rglob("*")):
        if path.name.startswith("."):
            continue
        try:
            if not path.is_dir():
                continue
        except OSError:
            continue

        depth = len(path.parts) - root_depth
        if depth > max_depth:
            continue

        score = fuzzy_score(q, path.name)
        if score <= 0:
            continue

        hits.append(
            DirectorySearchHit(
                name=path.name,
                path=str(path),
                score=score,
            )
        )

    hits.sort(key=lambda hit: (-hit.score, hit.path.lower()))
    return hits[:limit]
