"""
Database management module for the Redubber application.
Handles SQLite operations for project indexing and file management.
"""

import sqlite3
import os
from typing import List, Dict, Optional


class DatabaseManager:
    """Manages SQLite database operations for project and file indexing."""

    def __init__(self, db_path: str = "redubber.db"):
        self.db_path = db_path
        import os as _os
        _os.makedirs(_os.path.dirname(_os.path.abspath(db_path)), exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Video files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    language TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
                )
            """)

            # Subtitle files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS subtitle_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    language TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
                )
            """)

            # Project scan results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    scan_data TEXT NOT NULL,
                    scan_status TEXT DEFAULT 'completed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
                )
            """)

            # Video analysis table - detailed video file analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    size_mb REAL,
                    duration_seconds REAL,
                    audio_streams TEXT,
                    subtitle_matches TEXT,
                    status TEXT DEFAULT 'analyzed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
                )
            """)

            # Add duration_seconds column if it doesn't exist (migration)
            try:
                cursor.execute("SELECT duration_seconds FROM video_analysis LIMIT 1")
            except sqlite3.OperationalError:
                # Column doesn't exist, add it
                cursor.execute(
                    "ALTER TABLE video_analysis ADD COLUMN duration_seconds REAL DEFAULT 0"
                )

            # Add source_language_override column if it doesn't exist (migration)
            try:
                cursor.execute("SELECT source_language_override FROM projects LIMIT 1")
            except sqlite3.OperationalError:
                # Column doesn't exist, add it
                cursor.execute(
                    "ALTER TABLE projects ADD COLUMN source_language_override TEXT DEFAULT ''"
                )

            # Add voice column if it doesn't exist (migration)
            try:
                cursor.execute("SELECT voice FROM projects LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE projects ADD COLUMN voice TEXT DEFAULT ''")

            # Add voice_instructions column if it doesn't exist (migration)
            try:
                cursor.execute("SELECT voice_instructions FROM projects LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute(
                    "ALTER TABLE projects ADD COLUMN voice_instructions TEXT DEFAULT ''"
                )

            # Add target_language column if it doesn't exist (migration)
            try:
                cursor.execute("SELECT target_language FROM projects LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute(
                    "ALTER TABLE projects ADD COLUMN target_language TEXT DEFAULT 'eng'"
                )

            # Voice instruction generations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voice_instruction_generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    segment_id TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    voice_instructions TEXT NOT NULL,
                    llm_model TEXT DEFAULT 'gpt-4o',
                    detected_characteristics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
                )
            """)

            # TTS preview cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tts_preview_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    voice_name TEXT NOT NULL,
                    voice_instructions_hash TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    audio_file_path TEXT NOT NULL,
                    audio_duration_ms INTEGER,
                    tts_model TEXT DEFAULT 'tts-1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
                )
            """)

            # Unique per (project, hash, voice) — different projects never share cache
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_tts_cache_unique
                ON tts_preview_cache(project_id, voice_instructions_hash, voice_name)
            """)

            # Create index for LRU cache eviction
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tts_cache_accessed
                ON tts_preview_cache(accessed_at)
            """)

            # Voice selection history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voice_selection_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    voice_name TEXT NOT NULL,
                    voice_instructions TEXT NOT NULL,
                    segment_used TEXT NOT NULL,
                    selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
                )
            """)

            # App settings table (singleton row, id=1 enforced)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    openai_api_key TEXT DEFAULT '',
                    openai_base_url TEXT DEFAULT '',
                    stt_model TEXT DEFAULT 'gpt-4o-transcribe',
                    tts_model TEXT DEFAULT 'gpt-4o-mini-tts',
                    voice_analysis_model TEXT DEFAULT 'o4-mini',
                    voice_analysis_audio_model TEXT DEFAULT 'gpt-audio-1',
                    default_voice TEXT DEFAULT 'nova',
                    tts_concurrency INTEGER DEFAULT 20,
                    openai_timeout REAL DEFAULT 60.0,
                    openai_retries INTEGER DEFAULT 3,
                    tts_speed REAL DEFAULT 1.25,
                    audio_chunk_duration INTEGER DEFAULT 900,
                    projects_root_path TEXT DEFAULT '',
                    working_directory TEXT DEFAULT '',
                    auto_process BOOLEAN DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def get_app_settings(self) -> Optional[Dict]:
        """Fetch the app settings row, or None if not yet saved."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM app_settings WHERE id = 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    def save_app_settings(self, settings_dict: Dict) -> None:
        """Upsert all app settings as a single row (id=1)."""
        fields = [k for k in settings_dict if k != "id"]
        placeholders = ", ".join(f"{f} = ?" for f in fields)
        values = [settings_dict[f] for f in fields]
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Insert row if missing, then update
            cursor.execute(
                "INSERT OR IGNORE INTO app_settings (id) VALUES (1)"
            )
            cursor.execute(
                f"UPDATE app_settings SET {placeholders}, updated_at = CURRENT_TIMESTAMP WHERE id = 1",
                values,
            )
            conn.commit()

    def add_project(self, path: str, name: str) -> int:
        """Add a new project or update existing one."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Try to insert, if exists, update the timestamp
            cursor.execute(
                """
                INSERT OR REPLACE INTO projects (path, name, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
                (path, name),
            )

            # Get the project ID
            cursor.execute("SELECT id FROM projects WHERE path = ?", (path,))
            project_id = cursor.fetchone()[0]

            conn.commit()
            return project_id

    def get_project_by_path(self, path: str) -> Optional[Dict]:
        """Get project information by path."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM projects WHERE path = ?", (path,))
            row = cursor.fetchone()

            return dict(row) if row else None

    def remove_project_by_path(self, path: str):
        """Remove a project and all associated files by path."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get project ID first
            cursor.execute("SELECT id FROM projects WHERE path = ?", (path,))
            result = cursor.fetchone()

            if result:
                project_id = result[0]

                # Explicit deletes — SQLite foreign keys are disabled by default
                for table in (
                    "video_files",
                    "subtitle_files",
                    "project_scans",
                    "video_analysis",
                    "tts_preview_cache",
                    "voice_instruction_generations",
                    "voice_selection_history",
                ):
                    cursor.execute(
                        f"DELETE FROM {table} WHERE project_id = ?", (project_id,)
                    )

                cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))

                conn.commit()

    def add_video_file(
        self,
        project_id: int,
        file_path: str,
        filename: str,
        language: Optional[str] = None,
    ):
        """Add a video file to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO video_files (project_id, file_path, filename, language)
                VALUES (?, ?, ?, ?)
            """,
                (project_id, file_path, filename, language),
            )

            conn.commit()

    def add_subtitle_file(
        self,
        project_id: int,
        file_path: str,
        filename: str,
        language: Optional[str] = None,
    ):
        """Add a subtitle file to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO subtitle_files (project_id, file_path, filename, language)
                VALUES (?, ?, ?, ?)
            """,
                (project_id, file_path, filename, language),
            )

            conn.commit()

    def get_video_files(self, project_id: int) -> List[Dict]:
        """Get all video files for a project."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM video_files 
                WHERE project_id = ?
                ORDER BY filename
            """,
                (project_id,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_subtitle_files_for_video(
        self, project_id: int, video_filename: str
    ) -> List[Dict]:
        """Get subtitle files that match a video filename pattern."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Remove video extension and look for matching subtitle files
            base_name = os.path.splitext(video_filename)[0]

            cursor.execute(
                """
                SELECT * FROM subtitle_files 
                WHERE project_id = ? AND filename LIKE ?
                ORDER BY filename
            """,
                (project_id, f"{base_name}%"),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_project_by_id(self, project_id: int) -> Optional[Dict]:
        """Get project by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_projects(self) -> List[Dict]:
        """Get all projects in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM projects ORDER BY id DESC")

            return [dict(row) for row in cursor.fetchall()]

    def save_project_scan(self, project_id: int, scan_data: str) -> None:
        """Save project scan results to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Remove existing scan data for this project
            cursor.execute(
                "DELETE FROM project_scans WHERE project_id = ?", (project_id,)
            )

            # Insert new scan data
            cursor.execute(
                """
                INSERT INTO project_scans (project_id, scan_data)
                VALUES (?, ?)
            """,
                (project_id, scan_data),
            )

            conn.commit()

    def get_project_scan(self, project_id: int) -> Optional[str]:
        """Get project scan results from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT scan_data FROM project_scans
                WHERE project_id = ?
                ORDER BY id DESC
                LIMIT 1
            """,
                (project_id,),
            )

            result = cursor.fetchone()
            return result[0] if result else None

    def has_project_scan(self, project_id: int) -> bool:
        """Check if project has scan results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT COUNT(*) FROM project_scans WHERE project_id = ?
            """,
                (project_id,),
            )

            result = cursor.fetchone()
            return result[0] > 0 if result else False

    def clear_project_files(self, project_id: int) -> None:
        """Delete all video files, subtitle files, and video analysis for a project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM video_files WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM subtitle_files WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM video_analysis WHERE project_id = ?", (project_id,))
            conn.commit()

    def save_video_analysis(self, project_id: int, video_data: Dict) -> None:
        """Save individual video analysis data."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO video_analysis
                (project_id, filename, file_path, size_mb, duration_seconds, audio_streams, subtitle_matches)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    project_id,
                    video_data["filename"],
                    video_data["path"],
                    video_data["size_mb"],
                    video_data["duration_seconds"],
                    json.dumps(video_data["audio_streams"]),
                    json.dumps(video_data["subtitles"]),
                ),
            )

            conn.commit()

    def get_video_analysis(self, project_id: int) -> List[Dict]:
        """Get all video analysis data for a project."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM video_analysis
                WHERE project_id = ?
                ORDER BY filename
            """,
                (project_id,),
            )

            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                # Parse JSON fields
                row_dict["audio_streams"] = json.loads(row_dict["audio_streams"])
                row_dict["subtitle_matches"] = json.loads(row_dict["subtitle_matches"])
                results.append(row_dict)

            return results

    def get_source_language_override(self, project_id: int) -> str:
        """Get the source language override for a project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT source_language_override FROM projects
                WHERE id = ?
            """,
                (project_id,),
            )

            result = cursor.fetchone()
            return result[0] if result and result[0] else ""

    def set_source_language_override(
        self, project_id: int, language_override: str
    ) -> None:
        """Set the source language override for a project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE projects
                SET source_language_override = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (language_override, project_id),
            )

            conn.commit()

    def get_voice_settings(self, project_id: int) -> Dict[str, str]:
        """Get the voice settings for a project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT voice, voice_instructions FROM projects
                WHERE id = ?
            """,
                (project_id,),
            )

            result = cursor.fetchone()
            if result:
                return {
                    "voice": result[0] if result[0] else "",
                    "voice_instructions": result[1] if result[1] else "",
                }
            return {"voice": "", "voice_instructions": ""}

    def set_voice_settings(
        self, project_id: int, voice: str, voice_instructions: str
    ) -> None:
        """Set the voice settings for a project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE projects
                SET voice = ?, voice_instructions = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (voice, voice_instructions, project_id),
            )

            conn.commit()

    def get_target_language(self, project_id: int) -> str:
        """Get the target language for dubbing output of a project.

        Args:
            project_id: Unique project identifier.

        Returns:
            ISO 639-2/B language code, defaulting to 'eng' when not set.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT target_language FROM projects
                WHERE id = ?
            """,
                (project_id,),
            )

            result = cursor.fetchone()
            return result[0] if result and result[0] else "eng"

    def set_target_language(self, project_id: int, language: str) -> None:
        """Set the target language for dubbing output of a project.

        Args:
            project_id: Unique project identifier.
            language: ISO 639-2/B language code (e.g. 'eng', 'spa', 'fra').
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE projects
                SET target_language = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (language, project_id),
            )

            conn.commit()

    # Voice Refinement Methods

    def save_voice_instruction_generation(
        self,
        project_id: int,
        segment_id: str,
        original_text: str,
        translated_text: str,
        voice_instructions: str,
        llm_model: str = "gpt-4o",
        detected_characteristics: Optional[str] = None,
    ) -> int:
        """Save a voice instruction generation result."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO voice_instruction_generations (
                    project_id, segment_id, original_text, translated_text,
                    voice_instructions, llm_model, detected_characteristics
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    project_id,
                    segment_id,
                    original_text,
                    translated_text,
                    voice_instructions,
                    llm_model,
                    json.dumps(detected_characteristics) if detected_characteristics else None,
                ),
            )

            generation_id = cursor.lastrowid
            conn.commit()
            return generation_id

    def get_voice_instruction_generations(
        self, project_id: int, limit: int = 10
    ) -> List[Dict]:
        """Get recent voice instruction generations for a project."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM voice_instruction_generations
                WHERE project_id = ?
                ORDER BY id DESC
                LIMIT ?
            """,
                (project_id, limit),
            )

            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                if row_dict["detected_characteristics"]:
                    row_dict["detected_characteristics"] = json.loads(
                        row_dict["detected_characteristics"]
                    )
                results.append(row_dict)

            return results

    def get_tts_cache(
        self,
        project_id: int,
        voice_name: str,
        voice_instructions_hash: str,
    ) -> Optional[Dict]:
        """Get cached TTS preview for a specific project, voice, and instructions hash."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM tts_preview_cache
                WHERE project_id = ? AND voice_instructions_hash = ? AND voice_name = ?
            """,
                (project_id, voice_instructions_hash, voice_name),
            )

            row = cursor.fetchone()
            if row:
                cursor.execute(
                    "UPDATE tts_preview_cache SET accessed_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (row["id"],),
                )
                conn.commit()
                return dict(row)

            return None

    def save_tts_cache(
        self,
        project_id: int,
        voice_name: str,
        voice_instructions_hash: str,
        translated_text: str,
        audio_file_path: str,
        audio_duration_ms: int,
        tts_model: str = "tts-1",
    ) -> int:
        """Save TTS preview to cache, scoped to the project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO tts_preview_cache (
                    project_id, voice_name, voice_instructions_hash,
                    translated_text, audio_file_path, audio_duration_ms, tts_model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    project_id,
                    voice_name,
                    voice_instructions_hash,
                    translated_text,
                    audio_file_path,
                    audio_duration_ms,
                    tts_model,
                ),
            )

            cache_id = cursor.lastrowid
            conn.commit()
            return cache_id  # type: ignore[return-value]

    def clear_tts_cache_for_project(self, project_id: int) -> List[str]:
        """Delete all TTS cache rows for a project and return their audio file paths for disk cleanup."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT audio_file_path FROM tts_preview_cache WHERE project_id = ?",
                (project_id,),
            )
            paths = [row["audio_file_path"] for row in cursor.fetchall()]

            cursor.execute(
                "DELETE FROM tts_preview_cache WHERE project_id = ?",
                (project_id,),
            )
            conn.commit()
            return paths

    def cleanup_old_tts_cache(self, days: int = 30, max_entries_per_project: int = 100):
        """Clean up old TTS cache entries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Delete entries older than N days
            cursor.execute(
                """
                DELETE FROM tts_preview_cache
                WHERE created_at < datetime('now', '-' || ? || ' days')
            """,
                (days,),
            )

            # Keep only last N entries per project (LRU)
            cursor.execute(
                """
                DELETE FROM tts_preview_cache
                WHERE id NOT IN (
                    SELECT id FROM tts_preview_cache
                    WHERE project_id = tts_preview_cache.project_id
                    ORDER BY accessed_at DESC
                    LIMIT ?
                )
            """,
                (max_entries_per_project,),
            )

            conn.commit()

    def save_voice_selection(
        self,
        project_id: int,
        voice_name: str,
        voice_instructions: str,
        segment_used: str,
    ) -> int:
        """Save voice selection history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO voice_selection_history (
                    project_id, voice_name, voice_instructions, segment_used
                )
                VALUES (?, ?, ?, ?)
            """,
                (project_id, voice_name, voice_instructions, segment_used),
            )

            history_id = cursor.lastrowid
            conn.commit()
            return history_id

    def get_voice_selection_history(
        self, project_id: int, limit: int = 10
    ) -> List[Dict]:
        """Get voice selection history for a project."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM voice_selection_history
                WHERE project_id = ?
                ORDER BY id DESC
                LIMIT ?
            """,
                (project_id, limit),
            )

            return [dict(row) for row in cursor.fetchall()]
