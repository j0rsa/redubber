"""
Database management module for the Redubber application.
Handles SQLite operations for project indexing and file management.
"""

import sqlite3
import os
from typing import List, Dict, Optional
from pathlib import Path


class DatabaseManager:
    """Manages SQLite database operations for project and file indexing."""
    
    def __init__(self, db_path: str = "redubber.db"):
        self.db_path = db_path
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
                cursor.execute("ALTER TABLE video_analysis ADD COLUMN duration_seconds REAL DEFAULT 0")

            # Add source_language_override column if it doesn't exist (migration)
            try:
                cursor.execute("SELECT source_language_override FROM projects LIMIT 1")
            except sqlite3.OperationalError:
                # Column doesn't exist, add it
                cursor.execute("ALTER TABLE projects ADD COLUMN source_language_override TEXT DEFAULT ''")

            # Add voice column if it doesn't exist (migration)
            try:
                cursor.execute("SELECT voice FROM projects LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE projects ADD COLUMN voice TEXT DEFAULT ''")

            # Add voice_instructions column if it doesn't exist (migration)
            try:
                cursor.execute("SELECT voice_instructions FROM projects LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE projects ADD COLUMN voice_instructions TEXT DEFAULT ''")

            conn.commit()
    
    def add_project(self, path: str, name: str) -> int:
        """Add a new project or update existing one."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Try to insert, if exists, update the timestamp
            cursor.execute("""
                INSERT OR REPLACE INTO projects (path, name, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (path, name))
            
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
                
                # Delete associated video and subtitle files
                cursor.execute("DELETE FROM video_files WHERE project_id = ?", (project_id,))
                cursor.execute("DELETE FROM subtitle_files WHERE project_id = ?", (project_id,))
                
                # Delete the project
                cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
                
                conn.commit()
    
    def add_video_file(self, project_id: int, file_path: str, filename: str, language: Optional[str] = None):
        """Add a video file to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO video_files (project_id, file_path, filename, language)
                VALUES (?, ?, ?, ?)
            """, (project_id, file_path, filename, language))
            
            conn.commit()
    
    def add_subtitle_file(self, project_id: int, file_path: str, filename: str, language: Optional[str] = None):
        """Add a subtitle file to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO subtitle_files (project_id, file_path, filename, language)
                VALUES (?, ?, ?, ?)
            """, (project_id, file_path, filename, language))
            
            conn.commit()
    
    def get_video_files(self, project_id: int) -> List[Dict]:
        """Get all video files for a project."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM video_files 
                WHERE project_id = ?
                ORDER BY filename
            """, (project_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_subtitle_files_for_video(self, project_id: int, video_filename: str) -> List[Dict]:
        """Get subtitle files that match a video filename pattern."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Remove video extension and look for matching subtitle files
            base_name = os.path.splitext(video_filename)[0]
            
            cursor.execute("""
                SELECT * FROM subtitle_files 
                WHERE project_id = ? AND filename LIKE ?
                ORDER BY filename
            """, (project_id, f"{base_name}%"))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_projects(self) -> List[Dict]:
        """Get all projects in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM projects ORDER BY updated_at DESC")

            return [dict(row) for row in cursor.fetchall()]

    def save_project_scan(self, project_id: int, scan_data: str) -> None:
        """Save project scan results to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Remove existing scan data for this project
            cursor.execute("DELETE FROM project_scans WHERE project_id = ?", (project_id,))

            # Insert new scan data
            cursor.execute("""
                INSERT INTO project_scans (project_id, scan_data)
                VALUES (?, ?)
            """, (project_id, scan_data))

            conn.commit()

    def get_project_scan(self, project_id: int) -> Optional[str]:
        """Get project scan results from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT scan_data FROM project_scans
                WHERE project_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (project_id,))

            result = cursor.fetchone()
            return result[0] if result else None

    def has_project_scan(self, project_id: int) -> bool:
        """Check if project has scan results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM project_scans WHERE project_id = ?
            """, (project_id,))

            result = cursor.fetchone()
            return result[0] > 0 if result else False

    def save_video_analysis(self, project_id: int, video_data: Dict) -> None:
        """Save individual video analysis data."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO video_analysis
                (project_id, filename, file_path, size_mb, duration_seconds, audio_streams, subtitle_matches)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                project_id,
                video_data['filename'],
                video_data['path'],
                video_data['size_mb'],
                video_data['duration_seconds'],
                json.dumps(video_data['audio_streams']),
                json.dumps(video_data['subtitles'])
            ))

            conn.commit()

    def get_video_analysis(self, project_id: int) -> List[Dict]:
        """Get all video analysis data for a project."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM video_analysis
                WHERE project_id = ?
                ORDER BY filename
            """, (project_id,))

            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                # Parse JSON fields
                row_dict['audio_streams'] = json.loads(row_dict['audio_streams'])
                row_dict['subtitle_matches'] = json.loads(row_dict['subtitle_matches'])
                results.append(row_dict)

            return results

    def get_source_language_override(self, project_id: int) -> str:
        """Get the source language override for a project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT source_language_override FROM projects
                WHERE id = ?
            """, (project_id,))

            result = cursor.fetchone()
            return result[0] if result and result[0] else ''

    def set_source_language_override(self, project_id: int, language_override: str) -> None:
        """Set the source language override for a project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE projects
                SET source_language_override = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (language_override, project_id))

            conn.commit()

    def get_voice_settings(self, project_id: int) -> Dict[str, str]:
        """Get the voice settings for a project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT voice, voice_instructions FROM projects
                WHERE id = ?
            """, (project_id,))

            result = cursor.fetchone()
            if result:
                return {
                    'voice': result[0] if result[0] else '',
                    'voice_instructions': result[1] if result[1] else ''
                }
            return {'voice': '', 'voice_instructions': ''}

    def set_voice_settings(self, project_id: int, voice: str, voice_instructions: str) -> None:
        """Set the voice settings for a project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE projects
                SET voice = ?, voice_instructions = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (voice, voice_instructions, project_id))

            conn.commit()