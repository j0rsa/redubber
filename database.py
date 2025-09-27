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