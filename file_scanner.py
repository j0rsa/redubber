"""
File scanner module for detecting video and subtitle files in project folders.
"""

import os
from pathlib import Path
from typing import List, Tuple


class FileScanner:
    """Scans project folders for video and subtitle files."""
    
    # Common video file extensions
    VIDEO_EXTENSIONS = {
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', 
        '.webm', '.m4v', '.mpg', '.mpeg', '.3gp', '.ogv'
    }
    
    # Common subtitle file extensions
    SUBTITLE_EXTENSIONS = {
        '.srt', '.vtt', '.ass', '.ssa', '.sub', '.sbv', 
        '.ttml', '.dfxp', '.stl', '.scc'
    }
    
    def __init__(self):
        pass
    
    def scan_folder(self, folder_path: str) -> Tuple[List[Path], List[Path]]:
        """
        Scan a folder for video and subtitle files.
        
        Args:
            folder_path: Path to the project folder
            
        Returns:
            Tuple of (video_files, subtitle_files) as Path objects
        """
        folder = Path(folder_path)
        
        if not folder.exists() or not folder.is_dir():
            return [], []
        
        video_files = []
        subtitle_files = []
        
        # Scan recursively through the folder
        for file_path in folder.rglob("*"):
            if file_path.is_file():
                extension = file_path.suffix.lower()
                
                if extension in self.VIDEO_EXTENSIONS:
                    video_files.append(file_path)
                elif extension in self.SUBTITLE_EXTENSIONS:
                    subtitle_files.append(file_path)
        
        # Sort files for consistent ordering
        video_files.sort(key=lambda x: x.name)
        subtitle_files.sort(key=lambda x: x.name)
        
        return video_files, subtitle_files
    
    def is_video_file(self, file_path: Path) -> bool:
        """Check if a file is a video file based on extension."""
        return file_path.suffix.lower() in self.VIDEO_EXTENSIONS
    
    def is_subtitle_file(self, file_path: Path) -> bool:
        """Check if a file is a subtitle file based on extension."""
        return file_path.suffix.lower() in self.SUBTITLE_EXTENSIONS
    
    def get_file_stats(self, folder_path: str) -> dict:
        """Get statistics about files in the folder."""
        video_files, subtitle_files = self.scan_folder(folder_path)
        
        return {
            'total_videos': len(video_files),
            'total_subtitles': len(subtitle_files),
            'video_extensions': set(f.suffix.lower() for f in video_files),
            'subtitle_extensions': set(f.suffix.lower() for f in subtitle_files)
        }