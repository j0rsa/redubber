"""
Streamlit application for audio-redub project management.
This application allows users to manage video projects with subtitles,
indexing files in a SQLite database for fast access.
"""

import streamlit as st
import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Optional
import time

from database import DatabaseManager
from file_scanner import FileScanner
from utils import detect_video_language, detect_subtitle_language


def main():
    st.set_page_config(
        page_title="Redubber - Audio Redub Project Manager",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Redubber - Audio Redub Project Manager")
    
    # Initialize session state
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    if 'current_project_path' not in st.session_state:
        st.session_state.current_project_path = None
    
    # Sidebar for project selection
    with st.sidebar:
        st.header("Project Management")
        
        # Folder selection
        project_path = st.text_input(
            "Project Folder Path",
            value=st.session_state.current_project_path or "",
            help="Enter the path to your video project folder"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load Project", type="primary"):
                if project_path and os.path.isdir(project_path):
                    st.session_state.current_project_path = project_path
                    load_project(project_path)
                    st.success("Project loaded!")
                else:
                    st.error("Invalid folder path")
        
        with col2:
            if st.button("Refresh Project"):
                if st.session_state.current_project_path:
                    refresh_project(st.session_state.current_project_path)
                    st.success("Project refreshed!")
                else:
                    st.error("No project loaded")
    
    # Main content area
    if st.session_state.current_project_path:
        display_project_files()
    else:
        st.info("👈 Please select a project folder from the sidebar to get started.")


def load_project(project_path: str):
    """Load a project by scanning the folder and indexing files."""
    scanner = FileScanner()
    db_manager = st.session_state.db_manager
    
    with st.spinner("Scanning project folder..."):
        video_files, subtitle_files = scanner.scan_folder(project_path)
        
        # Store project in database
        project_id = db_manager.add_project(project_path, os.path.basename(project_path))
        
        # Index video files
        for video_file in video_files:
            video_language = detect_video_language(video_file)
            db_manager.add_video_file(
                project_id, 
                str(video_file), 
                video_file.name, 
                video_language
            )
        
        # Index subtitle files
        for subtitle_file in subtitle_files:
            subtitle_language = detect_subtitle_language(subtitle_file)
            db_manager.add_subtitle_file(
                project_id,
                str(subtitle_file),
                subtitle_file.name,
                subtitle_language
            )


def refresh_project(project_path: str):
    """Refresh the current project by re-scanning the folder."""
    db_manager = st.session_state.db_manager
    
    # Remove existing project data
    db_manager.remove_project_by_path(project_path)
    
    # Re-load the project
    load_project(project_path)


def display_project_files():
    """Display the current project's video files with subtitle information."""
    db_manager = st.session_state.db_manager
    project_path = st.session_state.current_project_path
    
    project_data = db_manager.get_project_by_path(project_path)
    
    if not project_data:
        st.error("Project data not found. Please refresh the project.")
        return
    
    project_id = project_data['id']
    st.header(f"📁 {project_data['name']}")
    st.caption(f"Path: {project_path}")
    
    # Get video files for this project
    video_files = db_manager.get_video_files(project_id)
    
    if not video_files:
        st.info("No video files found in this project.")
        return
    
    # Display video files with subtitle information
    for video_file in video_files:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 2])
            
            with col1:
                st.write(f"🎥 **{video_file['filename']}**")
                st.caption(f"Path: {video_file['file_path']}")
            
            with col2:
                # Video language tag
                if video_file['language']:
                    st.markdown(f"🎬 `{video_file['language']}`")
                else:
                    st.markdown("🎬 `unknown`")
            
            with col3:
                # Find matching subtitle files
                subtitle_files = db_manager.get_subtitle_files_for_video(
                    project_id, 
                    video_file['filename']
                )
                
                if subtitle_files:
                    for sub_file in subtitle_files:
                        if sub_file['language']:
                            st.markdown(f"📝 `{sub_file['language']}`")
                        else:
                            st.markdown("📝 `unknown`")
                else:
                    st.markdown("📝 *No subtitles*")
            
            st.divider()


if __name__ == "__main__":
    main()