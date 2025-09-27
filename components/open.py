"""
Open page for browsing and selecting project folders.
"""

import streamlit as st
import os
from pathlib import Path
from file_scanner import FileScanner


def count_video_files_in_current_folder(path):
    """Count video files only in the current folder (not subfolders)."""
    scanner = FileScanner()
    video_count = 0

    try:
        if not os.path.exists(path) or not os.path.isdir(path):
            return 0

        # Count video files only in current folder
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                if scanner.is_video_file(Path(item_path)):
                    video_count += 1

        return video_count

    except (PermissionError, OSError):
        return 0


def display_folder_tree():
    """Display folder tree with clickable folder links."""
    current_path = st.session_state.temp_browse_path

    try:
        if os.path.exists(current_path) and os.path.isdir(current_path):
            # Get list of directories only
            folders = []
            for item in os.listdir(current_path):
                item_path = os.path.join(current_path, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    folders.append((item, item_path))

            folders.sort(key=lambda x: x[0].lower())

            if folders:
                st.write("**Folders:**")
                for folder_name, folder_path in folders:
                    display_name = folder_name if len(folder_name) <= 20 else folder_name[:17] + "..."

                    # Create clickable folder link
                    if st.button(f"📁 {display_name}",
                               key=f"folder_{folder_path}",
                               help=f"Open {folder_name}",
                               use_container_width=True):
                        st.session_state.temp_browse_path = folder_path
                        st.rerun()
            else:
                st.caption("No folders found")
        else:
            st.error("Invalid directory")
            st.session_state.temp_browse_path = os.path.expanduser("~")

    except PermissionError:
        st.error("Access denied")
        st.session_state.temp_browse_path = os.path.expanduser("~")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.temp_browse_path = os.path.expanduser("~")


def display_open_page():
    """Display the Open page with file tree for browsing and selecting projects."""
    from app import load_project  # Import here to avoid circular imports

    st.header("📂 Open Project")

    # File tree for browsing and selecting projects
    st.subheader("Browse for Project Folder")

    # Current browsing path display
    st.text(f"📂 {st.session_state.temp_browse_path}")

    # Get video count for button logic (but don't display in path)
    video_count = count_video_files_in_current_folder(st.session_state.temp_browse_path)

    # Navigation controls
    col1, col2 = st.columns([2, 3])

    with col1:
        # Navigation buttons as separate, isolated buttons
        home_path = os.path.expanduser("~")
        parent_path = os.path.dirname(st.session_state.temp_browse_path)

        # Disable Up button at home directory - don't allow going to /Users or above
        can_go_up = (parent_path != st.session_state.temp_browse_path and
                     st.session_state.temp_browse_path != home_path)

        # Create three separate buttons in a row
        nav_col1, nav_col2, nav_col3 = st.columns(3)

        with nav_col1:
            if st.button("🏠", help="Home", use_container_width=True, key="nav_home"):
                st.session_state.temp_browse_path = home_path
                st.rerun()

        with nav_col2:
            if st.button("⬆️", help="Up", disabled=not can_go_up, use_container_width=True, key="nav_up"):
                st.session_state.temp_browse_path = parent_path
                st.rerun()

        with nav_col3:
            if st.button("🔄", help="Refresh", use_container_width=True, key="nav_refresh"):
                st.rerun()

    with col2:
        # Check if current folder has video files
        has_videos = video_count > 0

        button_help = f"Open this folder as a project ({video_count} video{'s' if video_count != 1 else ''})" if has_videos else "No video files found in this folder"

        if st.button("📁 Open as Project",
                    type="primary" if has_videos else "secondary",
                    disabled=not has_videos,
                    use_container_width=True,
                    help=button_help):
            st.session_state.current_project_path = st.session_state.temp_browse_path
            load_project(st.session_state.temp_browse_path)
            st.session_state.current_page = st.session_state.temp_browse_path
            st.query_params["page"] = st.session_state.temp_browse_path
            folder_name = os.path.basename(st.session_state.temp_browse_path) or "Root"
            st.success(f"Project loaded: {folder_name}")
            st.rerun()

    # File tree display
    display_folder_tree()

    # Show video count summary
    if video_count > 0:
        st.info(f"📹 Found {video_count} video file{'s' if video_count != 1 else ''} in this folder")
    else:
        st.warning("⚠️ No video files detected in this folder")