"""
Streamlit application for audio-redub project management.
This application allows users to manage video projects with subtitles,
indexing files in a SQLite database for fast access.
"""

import streamlit as st
import os

from database import DatabaseManager
from file_scanner import FileScanner
from utils import detect_video_language, detect_subtitle_language, get_language_flag
from components.open import display_open_page
from components.project import display_current_project_page


def load_openai_config():
    """Load OpenAI configuration from file."""
    config_file = "openai_config.json"
    try:
        if os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
                return (config.get('token', ''),
                       config.get('model', ''),
                       config.get('voice', ''),
                       config.get('voice_instructions', ''))
    except Exception:
        pass
    return '', '', '', ''


def save_openai_config(token, model='', target_language='', voice='', voice_instructions=''):
    """Save OpenAI configuration to file."""
    config_file = "openai_config.json"
    try:
        import json
        config = {
            'token': token,
            'model': model,
            'target_language': target_language,
            'voice': voice,
            'voice_instructions': voice_instructions
        }
        with open(config_file, 'w') as f:
            json.dump(config, f)
    except Exception:
        pass


def load_target_language():
    """Load target language from config file."""
    config_file = "openai_config.json"
    try:
        if os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('target_language', '')
    except Exception:
        pass
    return ''


def save_target_language(target_language):
    """Save target language to config file."""
    config_file = "openai_config.json"
    try:
        import json
        config = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        config['target_language'] = target_language
        with open(config_file, 'w') as f:
            json.dump(config, f)
    except Exception:
        pass


def validate_and_get_models(api_token):
    """Validate OpenAI token and fetch available models."""
    try:
        import openai

        # Create client with the provided token
        client = openai.OpenAI(api_key=api_token)

        # Test the token by fetching models
        with st.spinner("Validating OpenAI token..."):
            models_response = client.models.list()
            models = [model.id for model in models_response.data if model.id.startswith(('gpt-', 'o1-'))]
            models.sort()

            if models:
                st.session_state.openai_models = models
                st.session_state.openai_token_valid = True
                st.success("✅ Token validated successfully!")

                # Model selection dropdown
                current_model = st.session_state.get('openai_model', models[0] if models else None)
                selected_model = st.selectbox(
                    "Select Model",
                    models,
                    index=models.index(current_model) if current_model in models else 0,
                    help="Choose the OpenAI model to use"
                )

                if selected_model != st.session_state.get('openai_model'):
                    st.session_state.openai_model = selected_model
                    # Save to file
                    save_openai_config(
                        st.session_state.get('openai_token', ''),
                        selected_model,
                        st.session_state.get('target_language', ''),
                        st.session_state.get('openai_voice', ''),
                        st.session_state.get('openai_voice_instructions', '')
                    )
                    st.rerun()

            else:
                st.error("No compatible models found")
                st.session_state.openai_token_valid = False

    except ImportError:
        st.error("OpenAI library not installed. Please install with: pip install openai")
        st.session_state.openai_token_valid = False
    except Exception as e:
        st.error(f"❌ Invalid token or API error: {str(e)}")
        st.session_state.openai_token_valid = False
        if 'openai_models' in st.session_state:
            del st.session_state.openai_models


def main():
    st.set_page_config(
        page_title="Redubber - Audio Redub Project Manager",
        page_icon="🎬",
        layout="wide"
    )

    
    # st.title("🎬 Redubber - Audio Redub Project Manager")
    
    # Initialize session state
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

    if 'current_project_path' not in st.session_state:
        st.session_state.current_project_path = None

    # Initialize page selection with persistence
    if 'current_page' not in st.session_state:
        # Try to load from query params first, then default to "Open"
        query_params = st.query_params
        saved_page = query_params.get("page", "Open")
        st.session_state.current_page = saved_page

        # If saved page is a project path, also restore current_project_path
        if saved_page != "Open" and os.path.isdir(saved_page):
            st.session_state.current_project_path = saved_page

    # Initialize temporary path for browsing
    if 'temp_browse_path' not in st.session_state:
        st.session_state.temp_browse_path = os.path.expanduser("~")

    # Initialize OpenAI configuration from saved file
    if 'openai_config_loaded' not in st.session_state:
        saved_token, saved_model, saved_voice, saved_voice_instructions = load_openai_config()
        saved_target_language = load_target_language()
        st.session_state.openai_token = saved_token
        st.session_state.openai_model = saved_model
        st.session_state.openai_voice = saved_voice
        st.session_state.openai_voice_instructions = saved_voice_instructions
        # Set default target language to English if not set
        st.session_state.target_language = saved_target_language if saved_target_language else 'eng'
        st.session_state.openai_config_loaded = True

    # Sidebar for configuration and navigation
    with st.sidebar:
        # Configuration section
        st.header("⚙️ Configuration")

        # OpenAI Configuration
        with st.expander("🤖 OpenAI Settings", expanded=False):
            # Text input with loaded value from file
            openai_token = st.text_input(
                "OpenAI API Token",
                type="password",
                value=st.session_state.get('openai_token', ''),
                help="Enter your OpenAI API token (persisted locally)"
            )

            # Handle token changes
            if openai_token != st.session_state.get('openai_token', ''):
                st.session_state.openai_token = openai_token

                # Save to file for persistence
                save_openai_config(
                    openai_token,
                    st.session_state.get('openai_model', ''),
                    st.session_state.get('target_language', ''),
                    st.session_state.get('openai_voice', ''),
                    st.session_state.get('openai_voice_instructions', '')
                )

                # Clear model selection when token changes
                if 'openai_model' in st.session_state:
                    del st.session_state.openai_model
                    save_openai_config(openai_token, '', '', '', '')  # Clear saved model too
                if 'openai_models' in st.session_state:
                    del st.session_state.openai_models

            # Validate token and get models
            if openai_token:
                validate_and_get_models(openai_token)

            # Voice Configuration
            st.divider()
            st.subheader("🎤 Voice Settings")

            # OpenAI TTS Voice options
            voice_options = {
                '': 'Select voice...',
                'alloy': 'Alloy',
                'echo': 'Echo',
                'fable': 'Fable',
                'onyx': 'Onyx',
                'nova': 'Nova',
                'shimmer': 'Shimmer'
            }

            current_voice = st.session_state.get('openai_voice', '')
            selected_voice = st.selectbox(
                "OpenAI TTS Voice",
                options=list(voice_options.keys()),
                format_func=lambda x: voice_options[x],
                index=list(voice_options.keys()).index(current_voice) if current_voice in voice_options else 0,
                help="Select the OpenAI Text-to-Speech voice for redubbing"
            )

            # Save voice when changed
            if selected_voice != st.session_state.get('openai_voice', ''):
                st.session_state.openai_voice = selected_voice
                save_openai_config(
                    st.session_state.get('openai_token', ''),
                    st.session_state.get('openai_model', ''),
                    st.session_state.get('target_language', ''),
                    selected_voice,
                    st.session_state.get('openai_voice_instructions', '')
                )

            # Voice Instructions
            voice_instructions = st.text_area(
                "Voice Instructions",
                value=st.session_state.get('openai_voice_instructions', ''),
                height=100,
                help="Custom instructions for voice generation (tone, style, emphasis, etc.)",
                placeholder="e.g., Speak in a calm, professional tone with clear enunciation..."
            )

            # Save instructions when changed
            if voice_instructions != st.session_state.get('openai_voice_instructions', ''):
                st.session_state.openai_voice_instructions = voice_instructions
                save_openai_config(
                    st.session_state.get('openai_token', ''),
                    st.session_state.get('openai_model', ''),
                    st.session_state.get('target_language', ''),
                    st.session_state.get('openai_voice', ''),
                    voice_instructions
                )

        # Redub Settings
        with st.expander("🎬 Redub Settings", expanded=False):
            # Fixed target language setting - English only for now

            # Display as read-only text instead of disabled selectbox
            st.text_input(
                "Target Language for Processing",
                value="English (eng)",
                disabled=True,
                help="Will be able to change soon... hopefully"
            )

            # Ensure target language is always English
            if st.session_state.get('target_language') != 'eng':
                st.session_state.target_language = 'eng'
                save_target_language('eng')

        st.divider()

        st.header("📍 Navigation")

        # Get all existing projects
        db_manager = st.session_state.db_manager
        existing_projects = db_manager.get_all_projects()

        # Open page
        if st.button("Open", use_container_width=True,
                    type="primary" if st.session_state.current_page == "Open" else "secondary"):
            st.session_state.current_page = "Open"
            st.query_params["page"] = "Open"
            st.rerun()

        # Project pages - show full path relative to home directory
        for project in existing_projects:
            project_path = project['path']
            home_path = os.path.expanduser("~")

            # Convert to relative path from home directory
            if project_path.startswith(home_path):
                # Remove home path and leading slash to get relative path
                relative_path = project_path[len(home_path):].lstrip(os.sep)
                display_path = relative_path if relative_path else "~"
            else:
                # If not under home directory, show full path
                display_path = project_path

            button_type = "primary" if st.session_state.current_page == project_path else "secondary"

            if st.button(display_path, use_container_width=True, type=button_type,
                        key=f"nav_{project['id']}", help=project_path):
                st.session_state.current_page = project_path
                st.session_state.current_project_path = project_path
                st.query_params["page"] = project_path
                st.rerun()

    # Main content area - show different pages
    if st.session_state.current_page == "Open":
        display_open_page()
    elif st.session_state.current_page != "Open":
        # This is a project path page
        display_current_project_page()


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
                            st.markdown(f"{get_language_flag(sub_file['language'])} `{sub_file['language']}`")
                        else:
                            st.markdown("❓ `unknown`")
                else:
                    st.markdown("❌ *No subtitles*")
            
            st.divider()






if __name__ == "__main__":
    main()