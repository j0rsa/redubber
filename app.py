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
from redubber import extract_audio_sample, analyze_voice_with_gpt4o, generate_tts_sample


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
                       config.get('default_voice', 'nova'))
    except Exception:
        pass
    return '', '', 'nova'


def save_openai_config(token, model='', target_language='', default_voice='nova'):
    """Save OpenAI configuration to file."""
    config_file = "openai_config.json"
    try:
        import json
        config = {
            'token': token,
            'model': model,
            'target_language': target_language,
            'default_voice': default_voice
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
                        st.session_state.get('default_voice', 'nova')
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


def get_videos_with_segments(project_path: str) -> list:
    """Get list of videos that have segments generated."""
    videos_with_segments = []

    # Look for segment files in redubber_tmp
    tmp_dir = "redubber_tmp"
    if not os.path.exists(tmp_dir):
        return videos_with_segments

    # Walk through the tmp directory to find videos with segments
    for root, dirs, files in os.walk(tmp_dir):
        # Look for .seg files in 02_stt directories
        if "02_stt" in root:
            seg_files = [f for f in files if f.endswith('.seg')]
            if seg_files:
                # Extract video path from the directory structure
                # Format: redubber_tmp/relative_path/video_file/02_stt/
                parts = root.split(os.sep)
                stt_idx = parts.index("02_stt")
                video_name = parts[stt_idx - 1]  # Get the video folder name
                rel_path = os.sep.join(parts[1:stt_idx - 1])  # Get relative path

                # Construct full video path
                video_path = os.path.join(project_path, rel_path, video_name) if rel_path else os.path.join(project_path, video_name)

                # Find the actual video file (try common extensions)
                for ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
                    full_video_path = video_path.replace(os.path.splitext(video_path)[1], '') + ext if '.' in video_path else video_path + ext
                    if os.path.exists(full_video_path):
                        videos_with_segments.append({
                            'path': full_video_path,
                            'filename': os.path.basename(full_video_path),
                            'seg_dir': root,
                            'seg_files': seg_files
                        })
                        break

    return videos_with_segments


def parse_srt_to_segments(srt_path: str) -> list:
    """Parse an SRT file and return segment-like objects with start, end, text."""
    import re
    from dataclasses import dataclass

    @dataclass
    class SrtSegment:
        """Simple segment class matching TranscriptionSegment interface."""
        id: int
        start: float
        end: float
        text: str

    def parse_timestamp(ts: str) -> float:
        """Parse SRT timestamp (HH:MM:SS,mmm) to seconds."""
        match = re.match(r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})', ts.strip())
        if match:
            h, m, s, ms = match.groups()
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
        return 0.0

    segments = []
    try:
        with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Split into blocks (each subtitle entry)
        blocks = re.split(r'\n\s*\n', content.strip())

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Line 1: index number
                try:
                    idx = int(lines[0].strip())
                except ValueError:
                    continue

                # Line 2: timestamps
                ts_match = re.match(r'(.+?)\s*-->\s*(.+)', lines[1])
                if not ts_match:
                    continue

                start = parse_timestamp(ts_match.group(1))
                end = parse_timestamp(ts_match.group(2))

                # Remaining lines: text
                text = ' '.join(lines[2:]).strip()
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', '', text)

                if text:
                    segments.append(SrtSegment(id=idx, start=start, end=end, text=text))

    except Exception as e:
        print(f"Error parsing SRT file {srt_path}: {e}")

    return segments


def get_videos_with_subtitles(project_path: str) -> list:
    """Get list of videos that have external subtitle files."""
    from file_scanner import FileScanner

    scanner = FileScanner()
    video_files, subtitle_files = scanner.scan_folder(project_path)

    videos_with_subs = []
    for video_file in video_files:
        video_stem = video_file.stem
        video_dir = video_file.parent

        # Find matching subtitle files
        matching_subs = []
        for sub_file in subtitle_files:
            if sub_file.parent == video_dir:
                sub_stem = sub_file.stem
                if sub_stem == video_stem or sub_stem.startswith(video_stem + '.'):
                    matching_subs.append(str(sub_file))

        if matching_subs:
            videos_with_subs.append({
                'path': str(video_file),
                'filename': video_file.name,
                'subtitle_files': matching_subs,
                'has_generated_segments': False  # Will be updated below
            })

    # Check if any also have generated segments
    tmp_dir = "redubber_tmp"
    if os.path.exists(tmp_dir):
        for video in videos_with_subs:
            rel_path = os.path.relpath(video['path'], project_path)
            stt_dir = os.path.join(tmp_dir, rel_path, "02_stt")
            if os.path.exists(stt_dir):
                seg_files = [f for f in os.listdir(stt_dir) if f.endswith('.seg')]
                video['has_generated_segments'] = len(seg_files) > 0

    return videos_with_subs


def load_segments_for_video(video_path: str, project_path: str) -> list:
    """Load segments from segment files for a video."""
    from pydantic import TypeAdapter
    from openai.types.audio.transcription_segment import TranscriptionSegment

    # Construct the path to the segment files
    rel_path = os.path.relpath(video_path, project_path)
    stt_dir = os.path.join("redubber_tmp", rel_path, "02_stt")

    if not os.path.exists(stt_dir):
        return []

    all_segments = []
    seg_files = sorted([f for f in os.listdir(stt_dir) if f.endswith('.seg')])

    ta = TypeAdapter(list[TranscriptionSegment])
    for seg_file in seg_files:
        seg_path = os.path.join(stt_dir, seg_file)
        try:
            with open(seg_path, 'r') as f:
                segments = ta.validate_json(f.read())
                all_segments.extend(segments)
        except Exception as e:
            st.warning(f"Error loading segments from {seg_file}: {e}")

    return all_segments


@st.dialog("Voice Refinement", width="large")
def display_voice_refinement_modal():
    """Display the voice refinement modal for analyzing and previewing voice settings."""

    # Check prerequisites
    openai_token = st.session_state.get('openai_token', '')
    if not openai_token:
        st.error("OpenAI API token is not configured. Please set it in the OpenAI Settings.")
        if st.button("Close", use_container_width=True):
            st.session_state.show_voice_refinement_modal = False
            st.rerun()
        return

    project_path = st.session_state.get('current_project_path')
    if not project_path:
        st.error("No project selected.")
        if st.button("Close", use_container_width=True):
            st.session_state.show_voice_refinement_modal = False
            st.rerun()
        return

    # Get videos with subtitles (includes both generated segments and external subs)
    videos_with_subs = get_videos_with_subtitles(project_path)

    if not videos_with_subs:
        st.warning("No videos with subtitles found in this project.")
        st.info("Add subtitle files (.srt) or run the redub pipeline to generate segments.")
        if st.button("Close", use_container_width=True):
            st.session_state.show_voice_refinement_modal = False
            st.rerun()
        return

    # Get project voice settings from database
    db_manager = st.session_state.db_manager
    project_data = db_manager.get_project_by_path(project_path)
    project_id = project_data['id'] if project_data else None

    current_voice_settings = {'voice': '', 'voice_instructions': ''}
    if project_id:
        current_voice_settings = db_manager.get_voice_settings(project_id)

    # Initialize modal state from project settings
    if 'vr_voice' not in st.session_state:
        st.session_state.vr_voice = current_voice_settings.get('voice', '')
    if 'vr_instructions' not in st.session_state:
        st.session_state.vr_instructions = current_voice_settings.get('voice_instructions', '')
    if 'vr_original_sample' not in st.session_state:
        st.session_state.vr_original_sample = None
    if 'vr_generated_sample' not in st.session_state:
        st.session_state.vr_generated_sample = None
    if 'vr_segment_text' not in st.session_state:
        st.session_state.vr_segment_text = ''

    st.subheader("🎤 Voice Refinement")
    st.write("Analyze a video sample to find the best matching TTS voice and settings.")

    # Video selector - show source indicator
    def format_video_option(path):
        video = next((v for v in videos_with_subs if v['path'] == path), None)
        if video:
            if video['has_generated_segments']:
                return f"📊 {video['filename']}"  # Has generated segments
            else:
                return f"📄 {video['filename']}"  # External subs only
        return path

    video_options = {v['path']: v['filename'] for v in videos_with_subs}
    selected_video_path = st.selectbox(
        "Select Video",
        options=list(video_options.keys()),
        format_func=format_video_option,
        help="📊 = generated segments, 📄 = external subtitles"
    )

    # Get selected video info
    selected_video = next((v for v in videos_with_subs if v['path'] == selected_video_path), None)

    # Load segments - try generated segments first, then fall back to SRT
    segments = []
    segment_source = None

    if selected_video and selected_video['has_generated_segments']:
        segments = load_segments_for_video(selected_video_path, project_path)
        segment_source = "generated"

    if not segments and selected_video and selected_video['subtitle_files']:
        # Fall back to parsing external SRT
        srt_path = selected_video['subtitle_files'][0]  # Use first subtitle file
        segments = parse_srt_to_segments(srt_path)
        segment_source = "srt"

    if not segments:
        st.error("Could not load segments for this video.")
        return

    # Show segment source
    if segment_source == "srt":
        st.info(f"📄 Using external subtitle: {os.path.basename(selected_video['subtitle_files'][0])}")
    else:
        st.success("📊 Using generated segments")

    # Segment index selector
    col1, col2 = st.columns([1, 2])
    with col1:
        segment_index = st.number_input(
            "Segment Index",
            min_value=0,
            max_value=len(segments) - 1,
            value=0,
            help=f"Select segment to analyze (0 - {len(segments) - 1})"
        )
    with col2:
        if segment_index < len(segments):
            seg = segments[segment_index]
            st.text(f"Time: {seg.start:.1f}s - {seg.end:.1f}s ({seg.end - seg.start:.1f}s)")
            st.text(f"Text: {seg.text[:100]}..." if len(seg.text) > 100 else f"Text: {seg.text}")

    st.divider()

    # Voice settings (editable)
    voice_options = {
        '': 'Select voice...',
        'alloy': 'Alloy - Neutral, balanced',
        'echo': 'Echo - Warm, conversational male',
        'fable': 'Fable - British, expressive',
        'onyx': 'Onyx - Deep, authoritative male',
        'nova': 'Nova - Friendly, upbeat female',
        'shimmer': 'Shimmer - Warm, gentle female'
    }

    selected_voice = st.selectbox(
        "Voice",
        options=list(voice_options.keys()),
        format_func=lambda x: voice_options[x],
        index=list(voice_options.keys()).index(st.session_state.vr_voice) if st.session_state.vr_voice in voice_options else 0
    )
    st.session_state.vr_voice = selected_voice

    voice_instructions = st.text_area(
        "Voice Instructions",
        value=st.session_state.vr_instructions,
        height=100,
        help="Instructions for TTS voice style (tone, pace, emotion, etc.)"
    )
    st.session_state.vr_instructions = voice_instructions

    st.divider()

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Analyze", use_container_width=True, help="Analyze the segment audio with AI"):
            with st.spinner("Extracting audio sample..."):
                try:
                    seg = segments[segment_index]
                    # Create output directory for voice samples
                    rel_path = os.path.relpath(selected_video_path, project_path)
                    samples_dir = os.path.join("redubber_tmp", rel_path, "voice_samples")
                    os.makedirs(samples_dir, exist_ok=True)

                    original_sample_path = os.path.join(samples_dir, "original_sample.mp3")

                    # Extract audio sample
                    extract_audio_sample(selected_video_path, seg.start, seg.end, original_sample_path)
                    st.session_state.vr_original_sample = original_sample_path
                    st.session_state.vr_segment_text = seg.text

                    # Analyze with GPT-4o
                    with st.spinner("Analyzing voice with AI..."):
                        analysis = analyze_voice_with_gpt4o(original_sample_path, openai_token)
                        st.session_state.vr_voice = analysis['recommended_voice']
                        st.session_state.vr_instructions = analysis['voice_instructions']

                    st.success("Analysis complete!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error during analysis: {e}")

    with col2:
        can_preview = st.session_state.vr_voice and st.session_state.vr_original_sample
        if st.button("🔊 Preview", use_container_width=True, disabled=not can_preview,
                     help="Generate TTS sample for comparison" if can_preview else "Run Analyze first"):
            with st.spinner("Generating TTS sample..."):
                try:
                    seg = segments[segment_index]
                    rel_path = os.path.relpath(selected_video_path, project_path)
                    samples_dir = os.path.join("redubber_tmp", rel_path, "voice_samples")

                    generated_sample_path = os.path.join(samples_dir, "generated_sample.mp3")

                    # Generate TTS sample
                    text_to_speak = st.session_state.vr_segment_text or seg.text
                    generate_tts_sample(
                        text_to_speak,
                        st.session_state.vr_voice,
                        st.session_state.vr_instructions,
                        generated_sample_path,
                        openai_token
                    )
                    st.session_state.vr_generated_sample = generated_sample_path

                    st.success("TTS sample generated!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error generating TTS: {e}")

    # Audio players
    st.divider()
    st.subheader("🎧 Compare Samples")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Audio**")
        if st.session_state.vr_original_sample and os.path.exists(st.session_state.vr_original_sample):
            st.audio(st.session_state.vr_original_sample)
        else:
            st.info("Click 'Analyze' to extract original sample")

    with col2:
        st.write("**Generated Audio**")
        if st.session_state.vr_generated_sample and os.path.exists(st.session_state.vr_generated_sample):
            st.audio(st.session_state.vr_generated_sample)
        else:
            st.info("Click 'Preview' to generate TTS sample")

    st.divider()

    # Cancel / Accept buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Cancel", use_container_width=True):
            # Clear modal state
            for key in ['vr_voice', 'vr_instructions', 'vr_original_sample', 'vr_generated_sample', 'vr_segment_text']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.show_voice_refinement_modal = False
            st.rerun()

    with col2:
        can_accept = st.session_state.vr_voice
        if st.button("✅ Accept", use_container_width=True, type="primary", disabled=not can_accept):
            if project_id:
                # Save to project database
                db_manager.set_voice_settings(
                    project_id,
                    st.session_state.vr_voice,
                    st.session_state.vr_instructions
                )

                st.success("Voice settings saved to project!")

                # Clear modal state
                for key in ['vr_voice', 'vr_instructions', 'vr_original_sample', 'vr_generated_sample', 'vr_segment_text']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.show_voice_refinement_modal = False
                st.rerun()
            else:
                st.error("Could not save settings - project not found")


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
        saved_token, saved_model, saved_default_voice = load_openai_config()
        saved_target_language = load_target_language()
        st.session_state.openai_token = saved_token
        st.session_state.openai_model = saved_model
        st.session_state.default_voice = saved_default_voice
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
                    st.session_state.get('default_voice', 'nova')
                )

                # Clear model selection when token changes
                if 'openai_model' in st.session_state:
                    del st.session_state.openai_model
                    save_openai_config(openai_token, '', '', st.session_state.get('default_voice', 'nova'))
                if 'openai_models' in st.session_state:
                    del st.session_state.openai_models

            # Validate token and get models
            if openai_token:
                validate_and_get_models(openai_token)

            # Default Voice setting
            st.divider()
            st.subheader("🎤 Default Voice")

            default_voice_options = {
                'alloy': 'Alloy - Neutral, balanced',
                'echo': 'Echo - Warm, conversational male',
                'fable': 'Fable - British, expressive',
                'onyx': 'Onyx - Deep, authoritative male',
                'nova': 'Nova - Friendly, upbeat female',
                'shimmer': 'Shimmer - Warm, gentle female'
            }

            current_default_voice = st.session_state.get('default_voice', 'nova')
            selected_default_voice = st.selectbox(
                "Default TTS Voice",
                options=list(default_voice_options.keys()),
                format_func=lambda x: default_voice_options[x],
                index=list(default_voice_options.keys()).index(current_default_voice) if current_default_voice in default_voice_options else 4,  # nova is index 4
                help="Default voice for new projects"
            )

            # Save default voice when changed
            if selected_default_voice != current_default_voice:
                st.session_state.default_voice = selected_default_voice
                save_openai_config(
                    st.session_state.get('openai_token', ''),
                    st.session_state.get('openai_model', ''),
                    st.session_state.get('target_language', ''),
                    selected_default_voice
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

    # Check if voice refinement modal should be displayed
    if st.session_state.get('show_voice_refinement_modal', False):
        display_voice_refinement_modal()


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