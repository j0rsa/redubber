"""
Project page for displaying current project files and managing project.
"""

import streamlit as st
import streamlit.components.v1 as components
import os
import time
from video_analyzer import analyze_project_files
from utils import get_language_flag
from pipeline_status import get_pipeline_status, PipelineStatus, clear_downstream_stages


def display_pipeline_status(status: PipelineStatus, key_prefix: str):
    """Display a compact pipeline status indicator with circular progress."""
    percent = status.progress_percent
    size = 32
    stroke_width = 3
    radius = (size - stroke_width) / 2
    circumference = 2 * 3.14159 * radius
    offset = circumference - (percent / 100) * circumference

    # Color: green only at 100%, orange for in-progress, gray for not started
    if percent == 100:
        color = "#4CAF50"  # Green - complete
        label = "Done"
    elif percent > 0:
        color = "#FF9800"  # Orange - in progress
        label = status.current_stage
    else:
        color = "#9E9E9E"  # Gray - not started
        label = "—"

    # Render using st.components.v1.html for proper SVG support
    html_content = f'''
    <div style="display: flex; align-items: center; gap: 6px; font-family: sans-serif;">
        <div style="position: relative; width: {size}px; height: {size}px;">
            <svg width="{size}" height="{size}" style="transform: rotate(-90deg);">
                <circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none" stroke="#e0e0e0" stroke-width="{stroke_width}"/>
                <circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" stroke-linecap="round"/>
            </svg>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 9px; font-weight: bold; color: {color};">{percent}%</div>
        </div>
        <span style="font-size: 10px; color: {color};">{label}</span>
    </div>
    '''
    components.html(html_content, height=40)


def get_project_voice_settings():
    """Get voice settings for the current project."""
    import logging
    db_manager = st.session_state.get('db_manager')
    project_path = st.session_state.get('current_project_path')
    logging.info(f"get_project_voice_settings: db_manager={db_manager is not None}, project_path={project_path}")
    if db_manager and project_path:
        project_data = db_manager.get_project_by_path(project_path)
        if project_data:
            settings = db_manager.get_voice_settings(project_data['id'])
            logging.info(f"get_project_voice_settings: returning {settings}")
            return settings
    logging.info("get_project_voice_settings: returning defaults")
    return {'voice': 'nova', 'voice_instructions': ''}


def run_redub_pipeline(video_data: dict, progress_callback=None):
    """
    Run the redub pipeline for a single video.

    Args:
        video_data: Video data dictionary containing path and other info
        progress_callback: Optional callback function for progress updates
    """
    from redubber import Redubber
    from reproj import Reproj

    video_path = video_data.get('path', '')
    if not video_path:
        raise ValueError("Video path is required")

    project_path = st.session_state.current_project_path

    # Get OpenAI token from session state
    openai_token = st.session_state.get('openai_token', '')
    if not openai_token:
        raise ValueError("OpenAI token is required. Please configure it in the sidebar.")

    # Get voice settings from project
    voice_settings = get_project_voice_settings()

    # Initialize redubber with voice settings
    redubber = Redubber(
        openai_token=openai_token,
        interactive=False,
        voice=voice_settings.get('voice', 'nova'),
        voice_instructions=voice_settings.get('voice_instructions', '')
    )

    if not redubber.can_redub(video_path):
        raise ValueError(f"Cannot redub {video_path}: unsupported format")

    # Create reproj for working directory management
    reproj = Reproj(source=project_path, file_path=video_path)

    # Run pipeline stages
    if progress_callback:
        progress_callback("Extracting audio chunks...")

    # Stage 1: Get text and segments (includes audio extraction and transcription)
    segments = redubber.get_text_and_segments(reproj, compact=True)

    if progress_callback:
        progress_callback(f"Transcription complete. {len(segments)} segments.")

    # Stage 2: Generate subtitles
    if progress_callback:
        progress_callback("Generating subtitles...")
    redubber.generate_subtitles(reproj, segments)

    # Stage 3: Generate TTS
    if progress_callback:
        progress_callback(f"Generating TTS for {len(segments)} segments...")
    redubber.tts_segments(reproj, segments)

    # Stage 4: Assemble audio
    if progress_callback:
        progress_callback("Assembling audio...")
    duration = redubber.get_media_duration(video_path)
    audio_file = redubber.assemble_long_audio(segments, reproj, duration)

    # Stage 5: Mix audio with video
    if progress_callback:
        progress_callback("Mixing audio with video...")

    # Determine output file path
    video_dir = os.path.dirname(video_path)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_ext = os.path.splitext(video_path)[1]
    output_file = os.path.join(video_dir, f"{video_filename}.en{video_ext}")

    # Get existing audio streams
    audio_streams = redubber.get_media_audio_streams(video_path)

    # Check for source language override in project settings
    db_manager = st.session_state.db_manager
    project_data = db_manager.get_project_by_path(project_path)
    source_language_override = ''
    if project_data:
        source_language_override = db_manager.get_source_language_override(project_data['id'])

    # Apply override if set, otherwise use detected languages
    if source_language_override:
        # Replace all detected audio stream languages with the override
        languages = [source_language_override] * len(audio_streams) + ["eng"]
    else:
        languages = audio_streams + ["eng"]

    redubber.mix_audio_with_video(reproj, audio_file, output_file, languages)

    if progress_callback:
        progress_callback("Complete!")

    return output_file


def run_pipeline_stage(video_data: dict, stage: str, progress_callback=None):
    """
    Run a single pipeline stage for a video.

    Args:
        video_data: Video data dictionary containing path and other info
        stage: Stage to run ('audio', 'stt', 'subtitles', 'tts', 'mix')
        progress_callback: Optional callback function for progress updates

    Returns:
        True if successful
    """
    from redubber import Redubber
    from reproj import Reproj

    video_path = video_data.get('path', '')
    if not video_path:
        raise ValueError("Video path is required")

    project_path = st.session_state.current_project_path

    # Get OpenAI token from session state
    openai_token = st.session_state.get('openai_token', '')
    if not openai_token:
        raise ValueError("OpenAI token is required. Please configure it in the sidebar.")

    # Get voice settings from project
    voice_settings = get_project_voice_settings()

    # Initialize redubber with voice settings
    redubber = Redubber(
        openai_token=openai_token,
        interactive=False,
        voice=voice_settings.get('voice', 'nova'),
        voice_instructions=voice_settings.get('voice_instructions', '')
    )

    if not redubber.can_redub(video_path):
        raise ValueError(f"Cannot redub {video_path}: unsupported format")

    # Create reproj for working directory management
    reproj = Reproj(source=project_path, file_path=video_path)

    if stage == 'audio':
        if progress_callback:
            progress_callback("Extracting audio chunks...")
        redubber.extract_audio_chunks(video_path, reproj)
        return True

    elif stage == 'stt':
        if progress_callback:
            progress_callback("Transcribing audio...")
        # Need to get audio chunks first
        from reproj import Reproj as ReprojClass
        audio_dir = reproj.get_file_working_dir(ReprojClass.Section.SOURCE_AUDIO_CHUNKS)
        if not os.path.exists(audio_dir) or not os.listdir(audio_dir):
            raise ValueError("Audio chunks not found. Run audio extraction first.")
        # Run STT on the audio chunks
        redubber.get_text_and_segments(reproj, compact=True)
        return True

    elif stage == 'subtitles':
        if progress_callback:
            progress_callback("Generating subtitles...")
        # Get segments from STT
        segments = redubber.get_text_and_segments(reproj, compact=True)
        redubber.generate_subtitles(reproj, segments)
        return True

    elif stage == 'tts':
        if progress_callback:
            progress_callback("Generating TTS...")
        # Get segments - either from STT or from external subtitles
        segments = get_segments_for_video(video_path, project_path)
        if not segments:
            raise ValueError("No segments available. Need subtitles or STT results.")
        redubber.tts_segments(reproj, segments)
        return True

    elif stage == 'assemble':
        if progress_callback:
            progress_callback("Assembling audio track...")
        # Get segments for duration calculation
        segments = get_segments_for_video(video_path, project_path)
        if not segments:
            raise ValueError("No segments available.")

        # Assemble audio
        duration = redubber.get_media_duration(video_path)
        audio_file = redubber.assemble_long_audio(segments, reproj, duration)

        if progress_callback:
            progress_callback("Audio assembled!")
        return True

    elif stage == 'final':
        if progress_callback:
            progress_callback("Mixing audio with video...")

        # Get segments for audio assembly
        segments = get_segments_for_video(video_path, project_path)
        if not segments:
            raise ValueError("No segments available.")

        # Get or assemble audio file
        duration = redubber.get_media_duration(video_path)
        audio_file = redubber.assemble_long_audio(segments, reproj, duration)

        # Determine output file path
        video_dir = os.path.dirname(video_path)
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        video_ext = os.path.splitext(video_path)[1]
        output_file = os.path.join(video_dir, f"{video_filename}.en{video_ext}")

        # Get existing audio streams
        audio_streams = redubber.get_media_audio_streams(video_path)

        # Check for source language override in project settings
        db_manager = st.session_state.db_manager
        project_data = db_manager.get_project_by_path(project_path)
        source_language_override = ''
        if project_data:
            source_language_override = db_manager.get_source_language_override(project_data['id'])

        # Apply override if set, otherwise use detected languages
        if source_language_override:
            languages = [source_language_override] * len(audio_streams) + ["eng"]
        else:
            languages = audio_streams + ["eng"]

        redubber.mix_audio_with_video(reproj, audio_file, output_file, languages)

        if progress_callback:
            progress_callback("Complete!")
        return output_file

    return False


def get_segments_for_video(video_path: str, project_path: str):
    """
    Get segments for a video - either from STT results or from external subtitles.

    Returns list of segment objects or None if not available.
    """
    from reproj import Reproj
    from redubber import Redubber

    reproj = Reproj(source=project_path, file_path=video_path)
    pipeline_status = get_pipeline_status(video_path, project_path)

    # Try to get segments from STT first
    stt_dir = reproj.get_file_working_dir(Reproj.Section.STT)
    if os.path.exists(stt_dir):
        seg_files = [f for f in os.listdir(stt_dir) if f.endswith('.seg')]
        if seg_files:
            # Load segments from STT
            openai_token = st.session_state.get('openai_token', '')
            redubber = Redubber(openai_token=openai_token, interactive=False)
            return redubber.get_text_and_segments(reproj, compact=True)

    # If no STT segments, try external subtitles
    if pipeline_status.has_external_subs:
        return get_segments_from_external_subs(video_path)

    return None


def get_segments_from_external_subs(video_path: str):
    """
    Parse external subtitle file and convert to segment format.
    """
    from dataclasses import dataclass

    @dataclass
    class SubSegment:
        id: int
        start: float
        end: float
        text: str
        tts_file: str = None

    video_dir = os.path.dirname(video_path)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    # Find subtitle file
    subtitle_extensions = ['.srt', '.vtt', '.ass', '.ssa']
    subtitle_path = None

    for ext in subtitle_extensions:
        # Check exact match
        test_path = os.path.join(video_dir, video_filename + ext)
        if os.path.exists(test_path):
            subtitle_path = test_path
            break
        # Check language-suffixed
        for f in os.listdir(video_dir):
            if f.startswith(video_filename + '.') and f.endswith(ext):
                subtitle_path = os.path.join(video_dir, f)
                break
        if subtitle_path:
            break

    if not subtitle_path:
        return None

    # Parse SRT file
    segments = []
    with open(subtitle_path, 'r', encoding='utf-8') as f:
        content = f.read()

    import re
    # SRT format: index, timestamp, text, blank line
    pattern = r'(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n((?:(?!\n\d+\n).)+)'
    matches = re.findall(pattern, content, re.DOTALL)

    def parse_timestamp(ts):
        ts = ts.replace(',', '.')
        parts = ts.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    for match in matches:
        idx, start_ts, end_ts, text = match
        text = text.strip().replace('\n', ' ')
        if text:
            segments.append(SubSegment(
                id=int(idx),
                start=parse_timestamp(start_ts),
                end=parse_timestamp(end_ts),
                text=text
            ))

    return segments


def check_video_readiness(video_data, target_language):
    """
    Check if a video meets the criteria for being ready for processing.

    Criteria:
    - 2+ audio languages available
    - Target language present in audio streams
    - Subtitle available in target language (or matching external .srt file exists)

    Returns: bool
    """
    if not target_language:
        return False

    audio_streams = video_data.get('audio_streams', [])
    subtitles = video_data.get('subtitles', [])

    # Extract audio languages
    audio_languages = set()
    for stream in audio_streams:
        if isinstance(stream, dict):
            lang = stream.get('language', 'unknown')
            if lang and lang != 'unknown':
                # Convert 2-letter codes to 3-letter codes for comparison
                if len(lang) == 2:
                    lang_mapping = {
                        'en': 'eng', 'es': 'spa', 'fr': 'fra', 'de': 'deu',
                        'it': 'ita', 'pt': 'por', 'ru': 'rus', 'ja': 'jpn',
                        'ko': 'kor', 'zh': 'zho', 'ar': 'ara', 'hi': 'hin',
                        'nl': 'nld', 'sv': 'swe', 'no': 'nor', 'da': 'dan',
                        'fi': 'fin', 'pl': 'pol', 'tr': 'tur'
                    }
                    lang = lang_mapping.get(lang, lang)
                audio_languages.add(lang)

    # Extract subtitle languages and check for matching external files
    subtitle_languages = set()
    has_matching_external_subtitle = False

    for subtitle in subtitles:
        if isinstance(subtitle, dict):
            is_embedded = subtitle.get('embedded', False)
            lang = subtitle.get('language', 'unknown')

            # External subtitle file that matches video name = assume target language
            if not is_embedded and subtitle.get('path'):
                has_matching_external_subtitle = True

            if lang and lang != 'unknown':
                # Convert 2-letter codes to 3-letter codes for comparison
                if len(lang) == 2:
                    lang_mapping = {
                        'en': 'eng', 'es': 'spa', 'fr': 'fra', 'de': 'deu',
                        'it': 'ita', 'pt': 'por', 'ru': 'rus', 'ja': 'jpn',
                        'ko': 'kor', 'zh': 'zho', 'ar': 'ara', 'hi': 'hin',
                        'nl': 'nld', 'sv': 'swe', 'no': 'nor', 'da': 'dan',
                        'fi': 'fin', 'pl': 'pol', 'tr': 'tur'
                    }
                    lang = lang_mapping.get(lang, lang)
                subtitle_languages.add(lang)

    # Check criteria
    has_multiple_audio_languages = len(audio_languages) >= 2
    has_target_audio = target_language in audio_languages
    # Target subtitle is satisfied if: language matches OR external subtitle file exists
    has_target_subtitle = target_language in subtitle_languages or has_matching_external_subtitle

    return has_multiple_audio_languages and has_target_audio and has_target_subtitle


def display_current_project_page():
    """Display the current project page with files and refresh option."""
    if not st.session_state.current_project_path:
        st.error("No project loaded")
        return

    project_name = os.path.basename(st.session_state.current_project_path) or "Root"
    st.header(f"📁 {project_name}")

    # Show full project path
    st.caption(f"Path: {st.session_state.current_project_path}")

    # Get or create project in database
    db_manager = st.session_state.db_manager
    project_data = db_manager.get_project_by_path(st.session_state.current_project_path)

    if not project_data:
        # Create project entry
        project_id = db_manager.add_project(st.session_state.current_project_path, project_name)
        project_data = {'id': project_id}
    else:
        project_id = project_data['id']

    # Check if project has scan results in database
    has_scan_data = db_manager.has_project_scan(project_id)

    if not has_scan_data:
        st.info("🔄 Project needs to be initialized. Click 'Initialize Project' to scan files and analyze content.")

    # Action buttons
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        if not has_scan_data:
            if st.button("🚀 Initialize Project", use_container_width=True, type="primary"):
                initialize_project(project_id)
        else:
            if st.button("🔄 Re-initialize Project", use_container_width=True):
                refresh_project(project_id)

    with col2:
        if st.button("🗑️ Delete Project", type="secondary", use_container_width=True):
            st.session_state.show_delete_confirmation = True
            st.rerun()

    # Delete confirmation dialog
    if st.session_state.get('show_delete_confirmation', False):
        display_delete_confirmation_dialog()

    # Project Settings - Source Language Override and Voice Settings
    with st.expander("Project Settings", expanded=False):
        # Source language override options
        source_language_options = {
            '': 'No Override (use detected)',
            'rus': 'Russian',
            'zho': 'Chinese',
            'kor': 'Korean',
            'jpn': 'Japanese',
            'ita': 'Italian',
        }

        # Load current setting from database
        current_override = db_manager.get_source_language_override(project_id)

        selected_override = st.selectbox(
            "Source Language Override",
            options=list(source_language_options.keys()),
            format_func=lambda x: source_language_options[x],
            index=list(source_language_options.keys()).index(current_override) if current_override in source_language_options else 0,
            help="Override the detected source language when setting audio track metadata in the output file"
        )

        # Save if changed
        if selected_override != current_override:
            db_manager.set_source_language_override(project_id, selected_override)
            st.rerun()

        st.divider()
        st.subheader("🎤 Voice Settings")

        # Load voice settings from project database
        voice_settings = db_manager.get_voice_settings(project_id)

        # Voice dropdown
        voice_options = {
            '': 'Select voice...',
            'alloy': 'Alloy - Neutral, balanced',
            'echo': 'Echo - Warm, conversational male',
            'fable': 'Fable - British, expressive',
            'onyx': 'Onyx - Deep, authoritative male',
            'nova': 'Nova - Friendly, upbeat female',
            'shimmer': 'Shimmer - Warm, gentle female'
        }

        current_voice = voice_settings.get('voice', '')
        selected_voice = st.selectbox(
            "TTS Voice",
            options=list(voice_options.keys()),
            format_func=lambda x: voice_options[x],
            index=list(voice_options.keys()).index(current_voice) if current_voice in voice_options else 0,
            help="Select the OpenAI Text-to-Speech voice for this project"
        )

        # Voice instructions
        current_instructions = voice_settings.get('voice_instructions', '')
        voice_instructions = st.text_area(
            "Voice Instructions",
            value=current_instructions,
            height=100,
            help="Custom instructions for voice generation (tone, style, emphasis, etc.)",
            placeholder="e.g., Speak in a calm, professional tone with clear enunciation..."
        )

        # Save voice settings if changed
        if selected_voice != current_voice or voice_instructions != current_instructions:
            db_manager.set_voice_settings(project_id, selected_voice, voice_instructions)

        # Refine button
        st.divider()
        openai_token = st.session_state.get('openai_token', '')
        if st.button(
            "🔧 Refine Voice",
            use_container_width=True,
            disabled=not openai_token,
            help="Analyze a video sample to find the best matching voice" if openai_token else "Configure OpenAI token first"
        ):
            st.session_state.show_voice_refinement_modal = True
            st.rerun()

    # Display project files if initialized
    if has_scan_data:
        display_project_analysis(project_id)

    # Check if video modal should be displayed
    if st.session_state.get('show_video_modal', False):
        display_video_modal()

    # Check if redub modal should be displayed
    if st.session_state.get('show_redub_modal', False):
        display_redub_modal()


def initialize_project(project_id: int):
    """Initialize project by scanning and analyzing all files."""
    import json

    project_path = st.session_state.current_project_path
    db_manager = st.session_state.db_manager

    # Create full-width container for progress and status
    with st.container():
        progress_bar = st.progress(0)
        status_text = st.empty()

    def update_progress(message):
        status_text.text(f"🔄 {message}")
        # Simulate progress updates
        if "Scanning" in message:
            progress_bar.progress(0.1)
        elif "Analyzing video" in message:
            # Extract current/total from message if possible
            progress_bar.progress(0.3)
        elif "complete" in message.lower():
            progress_bar.progress(1.0)

    # Analyze project files
    try:
        target_language = st.session_state.get('target_language', 'eng')
        analysis_results = analyze_project_files(project_path, update_progress, target_language)

        # Save to database
        scan_data_json = json.dumps(analysis_results)
        db_manager.save_project_scan(project_id, scan_data_json)

        # Also save individual video analysis
        for video_data in analysis_results['videos']:
            db_manager.save_video_analysis(project_id, video_data)

        # Set project voice to default voice
        default_voice = st.session_state.get('default_voice', 'nova')
        current_voice_settings = db_manager.get_voice_settings(project_id)
        if not current_voice_settings.get('voice'):
            db_manager.set_voice_settings(project_id, default_voice, '')

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Full-width success message
        with st.container():
            st.success("✅ Project initialized successfully!")
        time.sleep(1)
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        # Full-width error message
        with st.container():
            st.error(f"❌ Error initializing project: {str(e)}")


def refresh_project(project_id: int):
    """Refresh project by dropping database data and re-scanning."""
    db_manager = st.session_state.db_manager

    # Remove existing scan data from database
    import sqlite3
    with sqlite3.connect(db_manager.db_path) as conn:
        cursor = conn.cursor()
        # Clear scan data
        cursor.execute("DELETE FROM project_scans WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM video_analysis WHERE project_id = ?", (project_id,))
        conn.commit()

    # Re-initialize
    initialize_project(project_id)


def display_project_analysis(project_id: int):
    """Display the analyzed project data in a compact table format."""
    import json

    db_manager = st.session_state.db_manager

    # Load scan data from database
    scan_data_json = db_manager.get_project_scan(project_id)

    if not scan_data_json:
        st.error("Project not initialized")
        return

    analysis_data = json.loads(scan_data_json)

    # Handle autoredub execution
    if st.session_state.get('autoredub_video'):
        video_data = st.session_state.autoredub_video
        video_filename = video_data.get('filename', 'Unknown')
        video_path = video_data.get('path', '')
        project_path = st.session_state.current_project_path

        # Show progress UI
        st.info(f"⚡ Auto-redubbing: **{video_filename}**")
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        try:
            # Get pipeline status
            pipeline_status = get_pipeline_status(video_path, project_path)

            def update_progress(message, progress=None):
                if progress is not None:
                    progress_placeholder.progress(progress / 100, text=f"{progress}% - {message}")
                else:
                    status_placeholder.caption(f"📍 {message}")

            # Run the redub
            output_file = run_smart_redub(video_data, pipeline_status, update_progress)

            # Clear autoredub state
            del st.session_state.autoredub_video
            if 'autoredub_index' in st.session_state:
                del st.session_state.autoredub_index

            progress_placeholder.empty()
            status_placeholder.empty()
            st.success(f"✅ Redub complete: {os.path.basename(output_file)}")
            time.sleep(1)
            st.rerun()

        except Exception as e:
            # Clear autoredub state on error
            del st.session_state.autoredub_video
            if 'autoredub_index' in st.session_state:
                del st.session_state.autoredub_index

            progress_placeholder.empty()
            status_placeholder.empty()
            st.error(f"❌ Autoredub failed: {e}")

    # Project statistics
    stats = analysis_data['stats']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Video Files", stats['total_videos'])
    with col2:
        st.metric("Subtitle Files", stats['total_subtitles'])
    with col3:
        st.metric("Total Files", stats['total_videos'] + stats['total_subtitles'])

    st.divider()

    # Video files table
    if analysis_data.get('videos'):
        st.subheader("📹 Video Files")

        # Always visible filter section
        # Use query params to persist filter values across page refreshes
        query_params = st.query_params

        # Read initial values from query params
        initial_hide_completed = query_params.get('hide_completed', 'false') == 'true'
        initial_debug_mode = query_params.get('debug_mode', 'false') == 'true'
        initial_filename_filter = query_params.get('filename_filter', '')

        # Define callbacks to save filter values to query params
        def on_hide_completed_change():
            value = st.session_state.get('hide_completed_filter', False)
            st.session_state['_hide_completed_preserved'] = value
            if value:
                st.query_params['hide_completed'] = 'true'
            elif 'hide_completed' in st.query_params:
                del st.query_params['hide_completed']

        def on_debug_mode_change():
            value = st.session_state.get('debug_mode_filter', False)
            st.session_state['_debug_mode_preserved'] = value
            if value:
                st.query_params['debug_mode'] = 'true'
            elif 'debug_mode' in st.query_params:
                del st.query_params['debug_mode']

        def on_filename_filter_change():
            value = st.session_state.get('video_filename_filter', '')
            if value:
                st.query_params['filename_filter'] = value
            elif 'filename_filter' in st.query_params:
                del st.query_params['filename_filter']

        with st.container():
            filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
            with filter_col1:
                filename_filter = st.text_input(
                    "Filter by filename",
                    value=initial_filename_filter,
                    placeholder="Enter partial filename to filter...",
                    help="Show only videos whose filename contains this text",
                    key="video_filename_filter",
                    on_change=on_filename_filter_change
                )
            with filter_col2:
                show_only_incomplete = st.checkbox(
                    "Hide completed",
                    value=initial_hide_completed,
                    help="Hide videos that have already been redubbed (100% pipeline)",
                    key="hide_completed_filter",
                    on_change=on_hide_completed_change
                )
            with filter_col3:
                debug_mode = st.checkbox(
                    "Debug",
                    value=initial_debug_mode,
                    help="Show Play and Redub modal buttons",
                    key="debug_mode_filter",
                    on_change=on_debug_mode_change
                )
            st.divider()

        # Get target language from session state
        target_language = st.session_state.get('target_language', '')

        # Create table data
        table_data = []
        for video in analysis_data['videos']:
            try:
                # Check if video is ready for processing
                is_ready = check_video_readiness(video, target_language)

                # Extract audio languages
                audio_langs = []
                audio_streams = video.get('audio_streams', [])
                if isinstance(audio_streams, list):
                    for stream in audio_streams:
                        if isinstance(stream, dict):
                            lang = stream.get('language', 'unknown')
                            if lang and lang != 'unknown':
                                audio_langs.append(f"{get_language_flag(lang)} {lang}")

                # Extract subtitle languages (both embedded and external)
                sub_langs = []
                subtitles = video.get('subtitles', [])
                if isinstance(subtitles, list):
                    for sub in subtitles:
                        if isinstance(sub, dict):
                            lang = sub.get('language') or 'unknown'
                            is_embedded = sub.get('embedded', False)
                            # Mark embedded subtitles with 📦 icon, external with 📄
                            prefix = "📦" if is_embedded else "📄"
                            if lang and lang != 'unknown':
                                sub_langs.append(f"{prefix}{get_language_flag(lang)} {lang}")
                            else:
                                # Show subtitle even if language unknown
                                sub_langs.append(f"{prefix}?")

                # Ensure size_mb is a valid number and format it
                size_mb = video.get('size_mb', 0)
                if not isinstance(size_mb, (int, float)):
                    size_mb = 0.0

                # Format file size in human readable format
                def format_file_size(size_mb):
                    size_bytes = size_mb * 1024 * 1024
                    if size_bytes < 1024:
                        return f"{size_bytes:.0f} B"
                    elif size_bytes < 1024**2:
                        return f"{size_bytes/1024:.1f} KB"
                    elif size_bytes < 1024**3:
                        return f"{size_bytes/(1024**2):.1f} MB"
                    else:
                        return f"{size_bytes/(1024**3):.1f} GB"

                size_formatted = format_file_size(size_mb)

                # Ensure duration_seconds is a valid number and format it
                duration_seconds = video.get('duration_seconds', 0)
                if not isinstance(duration_seconds, (int, float)):
                    duration_seconds = 0.0

                # Format duration as MM:SS or HH:MM:SS
                if duration_seconds > 0:
                    hours = int(duration_seconds // 3600)
                    minutes = int((duration_seconds % 3600) // 60)
                    seconds = int(duration_seconds % 60)
                    if hours > 0:
                        duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
                    else:
                        duration_str = f"{minutes}:{seconds:02d}"
                else:
                    duration_str = "0:00"

                # Get pipeline status for this video
                video_path = video.get('path', '')
                project_path = st.session_state.current_project_path
                pipeline_status = None
                if video_path and project_path:
                    pipeline_status = get_pipeline_status(video_path, project_path)

                table_data.append({
                    'File': str(video.get('filename', 'Unknown')),
                    'Size': size_formatted,
                    'Duration': duration_str,
                    'Audio': str(', '.join(audio_langs)) if audio_langs else '❌ No audio',
                    'Subtitles': str(', '.join(sub_langs)) if sub_langs else '❌ None',
                    'Pipeline': pipeline_status,
                    'Status': 'Analyzed',
                    'IsReady': is_ready  # Add readiness flag
                })
            except Exception as e:
                st.error(f"Error processing video data: {e}")
                continue

        # Apply filters
        # Priority: widget value > preserved session state > query params
        effective_filename_filter = filename_filter or st.session_state.get('_filename_filter_preserved', '') or initial_filename_filter
        effective_hide_completed = show_only_incomplete or st.session_state.get('_hide_completed_preserved', False) or initial_hide_completed
        effective_debug_mode = debug_mode or st.session_state.get('_debug_mode_preserved', False) or initial_debug_mode

        filtered_table_data = []
        filtered_video_indices = []  # Track original video indices for play button
        for i, row in enumerate(table_data):
            # Apply filename filter
            if effective_filename_filter and effective_filename_filter.lower() not in row['File'].lower():
                continue

            # Apply completion filter - hide videos with 100% pipeline progress
            pipeline = row.get('Pipeline')
            if effective_hide_completed and pipeline and pipeline.is_complete:
                continue

            filtered_table_data.append(row)
            filtered_video_indices.append(i)  # Store original index

        # Display as simple table using native Streamlit components
        if filtered_table_data:
            # Create table headers - columns depend on debug mode
            if effective_debug_mode:
                col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([2.3, 0.6, 0.6, 1.1, 1.1, 1.4, 0.5, 0.5, 0.5])
                with col1:
                    st.write("**📁 File**")
                with col2:
                    st.write("**📊 Size**")
                with col3:
                    st.write("**⏱️ Dur**")
                with col4:
                    st.write("**🎵 Audio**")
                with col5:
                    st.write("**📝 Subs**")
                with col6:
                    st.write("**🔄 Pipeline**")
                with col7:
                    st.write("**▶️**")
                with col8:
                    st.write("**🎬**")
                with col9:
                    st.write("**⚡**")
            else:
                col1, col2, col3, col4, col5, col6, col7 = st.columns([2.5, 0.7, 0.7, 1.2, 1.2, 1.5, 0.6])
                with col1:
                    st.write("**📁 File**")
                with col2:
                    st.write("**📊 Size**")
                with col3:
                    st.write("**⏱️ Dur**")
                with col4:
                    st.write("**🎵 Audio**")
                with col5:
                    st.write("**📝 Subs**")
                with col6:
                    st.write("**🔄 Pipeline**")
                with col7:
                    st.write("**⚡**")

            st.divider()

            # Display each row
            for display_index, row in enumerate(filtered_table_data):
                original_video_index = filtered_video_indices[display_index]

                # Apply dim green background for ready videos
                if row.get('IsReady', False):
                    # Use container with custom CSS for green background
                    with st.container():
                        st.markdown("""
                        <style>
                        .ready-row {
                            background-color: rgba(144, 238, 144, 0.3) !important;
                            padding: 10px;
                            border-radius: 5px;
                            margin: 2px 0;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        st.markdown('<div class="ready-row">', unsafe_allow_html=True)
                        if effective_debug_mode:
                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([2.3, 0.6, 0.6, 1.1, 1.1, 1.4, 0.5, 0.5, 0.5])
                        else:
                            col1, col2, col3, col4, col5, col6, col7 = st.columns([2.5, 0.7, 0.7, 1.2, 1.2, 1.5, 0.6])
                        with col1:
                            st.write(row['File'])
                        with col2:
                            st.write(row['Size'])
                        with col3:
                            st.write(row['Duration'])
                        with col4:
                            st.write(row['Audio'])
                        with col5:
                            st.write(row['Subtitles'])
                        with col6:
                            # Display pipeline status
                            pipeline = row.get('Pipeline')
                            if pipeline:
                                display_pipeline_status(pipeline, f"ready_{display_index}")
                            else:
                                st.write("—")
                        if effective_debug_mode:
                            with col7:
                                if st.button("▶️", key=f"play_filtered_{display_index}", help="Play video"):
                                    st.session_state.video_to_play = analysis_data['videos'][original_video_index]
                                    st.session_state.show_video_modal = True
                                    st.rerun()
                            with col8:
                                if st.button("🎬", key=f"redub_{display_index}", help="Redub video"):
                                    st.session_state.video_to_redub = analysis_data['videos'][original_video_index]
                                    st.session_state.show_redub_modal = True
                                    st.rerun()
                            with col9:
                                video_path = analysis_data['videos'][original_video_index].get('path', '')
                                pipeline = row.get('Pipeline')
                                is_complete = pipeline and pipeline.is_complete
                                if st.button("⚡", key=f"autoredub_{display_index}", help="Auto-redub without modal", disabled=is_complete):
                                    st.session_state.autoredub_video = analysis_data['videos'][original_video_index]
                                    st.session_state.autoredub_index = display_index
                                    st.rerun()
                        else:
                            with col7:
                                video_path = analysis_data['videos'][original_video_index].get('path', '')
                                pipeline = row.get('Pipeline')
                                is_complete = pipeline and pipeline.is_complete
                                if st.button("⚡", key=f"autoredub_{display_index}", help="Auto-redub", disabled=is_complete):
                                    st.session_state.autoredub_video = analysis_data['videos'][original_video_index]
                                    st.session_state.autoredub_index = display_index
                                    st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # Normal row without background
                    if effective_debug_mode:
                        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([2.3, 0.6, 0.6, 1.1, 1.1, 1.4, 0.5, 0.5, 0.5])
                    else:
                        col1, col2, col3, col4, col5, col6, col7 = st.columns([2.5, 0.7, 0.7, 1.2, 1.2, 1.5, 0.6])
                    with col1:
                        st.write(row['File'])
                    with col2:
                        st.write(row['Size'])
                    with col3:
                        st.write(row['Duration'])
                    with col4:
                        st.write(row['Audio'])
                    with col5:
                        st.write(row['Subtitles'])
                    with col6:
                        # Display pipeline status
                        pipeline = row.get('Pipeline')
                        if pipeline:
                            display_pipeline_status(pipeline, f"normal_{display_index}")
                        else:
                            st.write("—")
                    if effective_debug_mode:
                        with col7:
                            if st.button("▶️", key=f"play_normal_{display_index}", help="Play video"):
                                st.session_state.video_to_play = analysis_data['videos'][original_video_index]
                                st.session_state.show_video_modal = True
                                st.rerun()
                        with col8:
                            if st.button("🎬", key=f"redub_normal_{display_index}", help="Redub video"):
                                st.session_state.video_to_redub = analysis_data['videos'][original_video_index]
                                st.session_state.show_redub_modal = True
                                st.rerun()
                        with col9:
                            video_path = analysis_data['videos'][original_video_index].get('path', '')
                            pipeline = row.get('Pipeline')
                            is_complete = pipeline and pipeline.is_complete
                            if st.button("⚡", key=f"autoredub_normal_{display_index}", help="Auto-redub without modal", disabled=is_complete):
                                st.session_state.autoredub_video = analysis_data['videos'][original_video_index]
                                st.session_state.autoredub_index = display_index
                                st.rerun()
                    else:
                        with col7:
                            video_path = analysis_data['videos'][original_video_index].get('path', '')
                            pipeline = row.get('Pipeline')
                            is_complete = pipeline and pipeline.is_complete
                            if st.button("⚡", key=f"autoredub_normal_{display_index}", help="Auto-redub", disabled=is_complete):
                                st.session_state.autoredub_video = analysis_data['videos'][original_video_index]
                                st.session_state.autoredub_index = display_index
                                st.rerun()

            # Calculate and display summary row
            if table_data:
                st.divider()

                # Calculate totals - need to get original MB values for summation
                total_size_mb = sum([
                    video.get('size_mb', 0)
                    for video in analysis_data['videos']
                    if isinstance(video.get('size_mb'), (int, float))
                ])

                # Format total size in human readable format
                def format_file_size(size_mb):
                    size_bytes = size_mb * 1024 * 1024
                    if size_bytes < 1024:
                        return f"{size_bytes:.0f} B"
                    elif size_bytes < 1024**2:
                        return f"{size_bytes/1024:.1f} KB"
                    elif size_bytes < 1024**3:
                        return f"{size_bytes/(1024**2):.1f} MB"
                    else:
                        return f"{size_bytes/(1024**3):.1f} GB"

                total_size_formatted = format_file_size(total_size_mb)

                total_duration_seconds = sum([
                    video.get('duration_seconds', 0)
                    for video in analysis_data['videos']
                    if isinstance(video.get('duration_seconds'), (int, float))
                ])

                # Format total duration
                if total_duration_seconds > 0:
                    hours = int(total_duration_seconds // 3600)
                    minutes = int((total_duration_seconds % 3600) // 60)
                    seconds = int(total_duration_seconds % 60)
                    if hours > 0:
                        total_duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
                    else:
                        total_duration_str = f"{minutes}:{seconds:02d}"
                else:
                    total_duration_str = "0:00"

                # Find common audio languages (present in ALL videos)
                common_audio_langs = set()
                if analysis_data['videos']:
                    # Start with languages from first video
                    first_video_audio_langs = set()
                    for stream in analysis_data['videos'][0].get('audio_streams', []):
                        lang = stream.get('language')
                        if lang and lang != 'unknown':
                            first_video_audio_langs.add(lang)

                    common_audio_langs = first_video_audio_langs.copy()

                    # Intersect with languages from other videos
                    for video in analysis_data['videos'][1:]:
                        video_audio_langs = set()
                        for stream in video.get('audio_streams', []):
                            lang = stream.get('language')
                            if lang and lang != 'unknown':
                                video_audio_langs.add(lang)
                        common_audio_langs = common_audio_langs.intersection(video_audio_langs)

                # Find common subtitle languages (present in ALL videos)
                common_sub_langs = set()
                if analysis_data['videos']:
                    # Start with languages from first video
                    first_video_sub_langs = set()
                    for sub in analysis_data['videos'][0].get('subtitles', []):
                        lang = sub.get('language')
                        if lang and lang != 'unknown':
                            first_video_sub_langs.add(lang)

                    common_sub_langs = first_video_sub_langs.copy()

                    # Intersect with languages from other videos
                    for video in analysis_data['videos'][1:]:
                        video_sub_langs = set()
                        for sub in video.get('subtitles', []):
                            lang = sub.get('language')
                            if lang and lang != 'unknown':
                                video_sub_langs.add(lang)
                        common_sub_langs = common_sub_langs.intersection(video_sub_langs)

                # Display summary row
                if effective_debug_mode:
                    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([2.3, 0.6, 0.6, 1.1, 1.1, 1.4, 0.5, 0.5, 0.5])
                else:
                    col1, col2, col3, col4, col5, col6, col7 = st.columns([2.5, 0.7, 0.7, 1.2, 1.2, 1.5, 0.6])
                with col1:
                    st.write("**📊 TOTAL**")
                with col2:
                    st.write(f"**{total_size_formatted}**")
                with col3:
                    st.write(f"**{total_duration_str}**")
                with col4:
                    common_audio_display = ', '.join([f"{get_language_flag(lang)}" for lang in sorted(common_audio_langs)]) if common_audio_langs else "—"
                    st.write(f"**{common_audio_display}**")
                with col5:
                    common_sub_display = ', '.join([f"{get_language_flag(lang)}" for lang in sorted(common_sub_langs)]) if common_sub_langs else "—"
                    st.write(f"**{common_sub_display}**")
                with col6:
                    st.write("")  # Empty for pipeline column
                with col7:
                    st.write("")  # Empty cell
                if effective_debug_mode:
                    with col8:
                        st.write("")  # Empty cell
                    with col9:
                        st.write("")  # Empty cell
        else:
            if table_data:
                st.info("No videos match the current filter criteria")
            else:
                st.info("No video files found")

    # Standalone subtitles (if any)
    if analysis_data['subtitles']:
        st.subheader("📝 Standalone Subtitles")
        standalone_data = []
        for sub in analysis_data['subtitles']:
            standalone_data.append({
                'File': sub['filename'],
                'Language': f"{get_language_flag(sub['language'])} {sub['language']}" if sub['language'] else '❓ Unknown'
            })

        if standalone_data:
            # Create simple table for standalone subtitles
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**📁 File**")
            with col2:
                st.write("**🌐 Language**")

            st.divider()

            for sub in standalone_data:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(sub['File'])
                with col2:
                    st.write(sub['Language'])
        else:
            st.info("No standalone subtitles found")


@st.dialog("Delete Project")
def display_delete_confirmation_dialog():
    """Display confirmation dialog for project deletion."""
    project_name = os.path.basename(st.session_state.current_project_path) or "Root"

    st.write("⚠️ **Are you sure you want to delete this project?**")
    st.write(f"**Project:** {project_name}")
    st.write(f"**Path:** {st.session_state.current_project_path}")

    st.warning("This will remove the project from your project list. The actual files on disk will not be deleted.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_delete_confirmation = False
            st.rerun()

    with col2:
        if st.button("🗑️ Delete Project", type="primary", use_container_width=True):
            # Delete the project
            db_manager = st.session_state.db_manager
            db_manager.remove_project_by_path(st.session_state.current_project_path)

            # Clear current project and switch to Open page
            st.session_state.current_project_path = None
            st.session_state.current_page = "Open"
            st.query_params["page"] = "Open"
            st.session_state.show_delete_confirmation = False

            st.success(f"Project '{project_name}' deleted successfully!")
            st.rerun()



@st.dialog("Video Player")
def display_video_modal():
    """Display video player modal with subtitle support."""
    video_data = st.session_state.get('video_to_play', {})

    if not video_data:
        st.error("No video data available")
        return

    video_path = video_data.get('path', '')
    video_filename = video_data.get('filename', 'Unknown')
    subtitles = video_data.get('subtitles', [])

    st.subheader(f"🎬 {video_filename}")

    # Display video player with subtitle support using Streamlit's native functionality
    try:
        # Prepare subtitle tracks for st.video
        subtitle_tracks = {}
        for subtitle in subtitles:
            sub_path = subtitle.get('path', '')
            sub_language = subtitle.get('language', 'unknown')
            sub_filename = subtitle.get('filename', '')

            if sub_path and os.path.exists(sub_path):
                # Use language code as key, file path as value
                lang_code = sub_language[:2] if sub_language != 'unknown' else 'en'
                subtitle_tracks[lang_code] = sub_path

        # Display video with native subtitle support
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()

        if subtitle_tracks:
            st.video(video_bytes, subtitles=subtitle_tracks)
        else:
            st.video(video_bytes)

    except Exception as e:
        st.error(f"Unable to play video: {str(e)}")
        st.info(f"Video path: {video_path}")

    # Display subtitle information and files
    if subtitles:
        st.subheader("📝 Available Subtitles")

        for i, subtitle in enumerate(subtitles):
            sub_filename = subtitle.get('filename', 'Unknown')
            sub_language = subtitle.get('language', 'unknown')
            sub_path = subtitle.get('path', '')

            with st.expander(f"📄 {sub_filename} ({get_language_flag(sub_language)} {sub_language})"):
                try:
                    # Try to read and display subtitle content
                    with open(sub_path, 'r', encoding='utf-8') as sub_file:
                        subtitle_content = sub_file.read()

                    # Display first 2000 characters with scrollable text area
                    if len(subtitle_content) > 2000:
                        st.text_area(
                            "Subtitle Content (first 2000 characters):",
                            subtitle_content[:2000] + "...",
                            height=300,
                            key=f"subtitle_content_{i}"
                        )
                        st.info(f"Full subtitle has {len(subtitle_content)} characters")
                    else:
                        st.text_area(
                            "Subtitle Content:",
                            subtitle_content,
                            height=300,
                            key=f"subtitle_content_{i}"
                        )

                except Exception as e:
                    st.error(f"Unable to read subtitle file: {str(e)}")
                    st.code(f"Path: {sub_path}")
    else:
        st.info("No subtitles available for this video")

    # Close button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Close", use_container_width=True, type="primary"):
            st.session_state.show_video_modal = False
            if 'video_to_play' in st.session_state:
                del st.session_state.video_to_play
            st.rerun()


@st.dialog("Redub Video", width="large")
def display_redub_modal():
    """Display redub execution modal with individual stage controls."""
    video_data = st.session_state.get('video_to_redub', {})

    if not video_data:
        st.error("No video data available")
        return

    video_path = video_data.get('path', '')
    video_filename = video_data.get('filename', 'Unknown')

    st.subheader(f"🎬 Redub: {video_filename}")

    # Check OpenAI configuration
    openai_token = st.session_state.get('openai_token', '')
    if not openai_token:
        st.error("OpenAI API token is not configured. Please set it in the sidebar.")
        if st.button("Close", use_container_width=True):
            st.session_state.show_redub_modal = False
            if 'video_to_redub' in st.session_state:
                del st.session_state.video_to_redub
            st.rerun()
        return

    # Get current pipeline status
    project_path = st.session_state.current_project_path
    pipeline_status = get_pipeline_status(video_path, project_path)

    # Check if a stage is currently running
    if st.session_state.get('stage_running', False):
        running_stage = st.session_state.get('running_stage_name', 'Processing')
        with st.spinner(f"Running: {running_stage}..."):
            # Execute the stage
            stage_to_run = st.session_state.get('stage_to_run')
            if stage_to_run:
                try:
                    result = run_pipeline_stage(video_data, stage_to_run)
                    st.session_state.stage_running = False
                    st.session_state.stage_to_run = None
                    st.session_state.running_stage_name = None
                    st.success(f"Stage completed!")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.session_state.stage_running = False
                    st.session_state.stage_to_run = None
                    st.session_state.running_stage_name = None
                    st.error(f"Error: {e}")
        return

    # Check if full pipeline is running
    if st.session_state.get('redub_running', False):
        # Use st.status for live updates
        with st.status("Running redub pipeline...", expanded=True) as status:
            # Create placeholders for progress elements (so they update in place)
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            try:
                # Determine starting stage based on current status
                pipeline_status = get_pipeline_status(video_path, project_path)

                def update_progress(message, progress=None):
                    status.update(label=f"🔄 {message}", state="running")
                    if progress is not None:
                        progress_placeholder.progress(progress / 100, text=message)
                    else:
                        progress_placeholder.empty()
                    status_placeholder.caption(f"📍 {message}")

                output_file = run_smart_redub(video_data, pipeline_status, update_progress)
                progress_placeholder.progress(1.0, text="Complete!")
                status.update(label="✅ Redub complete!", state="complete")
                st.session_state.redub_running = False
                st.session_state.redub_complete = True
                st.session_state.redub_output = output_file

            except Exception as e:
                status.update(label=f"❌ Error: {e}", state="error")
                st.session_state.redub_running = False
                st.error(f"Error: {e}")

        # Show result and close button
        if st.session_state.get('redub_complete'):
            st.success(f"Output: {st.session_state.get('redub_output', '')}")
            if st.button("Close", use_container_width=True):
                st.session_state.redub_complete = False
                st.session_state.show_redub_modal = False
                if 'video_to_redub' in st.session_state:
                    del st.session_state.video_to_redub
                st.rerun()
        return

    # Define stages with their info
    stages = [
        ('audio', '🔊 Audio', 'Extract audio chunks from video'),
        ('stt', '📝 STT', 'Transcribe audio to text'),
        ('subtitles', '📄 Subs', 'Generate subtitle files'),
        ('tts', '🗣️ TTS', 'Generate speech from text'),
        ('assemble', '🎵 Assemble', 'Join TTS segments into audio track'),
        ('final', '🎬 Final mix', 'Mix audio with video'),
    ]

    st.write("**Pipeline Stages:**")

    # Display each stage with status and action button
    for stage_id, stage_name, stage_desc in stages:
        status = pipeline_status.get_stage_status(stage_id)
        can_run = pipeline_status.can_run_stage(stage_id)

        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            if status == 'skipped':
                st.markdown(f"⏭️ **{stage_name}**")
            elif status == 'done':
                st.markdown(f"✅ **{stage_name}**")
            else:
                st.markdown(f"⏳ **{stage_name}**")

        with col2:
            if status == 'skipped':
                st.caption(f"Skipped (external subs available)")
            elif status == 'done':
                # Show details based on stage
                if stage_id == 'audio':
                    st.caption(f"Done - {pipeline_status.audio_chunks} chunks")
                elif stage_id == 'stt':
                    st.caption(f"Done - {pipeline_status.transcripts} transcripts")
                elif stage_id == 'subtitles':
                    st.caption(f"Done - subtitles generated")
                elif stage_id == 'tts':
                    st.caption(f"Done - {pipeline_status.tts_segments} segments")
                elif stage_id == 'assemble':
                    st.caption(f"Done - {pipeline_status.target_audio_chunks} audio files")
                elif stage_id == 'final':
                    st.caption(f"Done - final file ready")
            else:
                st.caption(stage_desc)

        with col3:
            if status == 'skipped':
                st.button("—", key=f"stage_{stage_id}", disabled=True, help="Skipped")
            elif status == 'done':
                if st.button("🔄 Rerun", key=f"stage_{stage_id}", help="Rerun this stage (clears downstream)"):
                    # Clear downstream stages
                    cleared = clear_downstream_stages(video_path, project_path, stage_id)
                    st.session_state.stage_to_run = stage_id
                    st.session_state.stage_running = True
                    st.session_state.running_stage_name = stage_name
                    st.rerun()
            elif can_run:
                if st.button("▶️ Run", key=f"stage_{stage_id}", help="Run this stage"):
                    st.session_state.stage_to_run = stage_id
                    st.session_state.stage_running = True
                    st.session_state.running_stage_name = stage_name
                    st.rerun()
            else:
                st.button("▶️ Run", key=f"stage_{stage_id}", disabled=True, help="Dependencies not met")

    st.divider()

    # Show completion status
    if pipeline_status.is_complete:
        st.success(f"✅ Video redubbed successfully!")
        st.write(f"Output: `{pipeline_status.final_file_path}`")

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if not pipeline_status.is_complete:
            if st.button("🚀 Start Redub", use_container_width=True, type="primary",
                        help="Run all remaining stages automatically"):
                st.session_state.redub_running = True
                st.session_state.redub_stage = "Initializing..."
                st.rerun()
        else:
            if st.button("🔄 Re-run All", use_container_width=True, type="secondary",
                        help="Clear all and re-run from beginning"):
                # Clear all stages
                clear_downstream_stages(video_path, project_path, 'audio')
                st.session_state.redub_running = True
                st.session_state.redub_stage = "Initializing..."
                st.rerun()

    with col2:
        if st.button("Close", use_container_width=True):
            st.session_state.show_redub_modal = False
            if 'video_to_redub' in st.session_state:
                del st.session_state.video_to_redub
            st.rerun()


def run_smart_redub(video_data: dict, pipeline_status: PipelineStatus, progress_callback=None):
    """
    Run redub pipeline smartly, starting from current stage.
    Handles both normal flow and external subtitle flow.

    Progress stages (external subs path):
    - TTS: 0-70%
    - Assemble: 70-90%
    - Mix: 90-100%

    Progress stages (normal path):
    - Audio+STT: 0-30%
    - Subtitles: 30-35%
    - TTS: 35-80%
    - Assemble: 80-92%
    - Mix: 92-100%
    """
    from redubber import Redubber
    from reproj import Reproj
    import logging
    log = logging.getLogger(__name__)

    video_path = video_data.get('path', '')
    project_path = st.session_state.current_project_path

    openai_token = st.session_state.get('openai_token', '')
    voice_settings = get_project_voice_settings()

    # Log voice settings for debugging
    log.info(f"run_smart_redub: voice_settings={voice_settings}")
    log.info(f"run_smart_redub: pipeline_status.has_tts={pipeline_status.has_tts}, has_external_subs={pipeline_status.has_external_subs}")

    redubber = Redubber(
        openai_token=openai_token,
        interactive=False,
        voice=voice_settings.get('voice') or 'nova',
        voice_instructions=voice_settings.get('voice_instructions') or ''
    )
    reproj = Reproj(source=project_path, file_path=video_path)

    def make_stage_callback(stage_start: float, stage_end: float, stage_name: str):
        """Create a callback that maps stage progress (0-1) to absolute progress."""
        def callback(stage_progress: float):
            absolute = stage_start + stage_progress * (stage_end - stage_start)
            if progress_callback:
                progress_callback(stage_name, int(absolute))
        return callback

    # If external subs exist, use the short path (TTS → Assemble → Mix)
    if pipeline_status.has_external_subs:
        # Get segments from external subs
        segments = get_segments_from_external_subs(video_path)
        if not segments:
            raise ValueError("Failed to parse external subtitles")

        # Run TTS if not done (0-70%)
        if not pipeline_status.has_tts:
            log.info(f"run_smart_redub: Starting TTS generation for {len(segments)} segments with voice={redubber.voice}")
            tts_callback = make_stage_callback(0, 70, f"TTS ({len(segments)} segments)")
            tts_callback(0)  # Initial progress
            redubber.tts_segments(reproj, segments, progress_callback=tts_callback)
            log.info(f"run_smart_redub: TTS generation complete")
        else:
            log.info(f"run_smart_redub: Skipping TTS (already done)")
            if progress_callback:
                progress_callback("TTS already done", 70)

        # Assemble (70-90%)
        assemble_callback = make_stage_callback(70, 90, "Assembling audio")
        assemble_callback(0)
        duration = redubber.get_media_duration(video_path)
        audio_file = redubber.assemble_long_audio(segments, reproj, duration, progress_callback=assemble_callback)

        # Mix (90-100%)
        if progress_callback:
            progress_callback("Mixing audio with video...", 90)

        # Output path
        video_dir = os.path.dirname(video_path)
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        video_ext = os.path.splitext(video_path)[1]
        output_file = os.path.join(video_dir, f"{video_filename}.en{video_ext}")

        # Get audio streams and language config
        audio_streams = redubber.get_media_audio_streams(video_path)
        db_manager = st.session_state.db_manager
        project_data = db_manager.get_project_by_path(project_path)
        source_language_override = ''
        if project_data:
            source_language_override = db_manager.get_source_language_override(project_data['id'])

        if source_language_override:
            languages = [source_language_override] * len(audio_streams) + ["eng"]
        else:
            languages = audio_streams + ["eng"]

        redubber.mix_audio_with_video(reproj, audio_file, output_file, languages)

        if progress_callback:
            progress_callback("Complete!", 100)
        return output_file

    # Normal flow - run from current stage
    # Stage 1: Audio + STT (0-30%)
    stt_callback = make_stage_callback(0, 30, "Audio extraction & transcription")
    stt_callback(0)
    segments = redubber.get_text_and_segments(reproj, compact=True, progress_callback=stt_callback)

    if progress_callback:
        progress_callback(f"Transcription complete ({len(segments)} segments)", 30)

    # Stage 2: Subtitles (30-35%)
    if not pipeline_status.subtitles_generated:
        if progress_callback:
            progress_callback("Generating subtitles...", 32)
        redubber.generate_subtitles(reproj, segments)
        if progress_callback:
            progress_callback("Subtitles generated", 35)

    # Stage 3: TTS (35-80%)
    if not pipeline_status.has_tts:
        log.info(f"run_smart_redub: Starting TTS generation for {len(segments)} segments with voice={redubber.voice}")
        tts_callback = make_stage_callback(35, 80, f"TTS ({len(segments)} segments)")
        tts_callback(0)
        redubber.tts_segments(reproj, segments, progress_callback=tts_callback)
        log.info(f"run_smart_redub: TTS generation complete")
    else:
        log.info(f"run_smart_redub: Skipping TTS (already done)")
        if progress_callback:
            progress_callback("TTS already done", 80)

    # Stage 4: Assemble (80-92%)
    assemble_callback = make_stage_callback(80, 92, "Assembling audio")
    assemble_callback(0)
    duration = redubber.get_media_duration(video_path)
    audio_file = redubber.assemble_long_audio(segments, reproj, duration, progress_callback=assemble_callback)

    # Stage 5: Mix (92-100%)
    if progress_callback:
        progress_callback("Mixing audio with video...", 92)

    video_dir = os.path.dirname(video_path)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_ext = os.path.splitext(video_path)[1]
    output_file = os.path.join(video_dir, f"{video_filename}.en{video_ext}")

    audio_streams = redubber.get_media_audio_streams(video_path)
    db_manager = st.session_state.db_manager
    project_data = db_manager.get_project_by_path(project_path)
    source_language_override = ''
    if project_data:
        source_language_override = db_manager.get_source_language_override(project_data['id'])

    if source_language_override:
        languages = [source_language_override] * len(audio_streams) + ["eng"]
    else:
        languages = audio_streams + ["eng"]

    redubber.mix_audio_with_video(reproj, audio_file, output_file, languages)

    if progress_callback:
        progress_callback("Complete!", 100)

    return output_file


def execute_redub_in_modal():
    """Execute the redub pipeline within the modal context."""
    video_data = st.session_state.get('video_to_redub', {})
    if not video_data:
        return

    progress_container = st.empty()
    status_container = st.empty()

    def update_progress(message):
        st.session_state.redub_stage = message
        status_container.text(f"🔄 {message}")

    try:
        output_file = run_redub_pipeline(video_data, update_progress)
        st.session_state.redub_running = False
        st.session_state.redub_complete = True
        st.session_state.redub_output = output_file
        st.success(f"Redub complete! Output: {output_file}")
    except Exception as e:
        st.session_state.redub_running = False
        st.session_state.redub_error = str(e)
        st.error(f"Error during redub: {e}")