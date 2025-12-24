"""
Project page for displaying current project files and managing project.
"""

import streamlit as st
import os
import time
from video_analyzer import analyze_project_files
from utils import get_language_flag
from pipeline_status import get_pipeline_status, PipelineStatus


def display_pipeline_status(status: PipelineStatus, key_prefix: str):
    """Display a compact pipeline status indicator."""
    if status.is_complete:
        st.markdown("✅ **Done**")
    elif status.progress_percent == 0:
        st.markdown("⬜ Not started")
    else:
        # Show progress with stage indicators
        stages = []
        if status.has_audio_chunks:
            stages.append(f"🔊{status.audio_chunks}")
        if status.has_transcripts:
            stages.append(f"📝{status.transcripts}")
        if status.has_tts:
            stages.append(f"🗣️{status.tts_segments}")
        if status.has_target_audio:
            stages.append(f"🎵{status.target_audio_chunks}")

        progress_text = " ".join(stages) if stages else "..."
        st.markdown(f"{progress_text}")


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

    # Initialize redubber
    redubber = Redubber(openai_token=openai_token, interactive=False)

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


def check_video_readiness(video_data, target_language):
    """
    Check if a video meets the criteria for being ready for processing.

    Criteria:
    - 2+ audio languages available
    - Target language present in audio streams
    - Subtitle available in target language

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

    # Extract subtitle languages
    subtitle_languages = set()
    for subtitle in subtitles:
        if isinstance(subtitle, dict):
            lang = subtitle.get('language', 'unknown')
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
    has_target_subtitle = target_language in subtitle_languages

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

    # Project Settings - Source Language Override
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
        analysis_results = analyze_project_files(project_path, update_progress)

        # Save to database
        scan_data_json = json.dumps(analysis_results)
        db_manager.save_project_scan(project_id, scan_data_json)

        # Also save individual video analysis
        for video_data in analysis_results['videos']:
            db_manager.save_video_analysis(project_id, video_data)

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
        with st.container():
            filename_filter = st.text_input(
                "Filter by filename",
                value="",
                placeholder="Enter partial filename to filter...",
                help="Show only videos whose filename contains this text"
            )
            show_only_non_ready = st.checkbox(
                "Show only non-ready videos",
                value=False,
                help="Show only videos that don't meet all readiness criteria"
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

                # Extract subtitle languages
                sub_langs = []
                subtitles = video.get('subtitles', [])
                if isinstance(subtitles, list):
                    for sub in subtitles:
                        if isinstance(sub, dict):
                            lang = sub.get('language')
                            if lang and lang != 'unknown':
                                sub_langs.append(f"{get_language_flag(lang)} {lang}")

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
        filtered_table_data = []
        filtered_video_indices = []  # Track original video indices for play button
        for i, row in enumerate(table_data):
            # Apply filename filter
            if filename_filter and filename_filter.lower() not in row['File'].lower():
                continue

            # Apply readiness filter
            if show_only_non_ready and row.get('IsReady', False):
                continue

            filtered_table_data.append(row)
            filtered_video_indices.append(i)  # Store original index

        # Display as simple table using native Streamlit components
        if filtered_table_data:
            # Create table headers
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2.5, 0.7, 0.7, 1.2, 1.2, 1.5, 0.6, 0.6])
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
                        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2.5, 0.7, 0.7, 1.2, 1.2, 1.5, 0.6, 0.6])
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
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # Normal row without background
                    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2.5, 0.7, 0.7, 1.2, 1.2, 1.5, 0.6, 0.6])
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
                col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2.5, 0.7, 0.7, 1.2, 1.2, 1.5, 0.6, 0.6])
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
                with col8:
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
    """Display redub execution modal with progress."""
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

    # Display current status
    st.write("**Current Pipeline Status:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if pipeline_status.has_audio_chunks:
            st.success(f"🔊 Audio\n{pipeline_status.audio_chunks} chunks")
        else:
            st.info("🔊 Audio\nPending")
    with col2:
        if pipeline_status.has_transcripts:
            st.success(f"📝 STT\n{pipeline_status.transcripts} files")
        else:
            st.info("📝 STT\nPending")
    with col3:
        if pipeline_status.has_tts:
            st.success(f"🗣️ TTS\n{pipeline_status.tts_segments} segs")
        else:
            st.info("🗣️ TTS\nPending")
    with col4:
        if pipeline_status.has_target_audio:
            st.success(f"🎵 Mix\n{pipeline_status.target_audio_chunks} files")
        else:
            st.info("🎵 Mix\nPending")
    with col5:
        if pipeline_status.is_complete:
            st.success("✅ Done\nComplete")
        else:
            st.info("✅ Final\nPending")

    st.divider()

    # Check if already running
    if st.session_state.get('redub_running', False):
        st.warning("Redub is currently running...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Show current progress
        current_stage = st.session_state.get('redub_stage', 'Starting...')
        status_text.text(f"🔄 {current_stage}")

        if st.button("Cancel", use_container_width=True, type="secondary"):
            st.session_state.redub_running = False
            st.session_state.show_redub_modal = False
            if 'video_to_redub' in st.session_state:
                del st.session_state.video_to_redub
            st.rerun()
        return

    if pipeline_status.is_complete:
        st.success("This video has already been redubbed!")
        st.write(f"Output file: `{pipeline_status.final_file_path}`")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Close", use_container_width=True):
                st.session_state.show_redub_modal = False
                if 'video_to_redub' in st.session_state:
                    del st.session_state.video_to_redub
                st.rerun()
        with col2:
            if st.button("Re-run Pipeline", use_container_width=True, type="secondary"):
                st.session_state.redub_force_rerun = True
                st.rerun()
        return

    # Show start button
    st.write("**Ready to start redubbing?**")
    st.write(f"Next stage: **{pipeline_status.current_stage}**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Redub", use_container_width=True, type="primary"):
            st.session_state.redub_running = True
            st.session_state.redub_stage = "Initializing..."
            st.rerun()

    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.show_redub_modal = False
            if 'video_to_redub' in st.session_state:
                del st.session_state.video_to_redub
            st.rerun()


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