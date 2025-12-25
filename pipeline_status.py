"""
Pipeline status detection for redubber videos.
Checks the redubber_tmp directory to determine which pipeline stages are complete.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineStatus:
    """Status of the redubbing pipeline for a video file."""
    video_path: str
    video_filename: str

    # Stage completion counts
    audio_chunks: int = 0
    transcripts: int = 0
    tts_segments: int = 0
    target_audio_chunks: int = 0
    subtitles_generated: bool = False
    final_file_exists: bool = False
    final_file_path: Optional[str] = None

    # External subtitle support - allows skipping early stages
    has_external_subs: bool = False

    @property
    def has_audio_chunks(self) -> bool:
        return self.audio_chunks > 0

    @property
    def has_transcripts(self) -> bool:
        return self.transcripts > 0

    @property
    def has_tts(self) -> bool:
        return self.tts_segments > 0

    @property
    def has_target_audio(self) -> bool:
        return self.target_audio_chunks > 0

    @property
    def is_complete(self) -> bool:
        return self.final_file_exists

    @property
    def has_subtitles_ready(self) -> bool:
        """Subtitles are ready if generated OR external subs exist."""
        return self.subtitles_generated or self.has_external_subs

    @property
    def progress_percent(self) -> int:
        """Calculate rough progress percentage based on completed stages."""
        stages_complete = 0
        total_stages = 5

        # If external subs exist, first 3 stages are effectively complete
        if self.has_external_subs:
            stages_complete = 3  # Audio, STT, Subtitles all skipped
        else:
            if self.has_audio_chunks:
                stages_complete += 1
            if self.has_transcripts:
                stages_complete += 1
            if self.subtitles_generated:
                stages_complete += 1

        # Remaining stages
        if self.has_tts:
            stages_complete += 1
        if self.final_file_exists:
            stages_complete += 1

        return int((stages_complete / total_stages) * 100)

    @property
    def current_stage(self) -> str:
        """Return the current/next stage to process."""
        if self.final_file_exists:
            return "Complete"
        # If external subs exist, skip to TTS
        if self.has_external_subs:
            if not self.has_tts:
                return "Generate TTS"
            if not self.has_target_audio:
                return "Assemble Audio"
            return "Mix Final"
        # Normal flow without external subs
        if not self.has_audio_chunks:
            return "Extract Audio"
        if not self.has_transcripts:
            return "Transcribe"
        if not self.subtitles_generated:
            return "Gen Subtitles"
        if not self.has_tts:
            return "Generate TTS"
        if not self.has_target_audio:
            return "Assemble Audio"
        return "Mix Final"

    def get_stage_status(self, stage: str) -> str:
        """
        Get status for a specific stage: 'skipped', 'done', or 'pending'.

        Stages: 'audio', 'stt', 'subtitles', 'tts', 'assemble', 'final'
        """
        if stage == 'audio':
            if self.has_external_subs:
                return 'skipped'
            return 'done' if self.has_audio_chunks else 'pending'
        elif stage == 'stt':
            if self.has_external_subs:
                return 'skipped'
            return 'done' if self.has_transcripts else 'pending'
        elif stage == 'subtitles':
            if self.has_external_subs:
                return 'skipped'
            return 'done' if self.subtitles_generated else 'pending'
        elif stage == 'tts':
            return 'done' if self.has_tts else 'pending'
        elif stage == 'assemble':
            return 'done' if self.has_target_audio else 'pending'
        elif stage == 'final':
            return 'done' if self.final_file_exists else 'pending'
        return 'pending'

    def can_run_stage(self, stage: str) -> bool:
        """Check if a stage can be run (dependencies are met)."""
        if stage == 'audio':
            return not self.has_external_subs
        elif stage == 'stt':
            return not self.has_external_subs and self.has_audio_chunks
        elif stage == 'subtitles':
            return not self.has_external_subs and self.has_transcripts
        elif stage == 'tts':
            # Can run if we have external subs OR generated subtitles
            return self.has_external_subs or self.subtitles_generated
        elif stage == 'assemble':
            return self.has_tts
        elif stage == 'final':
            return self.has_target_audio
        return False


def get_pipeline_status(video_path: str, project_path: str, tmp_root: str = "redubber_tmp") -> PipelineStatus:
    """
    Check the pipeline status for a video file.

    Args:
        video_path: Full path to the video file
        project_path: Root path of the project
        tmp_root: Root directory for temporary files (default: redubber_tmp)

    Returns:
        PipelineStatus object with completion information
    """
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)

    # Calculate relative path from project to video
    rel_path = os.path.relpath(video_path, project_path)
    working_dir = os.path.join(tmp_root, rel_path)

    status = PipelineStatus(
        video_path=video_path,
        video_filename=video_filename
    )

    # Check for external subtitle files (e.g., video.srt, video.en.srt)
    subtitle_extensions = ['.srt', '.vtt', '.ass', '.ssa']
    for ext in subtitle_extensions:
        # Check for exact match (video.srt)
        if os.path.exists(os.path.join(video_dir, video_filename + ext)):
            status.has_external_subs = True
            break
        # Check for language-suffixed (video.en.srt, video.eng.srt)
        for f in os.listdir(video_dir):
            if f.startswith(video_filename + '.') and f.endswith(ext):
                status.has_external_subs = True
                break
        if status.has_external_subs:
            break

    # Check source audio chunks (01_source_audio_chunks)
    audio_chunks_dir = os.path.join(working_dir, "01_source_audio_chunks")
    if os.path.exists(audio_chunks_dir):
        mp3_files = [f for f in os.listdir(audio_chunks_dir) if f.endswith('.mp3')]
        status.audio_chunks = len(mp3_files)

    # Check STT/transcripts (02_stt)
    stt_dir = os.path.join(working_dir, "02_stt")
    if os.path.exists(stt_dir):
        # Count .seg files as they represent completed transcriptions
        seg_files = [f for f in os.listdir(stt_dir) if f.endswith('.seg')]
        status.transcripts = len(seg_files)

    # Check subtitles (03_subtitles)
    subtitles_dir = os.path.join(working_dir, "03_subtitles")
    if os.path.exists(subtitles_dir):
        srt_files = [f for f in os.listdir(subtitles_dir) if f.endswith('.srt')]
        status.subtitles_generated = len(srt_files) > 0

    # Check TTS segments (04_tts)
    tts_dir = os.path.join(working_dir, "04_tts")
    if os.path.exists(tts_dir):
        tts_files = [f for f in os.listdir(tts_dir) if f.endswith('.mp3')]
        status.tts_segments = len(tts_files)

    # Check target audio chunks (05_target_audio_chunks)
    target_audio_dir = os.path.join(working_dir, "05_target_audio_chunks")
    if os.path.exists(target_audio_dir):
        target_files = [f for f in os.listdir(target_audio_dir) if f.endswith('.mp3')]
        status.target_audio_chunks = len(target_files)

    # Check for final output file (look for .en.mp4 or similar in the project directory)
    video_ext = os.path.splitext(video_path)[1]
    final_filename = f"{video_filename}.en{video_ext}"
    final_path = os.path.join(video_dir, final_filename)

    if os.path.exists(final_path):
        status.final_file_exists = True
        status.final_file_path = final_path

    return status


def clear_downstream_stages(video_path: str, project_path: str, from_stage: str, tmp_root: str = "redubber_tmp") -> list[str]:
    """
    Clear all pipeline stages downstream from (and including) the specified stage.

    Args:
        video_path: Full path to the video file
        project_path: Root path of the project
        from_stage: Stage to clear from ('audio', 'stt', 'subtitles', 'tts', 'mix')
        tmp_root: Root directory for temporary files

    Returns:
        List of directories that were cleared
    """
    import shutil
    import logging
    log = logging.getLogger(__name__)
    log.info(f"clear_downstream_stages: from_stage={from_stage}, video_path={video_path}, project_path={project_path}")

    # Define stage order and their directories
    stage_dirs = {
        'audio': '01_source_audio_chunks',
        'stt': '02_stt',
        'subtitles': '03_subtitles',
        'tts': '04_tts',
        'assemble': '05_target_audio_chunks',
        'final': None,  # Final stage creates output file, no tmp directory
    }

    stage_order = ['audio', 'stt', 'subtitles', 'tts', 'assemble', 'final']

    # Calculate working directory
    rel_path = os.path.relpath(video_path, project_path)
    working_dir = os.path.join(tmp_root, rel_path)

    # Find starting index
    try:
        start_idx = stage_order.index(from_stage)
    except ValueError:
        return []

    cleared = []
    log.info(f"clear_downstream_stages: working_dir={working_dir}")

    # Clear all stages from start_idx onwards
    for stage in stage_order[start_idx:]:
        dir_name = stage_dirs[stage]
        if dir_name:  # Skip if None (final stage has no tmp directory)
            stage_dir = os.path.join(working_dir, dir_name)
            if os.path.exists(stage_dir):
                log.info(f"clear_downstream_stages: removing {stage_dir}")
                shutil.rmtree(stage_dir)
                cleared.append(stage_dir)
            else:
                log.info(f"clear_downstream_stages: {stage_dir} does not exist")

    # Also delete the final output file if clearing from any stage
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_ext = os.path.splitext(video_path)[1]
    video_dir = os.path.dirname(video_path)
    final_path = os.path.join(video_dir, f"{video_filename}.en{video_ext}")

    if os.path.exists(final_path):
        log.info(f"clear_downstream_stages: removing final file {final_path}")
        os.remove(final_path)
        cleared.append(final_path)

    log.info(f"clear_downstream_stages: cleared {len(cleared)} items: {cleared}")
    return cleared


def get_all_pipeline_statuses(project_path: str, video_paths: list[str], tmp_root: str = "redubber_tmp") -> dict[str, PipelineStatus]:
    """
    Get pipeline status for all videos in a project.

    Args:
        project_path: Root path of the project
        video_paths: List of video file paths
        tmp_root: Root directory for temporary files

    Returns:
        Dictionary mapping video paths to their PipelineStatus
    """
    statuses = {}
    for video_path in video_paths:
        statuses[video_path] = get_pipeline_status(video_path, project_path, tmp_root)
    return statuses
