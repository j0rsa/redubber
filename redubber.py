import os
import math
import subprocess
import json
import platform
from openai import OpenAI
from openai.types.audio.translation_verbose import TranslationVerbose
from openai.types.audio.transcription_segment import TranscriptionSegment
from typing import List
import logging
from pydantic import TypeAdapter
from concurrent.futures import ThreadPoolExecutor
import time
from pydantic import BaseModel
from reproj import Reproj
from tqdm import tqdm
from seg_postprocessor import postprocess_segments

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def get_aac_encoder() -> str:
    """Get the best available AAC encoder for current platform."""
    if platform.system() == "Darwin":
        return "aac_at"  # Apple AudioToolbox (hardware accelerated)
    else:
        return "aac"  # Native FFmpeg AAC encoder


class Redubber(BaseModel):
    supported_video_formats: List[str] = [
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".flv",
        ".wmv",
        ".webm",
        ".vob",
        ".m4v",
        ".3gp",
        ".3g2",
        ".m2ts",
        ".mts",
        ".ts",
        ".f4v",
        ".f4p",
        ".f4a",
        ".f4b",
        ".m2v",
        ".m4v",
        ".m1v",
        ".mpg",
        ".mpeg",
        ".mpv",
        ".mp2",
        ".mpe",
        ".m2p",
        ".m2t",
        ".mp2v",
        ".mpv2",
        ".m2ts",
        ".m2ts",
        ".mts",
        ".m2v",
    ]

    audio_ext: str = ".m4a"  # AAC in M4A container for hardware acceleration
    model: str = "gpt-4o"
    stt_model: str = "whisper-1"
    openai_token: str = ""
    openai_base_url: str = ""  # empty = use default OpenAI endpoint
    default_audio_chunk_duration: int = (
        15 * 60
    )  # 15 minutes (keeps under Whisper's 25MB limit)
    interactive: bool = False
    voice: str = "nova"
    voice_instructions: str = ""
    tts_speed: float = 1.25
    audio_chunk_duration: int = 15 * 60
    tts_concurrency: int = 20
    target_language: str = "eng"  # ISO 639-2/B code for dubbing output language

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        openai_token: str,
        interactive: bool = False,
        voice: str = "nova",
        voice_instructions: str = "",
        stt_model: str = "whisper-1",
        openai_base_url: str = "",
        tts_speed: float = 1.25,
        audio_chunk_duration: int = 15 * 60,
        tts_concurrency: int = 20,
        target_language: str = "eng",
    ):
        """Initialize the Redubber class.

        Args:
            openai_token: The OpenAI API token.
            interactive: Whether to show progress bars for long-running operations.
            voice: The TTS voice to use (alloy, echo, fable, onyx, nova, shimmer).
            voice_instructions: Instructions for voice style/tone.
            stt_model: OpenAI model for speech-to-text transcription.
            openai_base_url: Custom base URL for OpenAI-compatible APIs. Empty = default.
            tts_speed: TTS audio speed multiplier. 1.25 helps dubs fit original timing.
            audio_chunk_duration: Duration in seconds for audio chunks sent to Whisper.
            tts_concurrency: Number of parallel threads for TTS segment generation.
            target_language: ISO 639-2/B code for dubbing output language (default 'eng').
        """
        super().__init__(
            openai_token=openai_token,
            interactive=interactive,
            voice=voice,
            voice_instructions=voice_instructions,
            stt_model=stt_model,
            openai_base_url=openai_base_url,
            tts_speed=tts_speed,
            audio_chunk_duration=audio_chunk_duration,
            tts_concurrency=tts_concurrency,
            target_language=target_language,
        )

    def can_redub(self, source):
        result = os.path.splitext(source)[1] in self.supported_video_formats
        log.debug(f"Can redub {source}: {result}")
        return result

    def get_media_duration(self, file_path) -> float:
        log.debug(f"Getting media duration for {file_path}")
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        return float(probe_data["format"]["duration"])

    def get_media_audio_streams(self, file_path) -> List[str]:
        log.debug(f"Getting media audio streams for {file_path}")
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log.debug(f"FFprobe result: {result.stdout}")
        probe_data = json.loads(result.stdout)
        return [
            stream["tags"]["language"]
            for stream in probe_data["streams"]
            if stream["codec_type"] == "audio"
        ]

    def seconds_to_hms(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def extract_audio_chunks(
        self,
        reproj: Reproj,
        chunk_duration: int | None = None,
        replace: bool = False,
        progress_callback=None,
    ) -> list[str]:
        """Extract audio chunks from a video file using single-pass segmentation.

        Uses ffmpeg's segment muxer to extract all chunks in one pass, which is
        much faster than seeking and extracting each chunk separately.

        Args:
            reproj: The reproj object.
            chunk_duration: The duration of each audio chunk in seconds. Defaults to
                ``self.audio_chunk_duration`` (set at construction time).
            replace: Whether to replace the existing audio chunks.
            progress_callback: Optional callback(progress: float) where progress is 0.0-1.0.

        Returns:
            A list of paths to the audio chunks that are extracted and saved in SOURCE_AUDIO_CHUNKS section.
        """
        import re

        if chunk_duration is None:
            chunk_duration = self.audio_chunk_duration

        log.info(f"Extracting audio from {reproj.file_path}")
        target_rel_dir = reproj.get_file_working_dir(Reproj.Section.SOURCE_AUDIO_CHUNKS)
        total_duration = self.get_media_duration(reproj.file_path)
        log.info(f"Video duration {self.seconds_to_hms(total_duration)}")

        # Calculate number of chunks, but ensure last chunk is at least 2 seconds
        # If the last chunk would be < 2 seconds, reduce num_chunks by 1
        MIN_CHUNK_DURATION = 2.0
        num_chunks = math.ceil(total_duration / chunk_duration)
        last_chunk_duration = total_duration - (num_chunks - 1) * chunk_duration

        if last_chunk_duration < MIN_CHUNK_DURATION and num_chunks > 1:
            # Last chunk is too short, merge it with the previous chunk
            num_chunks -= 1
            last_chunk_duration = total_duration - (num_chunks - 1) * chunk_duration
            log.info(
                f"Last chunk would be {last_chunk_duration:.2f}s, merging with previous chunk"
            )

        log.info(
            f"Extracting {num_chunks} chunks of {self.seconds_to_hms(chunk_duration)} each (last: {self.seconds_to_hms(last_chunk_duration)})"
        )

        base_name = os.path.splitext(os.path.basename(reproj.file_path))[0]

        # Delete existing chunks if replace is set
        if replace and os.path.exists(target_rel_dir):
            for file in os.listdir(target_rel_dir):
                if file.endswith(self.audio_ext):
                    os.remove(os.path.join(target_rel_dir, file))

        # Build list of expected output files (1-indexed)
        result = []
        for i in range(num_chunks):
            chunk_filename = f"{base_name}_{i + 1:03d}{self.audio_ext}"
            result.append(os.path.join(target_rel_dir, chunk_filename))

        # Check if all chunks already exist
        all_exist = all(os.path.exists(p) for p in result)
        if all_exist and not replace:
            log.info("All audio chunks already exist, skipping extraction")
            if progress_callback:
                progress_callback(1.0)
            return result

        # Create output directory
        os.makedirs(target_rel_dir, exist_ok=True)

        # Use ffmpeg segment muxer for single-pass extraction
        # segment_start_number=1 makes it 1-indexed to match our naming convention
        # Calculate the exact duration to extract (to avoid creating a tiny last chunk)
        extract_duration = (num_chunks - 1) * chunk_duration + last_chunk_duration
        segment_template = os.path.join(
            target_rel_dir, f"{base_name}_%03d{self.audio_ext}"
        )

        cmd = [
            "ffmpeg",
            "-i",
            reproj.file_path,
            "-t",
            str(extract_duration),  # Limit total extraction duration
            "-vn",  # No video
            "-acodec",
            get_aac_encoder(),
            "-b:a",
            "128k",  # 128kbps is plenty for speech, keeps 15min chunks ~14MB (under 25MB Whisper limit)
            "-f",
            "segment",
            "-segment_time",
            str(chunk_duration),
            "-segment_start_number",
            "1",  # 1-indexed to match existing naming
            "-reset_timestamps",
            "1",
            "-y",
            segment_template,
        ]

        log.info("Running single-pass audio extraction")

        # Run ffmpeg with progress monitoring
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)

        # Parse ffmpeg progress from stderr (time=HH:MM:SS.xx)
        time_pattern = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

        if self.interactive:
            pbar = tqdm(total=100, desc="Extracting audio", unit="%")
            last_progress = 0

        for line in process.stderr:
            match = time_pattern.search(line)
            if match:
                hours, minutes, seconds = match.groups()
                current_time = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                progress = min(current_time / total_duration, 1.0)

                if progress_callback:
                    progress_callback(progress)

                if self.interactive:
                    new_progress = int(progress * 100)
                    if new_progress > last_progress:
                        pbar.update(new_progress - last_progress)
                        last_progress = new_progress

        if self.interactive:
            pbar.close()

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio extraction failed with return code {process.returncode}"
            )

        # Get actual files created
        actual_files = sorted(
            [
                os.path.join(target_rel_dir, f)
                for f in os.listdir(target_rel_dir)
                if f.endswith(self.audio_ext) and f.startswith(base_name)
            ]
        )

        # Filter out chunks that are too short (< MIN_CHUNK_DURATION seconds)
        # These cause Whisper API errors and typically contain no useful content
        valid_files = []
        removed_files = []

        for file_path in actual_files:
            duration = self.get_audio_duration(file_path)
            if duration < MIN_CHUNK_DURATION:
                log.warning(
                    f"Removing short audio chunk: {os.path.basename(file_path)} (duration={duration:.2f}s < {MIN_CHUNK_DURATION}s)"
                )
                os.remove(file_path)
                removed_files.append(os.path.basename(file_path))
            else:
                valid_files.append(file_path)

        if removed_files:
            log.info(
                f"Removed {len(removed_files)} short audio chunks: {removed_files}"
            )

        if progress_callback:
            progress_callback(1.0)

        log.info(
            f"Extracted {len(valid_files)} valid audio chunks (removed {len(removed_files)} too short)"
        )
        return valid_files

    def transcribe_audio(
        self, reproj: Reproj, audio_file: str, time_offset: float = 0.0
    ) -> tuple[str, List[TranscriptionSegment]]:
        """
        Transcribe the audio file and return the text and segments.

        Args:
            reproj: The reproj object.
            audio_file: The path to the audio file.
            time_offset: The time offset of the audio file.

        Returns:
            A tuple of the text and segments.
        """
        audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
        # location of the file is the same as the file_path
        target = reproj.get_file_working_dir(Reproj.Section.STT)
        transcript_file = os.path.join(target, audio_filename + ".transcript.json")
        text_file = os.path.join(target, audio_filename + ".txt")
        segments_file = os.path.join(target, audio_filename + ".seg")
        if os.path.exists(text_file) and os.path.exists(segments_file):
            log.info(f"Transcript and segments already exist for {audio_filename}")
            with open(text_file, "r") as f:
                text = f.read()
            with open(segments_file, "r") as f:
                ta = TypeAdapter(List[TranscriptionSegment])
                segments = ta.validate_json(f.read())
            return text, segments

        # # transcript
        if os.path.exists(transcript_file):
            log.info(f"Transcript already exists for {audio_filename}")
            with open(transcript_file, "r") as f:
                transcript = TranslationVerbose.model_validate_json(f.read())
        else:
            # Get audio file info for logging
            audio_duration = self.get_audio_duration(audio_file)
            audio_size = os.path.getsize(audio_file)

            log.info(f"Transcribing {audio_filename}")
            log.debug(f"  Audio file: {audio_file}")
            log.debug(f"  Duration: {audio_duration:.2f}s")
            log.debug(
                f"  File size: {audio_size:,} bytes ({audio_size / 1024 / 1024:.2f} MB)"
            )

            # Check if audio is too short
            if audio_duration < 0.1:
                log.error(
                    f"❌ Audio file is too short: {audio_duration:.3f}s (minimum 0.1s)"
                )
                log.error(f"  File: {audio_file}")
                raise ValueError(
                    f"Audio file {audio_filename} is too short ({audio_duration:.3f}s), minimum is 0.1s"
                )

            client = OpenAI(api_key=self.openai_token, base_url=self.openai_base_url or None)
            # https://platform.openai.com/docs/api-reference/audio/verbose-json-object
            try:
                with open(audio_file, "rb") as audio_file_buffer:
                    # Use translations.create only on the standard US endpoint.
                    # The EU endpoint (and any custom base_url) does not support it.
                    use_translation_api = (
                        self.target_language == "eng"
                        and not self.openai_base_url
                    )
                    if use_translation_api:
                        log.debug("Using Whisper translation API (target=eng, standard endpoint)")
                        transcript = client.audio.translations.create(
                            model=self.stt_model,
                            file=audio_file_buffer,
                            response_format="verbose_json",
                        )
                    else:
                        # transcriptions.create preserves source language; translate separately
                        log.debug(
                            f"Using Whisper transcription API + LLM translation (target={self.target_language})"
                        )

                        raw_transcription = client.audio.transcriptions.create(
                            model=self.stt_model,
                            file=audio_file_buffer,
                            response_format="verbose_json",
                        )
                        # Translate each segment text and the full text to target language
                        translated_text = self.translate_text_to(
                            raw_transcription.text, self.target_language
                        )
                        if raw_transcription.segments:
                            for seg in raw_transcription.segments:
                                seg.text = self.translate_text_to(
                                    seg.text, self.target_language
                                )
                        # Wrap into TranslationVerbose-compatible structure for uniform handling
                        transcript = TranslationVerbose(
                            text=translated_text,
                            task="translate",
                            language=self.target_language,
                            duration=raw_transcription.duration or 0.0,
                            segments=raw_transcription.segments,
                        )
                    log.debug(f"Transcript type: {type(transcript)}")
            except Exception as e:
                log.error(f"❌ Transcription failed for {audio_filename}")
                log.error(f"  File: {audio_file}")
                log.error(f"  Duration: {audio_duration:.2f}s")
                log.error(f"  Size: {audio_size:,} bytes")
                log.error(f"  Error: {e}")
                raise

            with open(transcript_file, "w") as f:
                f.write(transcript.model_dump_json())
            log.info(f"Transcript saved to {transcript_file}")

        transcript_segments: List[TranscriptionSegment] | None = transcript.segments
        if transcript_segments is None:
            log.error(f"Transcript segments are None for {audio_filename}")
            return transcript.text, []
        for segment in transcript_segments:
            segment.start += float(time_offset)
            segment.end += float(time_offset)

        with open(text_file, "w") as f:
            f.write(transcript.text)
        with open(segments_file, "w") as f:
            ta = TypeAdapter(List[TranscriptionSegment])
            json_compatible = ta.dump_python(transcript_segments)
            json.dump(json_compatible, f)

        return transcript.text, transcript_segments

    def time_to_srt_format(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    def write_srt(self, segments: List[TranscriptionSegment], output_file):
        with open(output_file, "w") as srt_file:
            for i, segment in enumerate(segments):
                start_time = segment.start
                end_time = segment.end
                text = segment.text

                # Convert time to SRT time format
                start_time_str = self.time_to_srt_format(start_time)
                end_time_str = self.time_to_srt_format(end_time)

                # Write to the file
                srt_file.write(f"{i + 1}\n")
                srt_file.write(f"{start_time_str} --> {end_time_str}\n")
                srt_file.write(f"{text}\n\n")

    def tts(self, text, output_file):
        client = OpenAI(api_key=self.openai_token, base_url=self.openai_base_url or None)
        log.debug(
            f"TTS request: voice={self.voice}, has_instructions={bool(self.voice_instructions)}, text_len={len(text)}"
        )

        # Validate text is not empty or too short
        if not text or len(text.strip()) == 0:
            log.error("❌ TTS text is empty!")
            raise ValueError("Cannot generate TTS for empty text")

        for attempt in range(3):
            try:
                # Use gpt-4o-mini-tts if we have voice instructions, otherwise use tts-1
                if self.voice_instructions:
                    response = client.audio.speech.create(
                        model="gpt-4o-mini-tts",
                        voice=self.voice,
                        input=text,
                        instructions=self.voice_instructions,
                        response_format="aac",  # AAC for hardware acceleration
                    )
                    response.stream_to_file(output_file)
                else:
                    # Fallback to tts-1 without instructions
                    with client.audio.speech.with_streaming_response.create(
                        model="tts-1",
                        voice=self.voice,
                        input=text,
                        speed=self.tts_speed,
                        response_format="aac",  # AAC for hardware acceleration
                    ) as response:
                        response.stream_to_file(output_file)
                return
            except Exception as e:
                log.error(f"❌ TTS attempt {attempt + 1}/3 failed")
                log.error(f"  Text length: {len(text)} chars")
                log.error(f"  Text: '{text[:200]}'")
                log.error(f"  Error: {e}")
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(1)
        log.error("❌ Failed to generate TTS after 3 attempts")
        raise Exception(f"Failed to generate TTS for text: '{text[:100]}...'")

    def translate_text_to(self, text: str, target_language: str) -> str:
        """Translate text to the specified language using GPT.

        Used when :attr:`target_language` is not 'eng' and Whisper's built-in
        translation (which only targets English) cannot be used.

        Args:
            text: Source text to translate.
            target_language: ISO 639-2/B language code for the desired output
                language (e.g. 'spa', 'fra', 'zho').

        Returns:
            Translated text. Falls back to the original text on any API error.
        """
        if not text or not text.strip():
            return text

        client = OpenAI(api_key=self.openai_token, base_url=self.openai_base_url or None)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Translate the following text to {target_language}. "
                            "Return only the translation, no explanations."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            return response.choices[0].message.content or text
        except Exception as e:
            log.error(f"❌ Translation to {target_language} failed: {e}")
            return text

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of an audio file in seconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except (ValueError, AttributeError):
            return 0.0

    def adjust_audio_speed(
        self, input_path: str, output_path: str, speed_factor: float
    ) -> str:
        """
        Adjust audio speed using ffmpeg atempo filter.
        Speed factor > 1.0 speeds up, < 1.0 slows down.
        """
        # atempo filter only supports 0.5 to 2.0, chain multiple for larger changes
        filters = []
        remaining = speed_factor
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining}")

        filter_str = ",".join(filters)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-filter:a",
            filter_str,
            "-vn",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def process_tts_segment(self, segment, output_dir, i) -> str:
        """
        Process a TTS segment and return the path to the audio file.
        Adjusts speed if generated audio is longer than segment duration.

        Args:
            segment: The transcription segment.
            output_dir: The directory to save the audio file.
            i: The index of the segment.

        Returns:
            The path to the audio file.
        """
        output_file = os.path.join(output_dir, f"{i:03d}.en{self.audio_ext}")
        temp_file = os.path.join(output_dir, f"{i:03d}.en.temp{self.audio_ext}")

        if os.path.exists(output_file):
            log.debug(f"TTS already exists for {i:03d}.en{self.audio_ext}")
            return f"{i:03d}.en{self.audio_ext}"

        # Calculate segment duration
        segment_duration = segment.end - segment.start

        # Log segment details
        log.debug(
            f"Segment {i:03d}: duration={segment_duration:.2f}s, start={segment.start:.2f}s, end={segment.end:.2f}s, text_len={len(segment.text)}"
        )

        # Generate TTS with error handling
        try:
            self.tts(segment.text, temp_file)
        except Exception as e:
            log.error(f"❌ TTS failed for segment {i:03d}")
            log.error(
                f"  Duration: {segment_duration:.3f}s (start={segment.start:.2f}s, end={segment.end:.2f}s)"
            )
            log.error(f"  Text length: {len(segment.text)} chars")
            log.error(f"  Text preview: {segment.text[:100]}")
            log.error(f"  Error: {e}")
            raise

        # Calculate target duration from segment
        target_duration = segment.end - segment.start
        actual_duration = self.get_audio_duration(temp_file)

        if actual_duration > 0 and actual_duration > target_duration:
            # Audio is too long - speed it up to fit
            speed_factor = actual_duration / target_duration
            # Cap speed factor to avoid making audio too fast (max 2x speed)
            speed_factor = min(speed_factor, 2.0)
            log.info(
                f"Segment {i}: adjusting speed {speed_factor:.2f}x ({actual_duration:.1f}s -> {target_duration:.1f}s)"
            )
            self.adjust_audio_speed(temp_file, output_file, speed_factor)
            os.remove(temp_file)
        else:
            # Audio fits - just rename
            os.rename(temp_file, output_file)

        return f"{i:03d}.en{self.audio_ext}"

    def tts_segments(
        self, reproj: Reproj, segments, progress_callback=None
    ) -> set[str]:
        """
        Process a list of TTS segments and return a dictionary of the paths to the audio files.

        Args:
            reproj: The reproj object.
            segments: The list of transcription segments.
            progress_callback: Optional callback(progress: float) where progress is 0.0-1.0.

        Returns:
            A list of the audio files.
        """
        from concurrent.futures import as_completed

        log.info(
            f"TTS segments({len(segments)}) for {reproj.file_path} using voice={self.voice}, has_instructions={bool(self.voice_instructions)}"
        )
        result = {}
        failed_segments = []
        output_dir = reproj.get_file_working_dir(Reproj.Section.TTS)
        total_segments = len(segments)

        with ThreadPoolExecutor(max_workers=self.tts_concurrency) as executor:
            futures = {
                executor.submit(self.process_tts_segment, segment, output_dir, i): i
                for i, segment in enumerate(segments)
            }

            if self.interactive:
                # Show progress bar when interactive mode is enabled
                with tqdm(
                    total=len(futures), desc="Processing TTS segments", unit="segment"
                ) as pbar:
                    for i, future in enumerate(as_completed(futures)):
                        try:
                            filename = future.result()
                            result[filename] = filename
                        except Exception as e:
                            segment_idx = futures[future]
                            log.error(
                                f"Failed to generate TTS for segment {segment_idx}: {e}"
                            )
                            failed_segments.append(segment_idx)
                        pbar.update(1)
                        if progress_callback:
                            progress_callback((i + 1) / total_segments)
            else:
                # Track progress as futures complete
                for i, future in enumerate(as_completed(futures)):
                    try:
                        filename = future.result()
                        result[filename] = filename
                    except Exception as e:
                        segment_idx = futures[future]
                        log.error(
                            f"Failed to generate TTS for segment {segment_idx}: {e}"
                        )
                        failed_segments.append(segment_idx)
                    if progress_callback:
                        progress_callback((i + 1) / total_segments)

        if failed_segments:
            log.warning(
                f"Failed to generate TTS for {len(failed_segments)} segments: {failed_segments}"
            )

        return set(result.values())

    def get_text_and_segments(
        self, reproj: Reproj, compact: bool = True, progress_callback=None
    ) -> List[TranscriptionSegment]:
        """Get the text and segments from the audio file.

        Args:
            reproj: The reproj object.
            compact: Whether to postprocess segments.
            progress_callback: Optional callback(progress: float) where progress is 0.0-1.0.

        Returns:
            A list of transcription segments.
        """

        # Audio extraction is 0-50% of this stage
        def audio_progress(p):
            if progress_callback:
                progress_callback(p * 0.5)

        audio_files = self.extract_audio_chunks(
            reproj, progress_callback=audio_progress
        )
        all_segments = []
        time_offset = 0.0

        # Create progress bar if interactive mode is enabled
        audio_file_iter = audio_files
        if self.interactive:
            audio_file_iter = tqdm(
                audio_files, desc="Transcribing audio files", unit="file"
            )

        # Transcription is 50-100% of this stage
        total_files = len(audio_files)
        for i, audio_file in enumerate(audio_file_iter):
            _text, segments = self.transcribe_audio(reproj, audio_file, time_offset)
            time_offset = segments[-1].end
            all_segments.extend(segments)
            if progress_callback:
                progress_callback(0.5 + ((i + 1) / total_files) * 0.5)

        if compact:
            all_segments = postprocess_segments(all_segments)
        return all_segments

    def generate_subtitles(
        self, reproj: Reproj, segments: List[TranscriptionSegment], replace=False
    ) -> str:
        """
        Generate subtitles for the video file.

        Args:
            reproj: The reproj object.
            segments: The list of transcription segments.
            replace: Whether to replace the existing subtitles.

        Returns:
            The path to the subtitles file.
        """
        filename = reproj.filename
        target = reproj.get_file_working_dir(Reproj.Section.SUBTITLES)
        target_file = os.path.join(target, filename + ".en.srt")
        # if replace is True, delete the existing subtitles
        if replace:
            log.info(f"Replacing subtitles for {filename}")
            if os.path.exists(target_file):
                os.remove(target_file)
        # if replace is False and the subtitles exist, return
        if os.path.exists(target_file):
            log.info(f"Subtitles already exist for {filename}")
            return target_file

        self.write_srt(segments, target_file)
        return target_file

    def assemble_long_audio(
        self,
        audio_dict: List[TranscriptionSegment],
        reproj: Reproj,
        duration,
        max_segments=50,  # Reduced from 600 - amix with many inputs is very slow
        progress_callback=None,
    ):
        """
        Assemble a long audio file from a list of transcription segments.
        If the number of segments is greater than max_segments, then split the audio into chunks of max_segments and assemble each chunk.
        If the number of segments is less than max_segments, then assemble the audio in one go.

        Args:
            audio_dict: A list of transcription segments.
            reproj: The reproj object.
            duration: The duration of the audio file.
            max_segments: The maximum number of segments to assemble.
            progress_callback: Optional callback(progress: float) where progress is 0.0-1.0.
        """
        if len(audio_dict) > max_segments:
            log.warning(
                f"Assembling long audio for {reproj.file_path} with {len(audio_dict)} segments"
            )
            working_dir = reproj.get_file_working_dir(
                Reproj.Section.TARGET_AUDIO_CHUNKS
            )
            output_file = os.path.join(
                working_dir, reproj.filename + ".en" + self.audio_ext
            )
            num_chunks = math.ceil(len(audio_dict) / max_segments)

            # Prepare chunk tasks
            chunk_tasks = []
            for i in range(0, len(audio_dict), max_segments):
                j = i // max_segments + 1
                index_suffix = f"_{j:03d}"
                input_file = os.path.join(
                    working_dir, reproj.filename + index_suffix + ".en" + self.audio_ext
                )
                indices = list(range(i, min(i + max_segments, len(audio_dict))))
                chunk_tasks.append((i, j, input_file, indices))

            # Process chunks in parallel
            from concurrent.futures import as_completed

            audio_file_indices = []
            max_workers = min(10, num_chunks)  # Limit to 10 parallel ffmpeg processes

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunk tasks
                future_to_chunk = {}
                for i, j, input_file, indices in chunk_tasks:
                    if os.path.exists(input_file):
                        log.info(f"Audio already exists for {input_file}")
                        audio_file_indices.append((i, audio_dict[i].start))
                    else:
                        future = executor.submit(
                            self.assemble_audio,
                            audio_dict,
                            reproj,
                            duration,
                            j,
                            indices,
                        )
                        future_to_chunk[future] = (i, input_file)

                # Wait for completion and track progress
                completed = 0
                for future in as_completed(future_to_chunk):
                    i, input_file = future_to_chunk[future]
                    try:
                        future.result()  # Raise any exceptions
                        audio_file_indices.append((i, audio_dict[i].start))
                    except Exception as e:
                        log.error(f"Error assembling chunk {input_file}: {e}")
                        raise
                    completed += 1
                    if progress_callback:
                        progress_callback(completed / num_chunks * 0.8)

            # Sort by original index to maintain order
            audio_file_indices.sort(key=lambda x: x[0])

            log.info(f"Mixing {len(audio_file_indices)} audio files to {output_file}")
            if os.path.exists(output_file):
                log.info(f"Audio already exists for {output_file}")
                return output_file
            input_args = []
            filter_complex_parts = []
            for i, _start_time in audio_file_indices:
                j = i // max_segments + 1
                index_suffix = f"_{j:03d}"
                input_file = os.path.join(
                    working_dir, reproj.filename + index_suffix + ".en" + self.audio_ext
                )
                input_args.extend(["-i", input_file])

            # Mix all inputs
            mix_inputs = "".join(f"[{i}]" for i in range(len(audio_file_indices)))
            filter_complex_parts.append(
                f"{mix_inputs}amix=inputs={len(audio_file_indices)}:normalize=0[mixed]"
            )

            # Trim and pad to exact duration
            filter_complex_parts.append(
                f"[mixed]atrim=end={duration},apad=whole_dur={duration}[final]"
            )

            filter_complex = ";".join(filter_complex_parts)

            cmd = [
                "ffmpeg",
                "-threads",
                "0",  # use all available threads
                *input_args,
                "-filter_complex",
                filter_complex,
                "-map",
                "[final]",
                "-acodec",
                get_aac_encoder(),  # Hardware accelerated on macOS
                "-b:a",
                "256k",  # AAC is more efficient
                "-ar",
                "44100",
                "-y",
                output_file,
            ]
            log.debug(f"Running command: {cmd}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log.error(f"ffmpeg stderr: {result.stderr}")
                log.error(f"ffmpeg stdout: {result.stdout}")
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )
            if progress_callback:
                progress_callback(1.0)
            return output_file

        else:
            result = self.assemble_audio(audio_dict, reproj, duration)
            if progress_callback:
                progress_callback(1.0)
            return result

    def assemble_audio(
        self,
        audio_dict: List[TranscriptionSegment],
        reproj: Reproj,
        duration: float,
        output_index: int | None = None,
        indices: List[int] | None = None,
    ) -> str:
        """
        Create a complex filter command for mixing audio files with delays

        Args:
            audio_dict: A list of transcription segments.
            reproj: The reproj object.
            duration: The duration of the final audio file.
            output_index: The index of the output file. If None, the output file will be named like the source file.
            indices: The indices of the segments to assemble. If None, all segments will be assembled.

        Returns:
            The path to the output file.
        """
        inputs = []
        filter_complex_parts = []
        index_suffix = f"_{output_index:03d}" if output_index is not None else ""
        tts_dir = reproj.get_file_working_dir(Reproj.Section.TTS)
        target_dir = reproj.get_file_working_dir(Reproj.Section.TARGET_AUDIO_CHUNKS)
        output_file = os.path.join(
            target_dir, reproj.filename + index_suffix + ".en" + self.audio_ext
        )
        log.info(
            f"Assembling audio for {reproj.filename} out of {len(audio_dict)} segments to {output_file}"
        )
        if indices is None:
            indices = list(range(len(audio_dict)))
        log.info(
            f"indices {indices[0]}..{indices[-1]}: {self.seconds_to_hms(audio_dict[indices[0]].start)} - {self.seconds_to_hms(audio_dict[indices[-1]].end)}"
        )
        if os.path.exists(output_file):
            log.info(f"Audio already exists for {output_file}")
            return output_file

        # Validate that all required TTS files exist
        missing_files = []
        for i in indices:
            input_file = f"{i:03d}.en{self.audio_ext}"
            input_path = os.path.join(tts_dir, input_file)
            if not os.path.exists(input_path):
                missing_files.append(input_file)

        if missing_files:
            error_msg = f"Cannot assemble audio - missing {len(missing_files)} TTS files: {missing_files[:10]}"
            if len(missing_files) > 10:
                error_msg += f" (and {len(missing_files) - 10} more)"
            log.error(error_msg)
            raise FileNotFoundError(error_msg)

        # adelay has a max delay limit (~8388 seconds), so use aevalsrc for silence + concat instead
        MAX_ADELAY_MS = 8000000  # ~8000 seconds to be safe

        for j, i in enumerate(indices):
            segment = audio_dict[i]
            start_time = segment.start
            input_file = f"{i:03d}.en{self.audio_ext}"
            input_path = os.path.join(tts_dir, input_file)
            inputs.extend(["-i", input_path])

            delay_ms = int(start_time * 1000)

            if delay_ms > MAX_ADELAY_MS:
                # Use anullsrc (null audio source) for silence - more efficient than aevalsrc
                silence_duration = start_time
                filter_complex_parts.append(
                    f"anullsrc=r=44100:cl=stereo:d={silence_duration}[silence{j}]"
                )
                filter_complex_parts.append(
                    f"[silence{j}][{j}]concat=n=2:v=0:a=1[delayed{j}]"
                )
            else:
                # Use adelay for smaller delays (faster)
                filter_complex_parts.append(
                    f"[{j}]adelay={delay_ms}|{delay_ms}[delayed{j}]"
                )

        # Mix all delayed inputs
        mix_inputs = "".join(f"[delayed{i}]" for i in range(len(indices)))
        filter_complex_parts.append(
            f"{mix_inputs}amix=inputs={len(indices)}:normalize=0[mixed]"
        )

        # Trim and pad to exact duration
        filter_complex_parts.append(
            f"[mixed]atrim=end={duration},apad=whole_dur={duration}[final]"
        )

        filter_complex = ";".join(filter_complex_parts)

        cmd = [
            "ffmpeg",
            "-threads",
            "0",  # use all available threads
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[final]",
            "-acodec",
            get_aac_encoder(),  # Hardware accelerated on macOS
            "-b:a",
            "256k",  # AAC is more efficient
            "-ar",
            "44100",
            "-y",
            output_file,
        ]
        log.debug(f"Running command: {cmd}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"ffmpeg stderr: {result.stderr}")
            log.error(f"ffmpeg stdout: {result.stdout}")
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        return output_file

    def mix_audio_with_video(
        self,
        reproj: Reproj,
        audio_file: str,
        output_file: str,
        # https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes
        # for example: "eng", "fra", "spa", "deu"
        languages: List[str],
    ):
        """
        Mix audio with video, with dubbed audio as first track and original audio as second track.

        Args:
            reproj: The reproj object.
            audio_file: The path to the dubbed audio file.
            output_file: The path to the output file.
            languages: List of language codes [original_language, dubbed_language].
                      First element is the original audio language.
                      Second element is the dubbed audio language.
        """
        log.info(
            f"Mixing video `{reproj.file_path}` with dubbed audio `{audio_file}` - languages: {languages}"
        )
        if os.path.exists(output_file):
            log.info(f"Audio with video already exists for {output_file}")
            return

        if len(languages) != 2:
            log.error(
                f"Expected 2 languages (original + dubbed), got {len(languages)}: {languages}"
            )
            raise ValueError(
                f"Expected 2 languages [original_lang, dubbed_lang], got {len(languages)}"
            )

        # Metadata for audio tracks - swapped to match the new track order
        metadata_args = [
            "-metadata:s:a:0",
            f"language={languages[1]}",  # Track 0: Dubbed audio (languages[1])
            "-metadata:s:a:1",
            f"language={languages[0]}",  # Track 1: Original audio (languages[0])
        ]

        cmd = [
            "ffmpeg",
            "-i",
            reproj.file_path,  # Input 0: original video with audio
            "-i",
            audio_file,  # Input 1: dubbed audio track
            "-threads",
            "0",  # Use all available threads
            "-c:v",
            "copy",  # Copy video stream without re-encoding
            "-c:a",
            "copy",  # Copy audio streams without re-encoding
            "-map",
            "0:v",  # Map video from input 0
            "-map",
            "1:a:0",  # Map dubbed audio from input 1 (becomes track a:0)
            "-map",
            "0:a:0",  # Map original audio from input 0 (becomes track a:1)
            *metadata_args,
            "-y",
            output_file,
        ]

        log.debug(f"Running ffmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"ffmpeg stderr: {result.stderr}")
            log.error(f"ffmpeg stdout: {result.stdout}")
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )


# Standalone functions for voice refinement feature


def extract_audio_sample(
    video_path: str, start_time: float, end_time: float, output_path: str
) -> str:
    """
    Extract an audio sample from a video file.

    Args:
        video_path: Path to the source video file.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        output_path: Path to save the extracted audio sample.

    Returns:
        Path to the extracted audio file.

    Raises:
        RuntimeError: If video has no audio stream or ffmpeg fails.
    """
    # First check if video has audio streams
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        video_path,
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if not probe_result.stdout.strip():
        raise RuntimeError(
            "Video file has no audio stream. Cannot extract audio sample."
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    duration = end_time - start_time
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file (must come early)
        "-ss",
        str(start_time),  # Seek before input for faster extraction
        "-i",
        video_path,
        "-t",
        str(duration),  # Duration (not -to when -ss is before -i)
        "-vn",  # No video
        "-acodec",
        get_aac_encoder(),  # Hardware accelerated on macOS
        "-b:a",
        "192k",  # AAC bitrate
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {result.stderr}")
    log.info(
        f"Extracted audio sample from {video_path} ({start_time}s - {end_time}s) to {output_path}"
    )
    return output_path


def analyze_voice_with_gpt4o(audio_path: str, openai_token: str) -> dict:
    """
    Analyze voice characteristics using GPT audio model.

    Args:
        audio_path: Path to the audio file to analyze.
        openai_token: OpenAI API token.

    Returns:
        Dictionary with 'recommended_voice' and 'voice_instructions' keys.
    """
    import base64

    client = OpenAI(api_key=openai_token)

    with open(audio_path, "rb") as audio_file:
        audio_content = base64.b64encode(audio_file.read()).decode("utf-8")

    system_prompt = """Analyze the voice in this audio sample. Return a JSON object with these fields:

1. "detected_gender": "male" or "female" - the gender of the speaker based on voice characteristics
2. "recommended_voice": one of [alloy, echo, fable, onyx, nova, shimmer] - the OpenAI TTS voice that best matches
3. "voice_instructions": detailed description for TTS generation (tone, pace, emotion, emphasis, pronunciation style)

OpenAI TTS Voice Characteristics:
- alloy: Neutral, balanced, versatile (works for any gender)
- echo: Warm, conversational male voice
- fable: British accent, expressive, storytelling (works for any gender)
- onyx: Deep, authoritative male voice
- nova: Friendly, upbeat female voice
- shimmer: Warm, gentle female voice

IMPORTANT: First determine the speaker's gender from pitch and vocal characteristics, then select a voice that matches that gender:
- For male speakers: prefer echo, onyx, or alloy
- For female speakers: prefer nova, shimmer, or alloy

Also consider: pitch, energy level, speaking pace, accent, emotional quality.
Return ONLY the JSON object, no additional text."""

    response = client.chat.completions.create(
        model="gpt-audio-mini",
        modalities=["text"],
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this voice sample:"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_content, "format": "m4a"},
                    },
                ],
            },
        ],
        temperature=0,
    )

    # Parse the JSON response
    response_text = response.choices[0].message.content
    log.info(f"GPT-4o voice analysis response: {response_text}")

    try:
        # Try to parse as JSON directly
        result = json.loads(response_text)
    except json.JSONDecodeError:
        # If parsing fails, try to extract JSON from the response
        import re

        json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # Fallback to defaults
            log.warning(f"Could not parse GPT-4o response as JSON: {response_text}")
            result = {
                "recommended_voice": "nova",
                "voice_instructions": "Speak in a natural, conversational tone.",
            }

    return {
        "recommended_voice": result.get("recommended_voice", "nova"),
        "voice_instructions": result.get("voice_instructions", ""),
    }


def generate_tts_sample(
    text: str, voice: str, voice_instructions: str, output_path: str, openai_token: str
) -> str:
    """
    Generate a TTS sample with voice instructions.

    Args:
        text: Text to convert to speech.
        voice: OpenAI TTS voice name (alloy, echo, fable, onyx, nova, shimmer).
        voice_instructions: Instructions for voice style/tone.
        output_path: Path to save the generated audio.
        openai_token: OpenAI API token.

    Returns:
        Path to the generated audio file.
    """
    client = OpenAI(api_key=openai_token)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use gpt-4o-mini-tts which supports instructions
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        instructions=voice_instructions,
        input=text,
        response_format="aac",  # AAC for hardware acceleration
    )

    with open(output_path, "wb") as f:
        f.write(response.content)

    log.info(f"Generated TTS sample to {output_path}")
    return output_path


# Pipeline finalization functions (Steps 8-11)


class ValidationError(Exception):
    """Raised when video validation fails."""

    pass


class ReplacementError(Exception):
    """Raised when original video replacement fails."""

    pass


def validate_video_file(path: str, reference: str | None = None) -> bool:
    """
    Validate that a video file is complete and playable.

    Validates file existence, minimum size, presence of video and audio streams,
    and optionally duration match with a reference file.

    Args:
        path: Path to the video file to validate.
        reference: Optional path to reference video for duration comparison.

    Returns:
        True if validation passes, False otherwise.

    Raises:
        subprocess.CalledProcessError: If ffprobe fails to run.
    """
    log.info(f"Validating video file: {path}")

    # Step 1: Check file exists and has minimum size
    if not os.path.exists(path):
        log.error(f"Validation failed: File does not exist - {path}")
        return False

    file_size = os.path.getsize(path)
    if file_size <= 1024:
        log.error(f"Validation failed: File too small ({file_size} bytes) - {path}")
        return False

    # Step 2: Run ffprobe to check video structure
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        log.error(f"Validation failed: ffprobe error - {e.stderr}")
        return False
    except json.JSONDecodeError as e:
        log.error(f"Validation failed: Invalid JSON from ffprobe - {e}")
        return False

    # Step 3: Check for video and audio streams
    streams = probe_data.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    has_audio = len(audio_streams) > 0

    if not has_video:
        log.error(f"Validation failed: No video stream found - {path}")
        return False

    if not has_audio:
        log.error(f"Validation failed: No audio stream found - {path}")
        return False

    # Step 3b: If this is a dubbed file (reference provided), require 2 audio tracks
    # with language metadata on both
    if reference:
        if len(audio_streams) < 2:
            log.error(
                f"Validation failed: Expected 2 audio tracks (dubbed + original), "
                f"found {len(audio_streams)} - {path}"
            )
            return False

        for i, stream in enumerate(audio_streams):
            lang = stream.get("tags", {}).get("language", "")
            if not lang or lang == "und":
                log.warning(
                    f"Audio track {i} has no language tag — metadata may be missing"
                )

        log.info(
            f"Audio tracks OK: {len(audio_streams)} tracks, "
            f"languages: {[s.get('tags', {}).get('language', 'und') for s in audio_streams]}"
        )

    # Step 4: If reference provided, compare durations (within 1%)
    if reference:
        video_duration = float(probe_data["format"]["duration"])

        # Get reference duration
        ref_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            reference,
        ]
        ref_result = subprocess.run(ref_cmd, capture_output=True, text=True, check=True)
        ref_duration = float(ref_result.stdout.strip())

        # Check if durations match within 1%
        duration_diff = abs(video_duration - ref_duration)
        max_diff = ref_duration * 0.01  # 1% tolerance

        if duration_diff > max_diff:
            log.error(
                f"Validation failed: Duration mismatch - "
                f"output={video_duration:.2f}s, reference={ref_duration:.2f}s, "
                f"diff={duration_diff:.2f}s (max={max_diff:.2f}s)"
            )
            return False

        log.info(
            f"Duration match OK: {video_duration:.2f}s (diff={duration_diff:.2f}s)"
        )

    log.info(
        f"Validation passed: video={has_video}, audio={has_audio}, size={file_size:,} bytes"
    )
    return True


def replace_original_video_file(
    original_path: str, redubbed_path: str, backup_dir: str = "redubber_backups"
) -> tuple[bool, str | None]:
    """
    Atomically replace original video with redubbed version, creating a backup.

    Uses os.replace() for POSIX atomic replacement. Checks disk space before
    proceeding and rolls back on failure.

    Args:
        original_path: Path to the original video file to replace.
        redubbed_path: Path to the redubbed video file.
        backup_dir: Absolute path or directory name (relative to original) for backups.

    Returns:
        Tuple of (success: bool, backup_path: str | None).
        backup_path is the path to the backup file on success, None on failure.

    Raises:
        OSError: If disk space check fails or file operations fail.
    """
    import shutil
    from datetime import datetime

    log.info(f"Replacing original video: {original_path}")

    # Accept either an absolute path or a name relative to the video directory
    original_dir = os.path.dirname(original_path)
    if os.path.isabs(backup_dir):
        backup_full_dir = backup_dir
    else:
        backup_full_dir = os.path.join(original_dir, backup_dir)
    os.makedirs(backup_full_dir, exist_ok=True)

    # Generate timestamped backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_basename = os.path.basename(original_path)
    name, ext = os.path.splitext(original_basename)
    backup_filename = f"{name}.{timestamp}{ext}"
    backup_path = os.path.join(backup_full_dir, backup_filename)

    try:
        # Check disk space: need 2x video size (backup + final)
        original_size = os.path.getsize(original_path)
        redubbed_size = os.path.getsize(redubbed_path)
        required_space = original_size + redubbed_size

        disk_usage = shutil.disk_usage(original_dir)
        available_space = disk_usage.free

        if available_space < required_space:
            log.error(
                f"Insufficient disk space: need {required_space:,} bytes, "
                f"have {available_space:,} bytes"
            )
            return (False, None)

        log.info(
            f"Disk space OK: need {required_space:,} bytes, have {available_space:,} bytes"
        )

        # Create backup (copy with metadata)
        log.info(f"Creating backup: {backup_path}")
        shutil.copy2(original_path, backup_path)

        # Atomic replacement: os.replace is atomic on POSIX systems
        log.info(f"Atomically replacing: {original_path}")
        os.replace(redubbed_path, original_path)

        log.info(f"Successfully replaced video, backup at: {backup_path}")
        return (True, backup_path)

    except Exception as e:
        log.error(f"Error during replacement: {e}")

        # Rollback: restore from backup if it exists
        if os.path.exists(backup_path):
            try:
                log.info("Rolling back: restoring from backup")
                os.replace(backup_path, original_path)
                log.info("Rollback successful")
            except Exception as rollback_error:
                log.error(f"CRITICAL: Rollback failed: {rollback_error}")
                log.error(f"Manual recovery needed: backup is at {backup_path}")

        return (False, None)


def sync_video_metadata(db, project_id: int, video_path: str) -> None:
    """
    Re-detect and sync video metadata to database after redubbing.

    Updates video language, audio streams, and project timestamp in the database.

    Args:
        db: DatabaseManager instance.
        project_id: Project ID in the database.
        video_path: Path to the video file to analyze.

    Raises:
        Exception: If language detection or database update fails.
    """
    from utils import detect_video_language
    from pathlib import Path

    log.info(f"Syncing metadata for video: {video_path}")

    from video_analyzer import get_video_info_with_duration

    # Re-detect language from filename
    detected_lang = detect_video_language(Path(video_path))
    log.info(f"Detected language: {detected_lang}")

    from utils import detect_subtitle_language

    # Get full video info (audio streams, duration, size)
    video_info = get_video_info_with_duration(Path(video_path))
    audio_streams = video_info["audio_streams"]
    duration_seconds = video_info["duration_seconds"]
    size_mb = round(Path(video_path).stat().st_size / (1024 * 1024), 2)
    log.info(f"Audio streams: {audio_streams}, duration: {duration_seconds:.2f}s, size: {size_mb} MB")

    # Detect subtitle files alongside the video (e.g. <name>.en.srt placed by finalization)
    video_dir = Path(video_path).parent
    video_stem = Path(video_path).stem
    subtitle_exts = {".srt", ".vtt", ".ass", ".ssa", ".sub"}
    found_subs = [
        f for f in video_dir.iterdir()
        if f.stem.startswith(video_stem) and f.suffix.lower() in subtitle_exts
    ]
    subtitle_matches = []
    for sub_path in sorted(found_subs):
        sub_lang = detect_subtitle_language(sub_path)
        subtitle_matches.append({
            "language": sub_lang or "",
            "embedded": False,
            "path": str(sub_path),
            "filename": sub_path.name,
        })
        # Register in subtitle_files table if not already present
        db.add_subtitle_file(
            project_id=project_id,
            file_path=str(sub_path),
            filename=sub_path.name,
            language=sub_lang,
        )
    log.info(f"Subtitle files: {[s['filename'] for s in subtitle_matches]}")

    # Update database
    import sqlite3

    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()

        # Update video_files table
        cursor.execute(
            """
            UPDATE video_files
            SET language = ?
            WHERE project_id = ? AND file_path = ?
        """,
            (detected_lang, project_id, video_path),
        )

        # Update video_analysis table — refresh all mutable fields including subtitle_matches
        cursor.execute(
            """
            UPDATE video_analysis
            SET audio_streams = ?, duration_seconds = ?, size_mb = ?, subtitle_matches = ?
            WHERE project_id = ? AND file_path = ?
        """,
            (json.dumps(audio_streams), duration_seconds, size_mb,
             json.dumps(subtitle_matches), project_id, video_path),
        )

        # Update projects table timestamp
        cursor.execute(
            """
            UPDATE projects
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """,
            (project_id,),
        )

        conn.commit()

    log.info("Metadata sync complete")


def cleanup_temp_files(reproj: Reproj) -> None:
    """
    Clean up temporary files created during redubbing process.

    Removes temporary audio chunks, transcripts, and intermediate files.

    Args:
        reproj: The reproj object containing working directory paths.
    """
    import shutil

    log.info(f"Cleaning up temporary files for: {reproj.file_path}")

    sections_to_clean = [
        Reproj.Section.SOURCE_AUDIO_CHUNKS,
        Reproj.Section.STT,
        Reproj.Section.TTS,
        Reproj.Section.TARGET_AUDIO_CHUNKS,
    ]

    total_freed = 0

    for section in sections_to_clean:
        section_dir = reproj.get_file_working_dir(section)

        if os.path.exists(section_dir):
            # Calculate size before deletion
            dir_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(section_dir)
                for filename in filenames
            )

            # Remove directory
            shutil.rmtree(section_dir)
            total_freed += dir_size

            log.info(f"Removed {section.value}: freed {dir_size / 1024 / 1024:.2f} MB")

    log.info(f"Cleanup complete: freed {total_freed / 1024 / 1024:.2f} MB total")


def finalize_redubbing(
    db,
    reproj: Reproj,
    final_video_path: str,
    project_id: int,
    replace_original: bool = True,
    target_language: str = "eng",
) -> str:
    """
    Finalize redubbing pipeline: validate, replace, copy subs, sync metadata, cleanup.

    Steps:
    - Step 8: Validate output video (streams, duration, 2 audio tracks)
    - Step 9: Replace original with backup (if enabled)
    - Step 9b: Copy subtitle file next to the original as <name>.<target_lang>.srt
    - Step 10: Sync metadata to database
    - Step 11: Cleanup temporary files (frees disk space)

    Args:
        db: DatabaseManager instance.
        reproj: The reproj object.
        final_video_path: Path to the final redubbed video.
        project_id: Project ID in the database.
        replace_original: Whether to replace the original file (default: True).
        target_language: ISO 639-2/3 code for the dubbed language (e.g. 'eng').

    Returns:
        Path to the final video (either original location or final_video_path).

    Raises:
        ValidationError: If output validation fails.
        ReplacementError: If file replacement fails.
    """
    log.info("=" * 80)
    log.info("FINALIZATION PIPELINE (Steps 8-11)")
    log.info("=" * 80)

    # Step 8: Validate output video
    log.info("Step 8: Validating output video")
    if not validate_video_file(final_video_path, reference=reproj.file_path):
        raise ValidationError(
            f"Output validation failed: {final_video_path}\n"
            f"Video may be corrupted or incomplete. Original file unchanged."
        )
    log.info("✓ Step 8 complete: Validation passed")

    # If not replacing, we're done
    if not replace_original:
        log.info("replace_original=False, skipping steps 9-11")
        return final_video_path

    import shutil as _shutil

    # Backup lives inside the project working directory, not next to the video
    backup_dir_path = os.path.join(reproj.root, "backups")
    os.makedirs(backup_dir_path, exist_ok=True)

    # Step 9: Replace original with backup (backup goes to working dir)
    log.info("Step 9: Replacing original video with backup")
    success, backup_path = replace_original_video_file(
        reproj.file_path, final_video_path, backup_dir=backup_dir_path
    )

    if not success:
        raise ReplacementError(
            f"Failed to replace original video: {reproj.file_path}\n"
            f"Redubbed video is available at: {final_video_path}"
        )

    log.info(f"✓ Step 9 complete: Original replaced, backup at {backup_path}")

    # Step 9b: Copy subtitle file next to original as <name>.<lang2>.srt
    _ISO639_2_TO_1: dict[str, str] = {
        "eng": "en", "jpn": "ja", "fra": "fr", "spa": "es", "deu": "de",
        "ita": "it", "por": "pt", "rus": "ru", "zho": "zh", "kor": "ko",
        "ara": "ar", "hin": "hi", "nld": "nl", "pol": "pl", "swe": "sv",
        "nor": "no", "dan": "da", "fin": "fi", "tur": "tr", "vie": "vi",
        "tha": "th", "ind": "id", "msa": "ms", "ces": "cs", "slk": "sk",
        "hun": "hu", "ron": "ro", "bul": "bg", "ukr": "uk", "hrv": "hr",
    }
    lang2 = _ISO639_2_TO_1.get(target_language.lower(), target_language[:2].lower())

    subs_copied = False
    log.info("Step 9b: Copying subtitle file")
    srt_src = os.path.join(
        reproj.get_file_working_dir(Reproj.Section.SUBTITLES),
        reproj.filename + ".en.srt",
    )
    if os.path.exists(srt_src):
        video_dir = os.path.dirname(reproj.file_path)
        srt_dst = os.path.join(video_dir, f"{reproj.filename}.{lang2}.srt")
        try:
            _shutil.copy2(srt_src, srt_dst)
            subs_copied = True
            log.info(f"✓ Step 9b complete: Subtitles copied to {srt_dst}")
        except Exception as e:
            log.warning(f"Step 9b: Could not copy subtitles: {e}")
    else:
        log.warning(f"Step 9b: Subtitle file not found at {srt_src}, skipping")

    # Step 10: Sync metadata to database
    log.info("Step 10: Syncing metadata to database")
    sync_video_metadata(db, project_id, reproj.file_path)
    log.info("✓ Step 10 complete: Metadata synced")

    # Step 11: Cleanup — only if subs were copied (or there were none to copy)
    subs_ok = subs_copied or not os.path.exists(srt_src)
    if subs_ok:
        log.info("Step 11: Cleaning up temporary files")
        cleanup_temp_files(reproj)
        log.info("✓ Step 11 complete: Temp files cleaned")
    else:
        log.warning(
            "Step 11: Skipping cleanup — subtitle copy failed. "
            "Re-run finalization once subtitles are available."
        )

    log.info("=" * 80)
    log.info("FINALIZATION COMPLETE")
    log.info(f"Final video: {reproj.file_path}")
    log.info(f"Backup: {backup_path}")
    log.info("=" * 80)

    return reproj.file_path
