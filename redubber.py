import os
import math
import subprocess
import json
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

    audio_ext: str = ".mp3"
    model: str = "gpt-4o"
    openai_token: str = ""
    default_audio_chunk_duration: int = 20 * 60  # 20 minutes
    interactive: bool = False
    voice: str = "nova"
    voice_instructions: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, openai_token: str, interactive: bool = False, voice: str = "nova", voice_instructions: str = ""):
        """Initialize the Redubber class.

        Args:
            openai_token: The OpenAI API token.
            interactive: Whether to show progress bars for long-running operations.
            voice: The TTS voice to use (alloy, echo, fable, onyx, nova, shimmer).
            voice_instructions: Instructions for voice style/tone.
        """
        super().__init__(openai_token=openai_token, interactive=interactive, voice=voice, voice_instructions=voice_instructions)

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
        chunk_duration: int = default_audio_chunk_duration,
        replace: bool = False,
        progress_callback=None,
    ) -> list[str]:
        """Extract audio chunks from a video file.

        Args:
            reproj: The reproj object.
            chunk_duration: The duration of each audio chunk in seconds.
            replace: Whether to replace the existing audio chunks.
            progress_callback: Optional callback(progress: float) where progress is 0.0-1.0.

        Returns:
            A list of paths to the audio chunks that are extracted and saved in SOURCE_AUDIO_CHUNKS section.
        """
        log.info(f"Extracting audio from {reproj.file_path}")
        # keep all audio files in the directory, that is constructed out of rel_file
        # E.g. if rel_file is "test.mp4" then target_rel_dir is "redubber_tmp/test.mp4/"
        target_rel_dir = reproj.get_file_working_dir(Reproj.Section.SOURCE_AUDIO_CHUNKS)
        total_duration = self.get_media_duration(reproj.file_path)
        log.info(f"Video duration {self.seconds_to_hms(total_duration)}")
        num_chunks = math.ceil(total_duration / chunk_duration)
        log.info(
            f"Extracting {num_chunks} chunks of {self.seconds_to_hms(chunk_duration)} each"
        )

        audio_file_template = (
            os.path.splitext(os.path.basename(reproj.file_path))[0]
            + "_{:03d}"
            + self.audio_ext
        )
        # delete all mp3 files in the directory
        if replace:
            for source, _dirs, files in os.walk(target_rel_dir):
                for file in files:
                    if file.endswith(self.audio_ext):
                        os.remove(os.path.join(source, file))

        total_audio_duration = 0.0
        result = []

        # Create progress bar if interactive mode is enabled
        chunk_range = range(num_chunks)
        if self.interactive:
            chunk_range = tqdm(
                chunk_range, desc="Extracting audio chunks", unit="chunk"
            )

        for i in chunk_range:
            start_time = i * chunk_duration
            output_audio_path = audio_file_template.format(i + 1)  # Naming each chunk
            audio_path = os.path.join(target_rel_dir, output_audio_path)

            # Check if the audio file already exists
            if os.path.exists(audio_path):
                log.info(f"Audio file {audio_path} already exists")
                result.append(audio_path)
                total_audio_duration += self.get_media_duration(audio_path)
                if progress_callback:
                    progress_callback((i + 1) / num_chunks)
                continue

            os.makedirs(target_rel_dir, exist_ok=True)
            # Use subprocess to call ffmpeg directly
            cmd = [
                "ffmpeg",
                "-i",
                reproj.file_path,
                "-ss",
                str(start_time),
                "-t",
                str(chunk_duration),
                "-vn",
                "-acodec",
                "libmp3lame",
                "-y",
                audio_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            log.info(f"Extracted chunk {i + 1}: {output_audio_path}")
            result.append(audio_path)
            total_audio_duration += self.get_media_duration(audio_path)

            if progress_callback:
                progress_callback((i + 1) / num_chunks)

        log.info(f"Audio duration {self.seconds_to_hms(total_audio_duration)}")

        return result

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
            log.info(f"Transcribing {audio_filename}")
            client = OpenAI(api_key=self.openai_token)
            # https://platform.openai.com/docs/api-reference/audio/verbose-json-object
            with open(audio_file, "rb") as audio_file_buffer:
                # Transcribe the audio using the OpenAI API transcribe model
                transcript = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file_buffer,
                    response_format="verbose_json",
                )
                log.debug(f"Transcript type: {type(transcript)}")
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
        client = OpenAI(api_key=self.openai_token)
        log.info(f"TTS using voice={self.voice}, has_instructions={bool(self.voice_instructions)}")
        for _ in range(3):
            try:
                # Use gpt-4o-mini-tts if we have voice instructions, otherwise use tts-1
                if self.voice_instructions:
                    response = client.audio.speech.create(
                        model="gpt-4o-mini-tts",
                        voice=self.voice,
                        input=text,
                        instructions=self.voice_instructions,
                        response_format="mp3",
                    )
                    response.stream_to_file(output_file)
                else:
                    # Fallback to tts-1 without instructions
                    with (
                        client.audio.speech.with_streaming_response.create(
                            model="tts-1",
                            voice=self.voice,
                            input=text,
                            speed=1.25,
                        ) as response
                    ):
                        response.stream_to_file(output_file)
                return
            except Exception as e:
                log.error(f"Error generating TTS for {text}: {e}")
                time.sleep(1)
        log.error(f"Failed to generate TTS for {text}")
        raise Exception(f"Failed to generate TTS for {text}")

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of an audio file in seconds using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except (ValueError, AttributeError):
            return 0.0

    def adjust_audio_speed(self, input_path: str, output_path: str, speed_factor: float) -> str:
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
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter:a", filter_str,
            "-vn",
            output_path
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
        output_file = os.path.join(output_dir, f"{i:03d}.en.mp3")
        temp_file = os.path.join(output_dir, f"{i:03d}.en.temp.mp3")

        if os.path.exists(output_file):
            log.debug(f"TTS already exists for {i:03d}.en.mp3")
            return f"{i:03d}.en.mp3"

        # Generate TTS
        self.tts(segment.text, temp_file)

        # Calculate target duration from segment
        target_duration = segment.end - segment.start
        actual_duration = self.get_audio_duration(temp_file)

        if actual_duration > 0 and actual_duration > target_duration:
            # Audio is too long - speed it up to fit
            speed_factor = actual_duration / target_duration
            # Cap speed factor to avoid making audio too fast (max 2x speed)
            speed_factor = min(speed_factor, 2.0)
            log.info(f"Segment {i}: adjusting speed {speed_factor:.2f}x ({actual_duration:.1f}s -> {target_duration:.1f}s)")
            self.adjust_audio_speed(temp_file, output_file, speed_factor)
            os.remove(temp_file)
        else:
            # Audio fits - just rename
            os.rename(temp_file, output_file)

        return f"{i:03d}.en.mp3"

    def tts_segments(self, reproj: Reproj, segments, progress_callback=None) -> set[str]:
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

        log.info(f"TTS segments({len(segments)}) for {reproj.file_path} using voice={self.voice}, has_instructions={bool(self.voice_instructions)}")
        result = {}
        output_dir = reproj.get_file_working_dir(Reproj.Section.TTS)
        n_threads = 20
        total_segments = len(segments)

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(self.process_tts_segment, segment, output_dir, i)
                for i, segment in enumerate(segments)
            ]

            if self.interactive:
                # Show progress bar when interactive mode is enabled
                with tqdm(
                    total=len(futures), desc="Processing TTS segments", unit="segment"
                ) as pbar:
                    for i, future in enumerate(as_completed(futures)):
                        result[future.result()] = future.result()
                        pbar.update(1)
                        if progress_callback:
                            progress_callback((i + 1) / total_segments)
            else:
                # Track progress as futures complete
                for i, future in enumerate(as_completed(futures)):
                    result[future.result()] = future.result()
                    if progress_callback:
                        progress_callback((i + 1) / total_segments)

        return set(result.values())

    def get_text_and_segments(self, reproj: Reproj, compact: bool = True, progress_callback=None) -> List[TranscriptionSegment]:
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

        audio_files = self.extract_audio_chunks(reproj, progress_callback=audio_progress)
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

    def seconds_to_hms(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def assemble_long_audio(
        self,
        audio_dict: List[TranscriptionSegment],
        reproj: Reproj,
        duration,
        max_segments=600,
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
            audio_file_indices = []
            working_dir = reproj.get_file_working_dir(
                Reproj.Section.TARGET_AUDIO_CHUNKS
            )
            output_file = os.path.join(working_dir, reproj.filename + ".en.mp3")
            num_chunks = math.ceil(len(audio_dict) / max_segments)
            for chunk_idx, i in enumerate(range(0, len(audio_dict), max_segments)):
                j = i // max_segments + 1
                index_suffix = f"_{j:03d}"
                input_file = os.path.join(
                    working_dir, reproj.filename + index_suffix + ".en.mp3"
                )
                if os.path.exists(input_file):
                    log.info(f"Audio already exists for {input_file}")
                else:
                    self.assemble_audio(
                        audio_dict, reproj, duration, j, indices=list(range(i, min(i + max_segments, len(audio_dict)-1)))
                    )
                audio_file_indices.append((i, audio_dict[i].start))
                # Progress: chunk assembly is 0-80%, final mix is 80-100%
                if progress_callback:
                    progress_callback((chunk_idx + 1) / num_chunks * 0.8)

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
                    working_dir, reproj.filename + index_suffix + ".en.mp3"
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
                "-threads", "0", # use all available threads
                *input_args,
                "-filter_complex",
                filter_complex,
                "-map",
                "[final]",
                "-acodec",
                "libmp3lame", # CPU intensive codec
                "-q:a", "5",  # Faster MP3 encoding
                "-b:a", "320k",
                "-ar", "44100",
                "-y",
                output_file,
            ]
            log.debug(f"Running command: {cmd}")
            subprocess.run(cmd, check=True, capture_output=True)
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
            target_dir, reproj.filename + index_suffix + ".en.mp3"
        )
        log.info(
            f"Assembling audio for {reproj.filename} out of {len(audio_dict)} segments to {output_file}"
        )
        if indices is None:
            indices = list(range(len(audio_dict)))
        log.info(f"indices {indices[0]}..{indices[-1]}: {self.seconds_to_hms(audio_dict[indices[0]].start)} - {self.seconds_to_hms(audio_dict[indices[-1]].end)}")
        if os.path.exists(output_file):
            log.info(f"Audio already exists for {output_file}")
            return output_file

        for j, i in enumerate(indices):
            segment = audio_dict[i]
            
            start_time = segment.start
            input_file = f"{i:03d}.en.mp3"
            input_path = os.path.join(tts_dir, input_file)
            inputs.extend(["-i", input_path])
            # Add delay filter for each input
            delay_ms = int(start_time * 1000)
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
            "-threads", "0", # use all available threads
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[final]",
            "-acodec",
            "libmp3lame", # CPU intensive codec
            "-q:a", "5",  # Faster MP3 encoding
            "-b:a", "320k",
            "-ar", "44100",
            "-y",
            output_file,
        ]
        log.debug(f"Running command: {cmd}")
        subprocess.run(cmd, check=True, capture_output=True)
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
        Mix audio with video.

        Args:
            reproj: The reproj object.
            audio_file: The path to the audio file.
            output_file: The path to the output file.
            languages: The list of languages.
        """
        log.info(
            f"Mixing video `{reproj.file_path}` with audio `{audio_file}` out of {len(languages)} languages"
        )
        if os.path.exists(output_file):
            log.info(f"Audio with video already exists for {output_file}")
            return

        audio_streams = self.get_media_audio_streams(reproj.file_path)
        if len(audio_streams) != len(languages) - 1:
            log.error(
                f"Number of audio streams ({len(audio_streams)}) does not match number of languages ({len(languages)})"
            )
            raise Exception(
                f"Number of audio streams ({len(audio_streams)}) does not match number of languages ({len(languages)})"
            )

        args = []
        for i, language in enumerate(languages):
            args.append(f"-metadata:s:a:{i}")
            args.append(f"language={language}")

        cmd = [
            "ffmpeg",
            "-i", reproj.file_path,
            "-i", audio_file,
            "-threads", "0", # use all available threads
            "-c:v", "copy",
            "-c:a",
            #"aac", # re-encodes the audio to AAC codec
            #"aac_at" # Use Apple hardware AAC encoder
            "copy", # keeps the original audio codec
            "-b:a", "192k",
            "-map", "0",
            "-map", "1",
            *args,
            "-y",
            output_file,
        ]
        subprocess.run(cmd, check=True, capture_output=True)


# Standalone functions for voice refinement feature

def extract_audio_sample(video_path: str, start_time: float, end_time: float, output_path: str) -> str:
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
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        video_path
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if not probe_result.stdout.strip():
        raise RuntimeError("Video file has no audio stream. Cannot extract audio sample.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    duration = end_time - start_time
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file (must come early)
        "-ss", str(start_time),  # Seek before input for faster extraction
        "-i", video_path,
        "-t", str(duration),  # Duration (not -to when -ss is before -i)
        "-vn",  # No video
        "-acodec", "libmp3lame",
        "-q:a", "2",  # Good quality
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {result.stderr}")
    log.info(f"Extracted audio sample from {video_path} ({start_time}s - {end_time}s) to {output_path}")
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

    with open(audio_path, 'rb') as audio_file:
        audio_content = base64.b64encode(audio_file.read()).decode('utf-8')

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
        model='gpt-4o-audio-preview',
        modalities=["text"],
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this voice sample:"},
                    {"type": "input_audio", "input_audio": {"data": audio_content, "format": "mp3"}}
                ]
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
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # Fallback to defaults
            log.warning(f"Could not parse GPT-4o response as JSON: {response_text}")
            result = {
                'recommended_voice': 'nova',
                'voice_instructions': 'Speak in a natural, conversational tone.'
            }

    return {
        'recommended_voice': result.get('recommended_voice', 'nova'),
        'voice_instructions': result.get('voice_instructions', '')
    }


def generate_tts_sample(text: str, voice: str, voice_instructions: str, output_path: str, openai_token: str) -> str:
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
        response_format="mp3"
    )

    with open(output_path, 'wb') as f:
        f.write(response.content)

    log.info(f"Generated TTS sample to {output_path}")
    return output_path
