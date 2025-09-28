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

logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
log = logging.getLogger(__name__)


class Redubber:
    supported_video_formats = [".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".vob", ".m4v", ".3gp", ".3g2", ".m2ts", ".mts", ".ts", ".f4v", ".f4p",
                               ".f4a", ".f4b", ".m2v", ".m4v", ".m1v", ".mpg", ".mpeg", ".mpv", ".mp2", ".mpe", ".m2p", ".m2t", ".mp2v", ".mpv2", ".m2ts", ".m2ts", ".mts", ".m2v"]
    tmp = "redubber_tmp"
    audio_ext = ".mp3"
    model = "gpt-4o"
    openai_token = ""
    default_audio_chunk_duration = 20*60  # 20 minutes

    def can_redub(self, source):
        result = os.path.splitext(source)[1] in self.supported_video_formats
        log.debug(f"Can redub {source}: {result}")
        return result

    def get_media_duration(self, file_path) -> float:
        log.debug(f"Getting media duration for {file_path}")
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', '-show_streams', file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        return float(probe_data['format']['duration'])

    def get_media_audio_streams(self, file_path) -> List[str]:
        log.debug(f"Getting media audio streams for {file_path}")
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log.debug(f"FFprobe result: {result.stdout}")
        probe_data = json.loads(result.stdout)
        return [stream['tags']['language'] for stream in probe_data['streams'] if stream['codec_type'] == 'audio']

    def seconds_to_hms(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def get_file_working_dir(self, source, file_path):
        rel_file = os.path.relpath(file_path, source)
        return os.path.join(self.tmp, rel_file)

    def extract_audio_chunks(self, source, file_path, chunk_duration=default_audio_chunk_duration, replace=False) -> list[str]:
        """Extract audio chunks from a video file.

        Args:
            source: The source directory of the video file.
            file_path: The path to the video file.
            chunk_duration: The duration of each audio chunk in seconds.
            replace: Whether to replace the existing audio chunks.

        Returns:
            A list of paths to the audio chunks.
        """
        log.info(f"Extracting audio from {file_path}")
        # keep all audio files in the directory, that is constructed out of rel_file
        # E.g. if rel_file is "test.mp4" then target_rel_dir is "redubber_tmp/test.mp4/"
        target_rel_dir = self.get_file_working_dir(source, file_path)
        total_duration = self.get_media_duration(file_path)
        log.info(f"Video duration {self.seconds_to_hms(total_duration)}")
        num_chunks = math.ceil(total_duration / chunk_duration)
        log.info(f"Extracting {num_chunks} chunks of {self.seconds_to_hms(chunk_duration)} each")

        audio_file_template = os.path.splitext(os.path.basename(file_path))[
            0] + "_{:03d}" + self.audio_ext
        # delete all mp3 files in the directory
        if replace:
            for source, _dirs, files in os.walk(target_rel_dir):
                for file in files:
                    if file.endswith(self.audio_ext):
                        os.remove(os.path.join(source, file))

        total_audio_duration = 0.0
        result = []
        for i in range(num_chunks):
            start_time = i * chunk_duration
            output_audio_path = audio_file_template.format(i+1)  # Naming each chunk
            audio_path = os.path.join(target_rel_dir, output_audio_path)

            # Check if the audio file already exists
            if os.path.exists(audio_path):
                log.info(f"Audio file {audio_path} already exists")
                result.append(audio_path)
                total_audio_duration += self.get_media_duration(audio_path)
                continue

            os.makedirs(target_rel_dir, exist_ok=True)
            # Use subprocess to call ffmpeg directly
            cmd = [
                'ffmpeg', '-i', file_path, '-ss', str(start_time), 
                '-t', str(chunk_duration), '-vn', '-acodec', 'libmp3lame',
                '-y', audio_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            log.info(f'Extracted chunk {i+1}: {output_audio_path}')
            result.append(audio_path)
            total_audio_duration += self.get_media_duration(audio_path)

        log.info(f"Audio duration {self.seconds_to_hms(total_audio_duration)}")

        return result

    def transcribe_audio(self, file_path, time_offset: float=0.0) -> tuple[str, List[TranscriptionSegment]]:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        # location of the file is the same as the file_path
        target = os.path.dirname(file_path)
        transcript_file = os.path.join(target, filename + ".transcript.json")
        text_file = os.path.join(target, filename + ".txt")
        segments_file = os.path.join(target, filename + ".seg")
        if os.path.exists(text_file) and os.path.exists(segments_file):
            log.info(f"Transcript and segments already exist for {filename}")
            with open(text_file, 'r') as f:
                text = f.read()
            with open(segments_file, 'r') as f:
                ta = TypeAdapter(List[TranscriptionSegment])
                segments = ta.validate_json(f.read())
            return text, segments

        # # transcript
        if os.path.exists(transcript_file):
            log.info(f"Transcript already exists for {filename}")
            with open(transcript_file, 'r') as f:
                transcript = TranslationVerbose.model_validate_json(f.read())
        else:
            log.info(f"Transcribing {filename}")
            client = OpenAI(api_key=self.openai_token)
            # https://platform.openai.com/docs/api-reference/audio/verbose-json-object
            with open(file_path, "rb") as audio_file:
                # Transcribe the audio using the Whisper API
                transcript = client.audio.translations.create(
                    model="whisper-1", file=audio_file, response_format='verbose_json')
                log.info(f"Transcript type: {type(transcript)}")
            with open(transcript_file, 'w') as f:
                f.write(transcript.model_dump_json())
            log.info(f"Transcript saved to {transcript_file}")


        transcript_segments: List[TranscriptionSegment] | None = transcript.segments
        if transcript_segments is None:
            log.error(f"Transcript segments are None for {filename}")
            return transcript.text, []
        for segment in transcript_segments:
            segment.start += float(time_offset)
            segment.end += float(time_offset)

        with open(text_file, 'w') as f:
            f.write(transcript.text)
        with open(segments_file, 'w') as f:
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
        with open(output_file, 'w') as srt_file:
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
        for _ in range(3):
            try:
                # the speed is from 0.25 to 4.0 wtih the 1.0 being the default
                with client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="nova",
                    input=text,
                    speed = 1.25,  # todo: calculate speed based on the number of words and the desired duration
                ) as response:
                    response.stream_to_file(output_file)
                return
            except Exception as e:
                log.error(f"Error generating TTS for {text}: {e}")
                time.sleep(1)
        log.error(f"Failed to generate TTS for {text}")
        raise Exception(f"Failed to generate TTS for {text}")
        
    
    def process_tts_segment(self, segment, output_dir, i):
        if os.path.exists(os.path.join(output_dir, f"{i:03d}.en.mp3")):
            log.debug(f"TTS already exists for {i:03d}.en.mp3")
        else:    
            self.tts(segment.text, os.path.join(
                output_dir, f"{i:03d}.en.mp3"))
        return f"{i:03d}.en.mp3"

    def tts_segments(self, segments, source, src_file):
        log.info(f"TTS segments({len(segments)}) for {src_file}")
        result = {}
        output_dir = self.get_file_working_dir(source, src_file)
        n_threads = 10
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(self.process_tts_segment, segment, output_dir, i) for i, segment in enumerate(segments)]
            for future in futures:
                result[future.result()] = future.result()
        return result

    def get_text_and_segments(self, source, src_file) -> List[TranscriptionSegment]:
        """Get the text and segments from the audio file.

        Args:
            source: The source directory of the video file.
            src_file: The path to the video file.

        Returns:
            A list of transcription segments.
        """
        audio_files = self.extract_audio_chunks(source, src_file)
        all_segments = []
        time_offset = 0.0
        for audio_file in audio_files:
            _text, segments = self.transcribe_audio(audio_file, time_offset)
            time_offset = segments[-1].end
            all_segments.extend(segments)
        return all_segments

    def generate_subtitles(self, source, src_file, target, replace=False) -> List[TranscriptionSegment]:
        filename = os.path.splitext(os.path.basename(src_file))[0]
        # if replace is True, delete the existing subtitles
        if replace:
            log.info(f"Replacing subtitles for {filename}")
            if os.path.exists(os.path.join(target, filename + ".en.srt")):
                os.remove(os.path.join(target, filename + ".en.srt"))
        # if replace is False and the subtitles exist, return
        if os.path.exists(os.path.join(source, filename + ".en.srt")):
            log.info(f"Subtitles already exist for {filename}")
            return []

        all_segments = self.get_text_and_segments(source, src_file)
        self.write_srt(all_segments, os.path.join(
            target, os.path.splitext(os.path.basename(src_file))[0] + ".en.srt"))
        return all_segments

    def assemble_audio(self, audio_dict: List[TranscriptionSegment], source, src_file, duration):
        # Create a complex filter command for mixing audio files with delays
        inputs = []
        filter_complex_parts = []
        output_file = os.path.join(self.get_file_working_dir(source, src_file), os.path.splitext(os.path.basename(src_file))[0] + ".en.mp3")
        log.info(f"Assembling audio for {src_file} out of {len(audio_dict)} segments")
        if os.path.exists(output_file):
            log.info(f"Audio already exists for {output_file}")
            return output_file

        for i, segment in enumerate(sorted(audio_dict, key=lambda s: s.start)):
            start_time = segment.start
            input_file = f"{i:03d}.en.mp3"
            input_path = os.path.join(self.get_file_working_dir(source, src_file), input_file)
            inputs.extend(['-i', input_path])
            # Add delay filter for each input
            delay_ms = int(start_time * 1000)
            filter_complex_parts.append(f'[{i}]adelay={delay_ms}|{delay_ms}[delayed{i}]')
        
        # Mix all delayed inputs
        mix_inputs = ''.join(f'[delayed{i}]' for i in range(len(audio_dict)))
        filter_complex_parts.append(f'{mix_inputs}amix=inputs={len(audio_dict)}:normalize=0[mixed]')
        
        # Trim and pad to exact duration
        filter_complex_parts.append(f'[mixed]atrim=end={duration},apad=whole_dur={duration}[final]')
        
        filter_complex = ';'.join(filter_complex_parts)
        
        cmd = [
            'ffmpeg', *inputs, '-filter_complex', filter_complex,
            '-map', '[final]', '-acodec', 'libmp3lame', '-b:a', '320k',
            '-ar', '44100', '-y', output_file
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_file

    def mix_audio_with_video(self, 
        source: str, 
        src_file: str, 
        audio_file: str, 
        output_file: str,
        # https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes
        # for example: "eng", "fra", "spa", "deu"
        languages: List[str]):
        log.info(f"Mixing audio with video for {src_file} {audio_file} out of {len(languages)} languages")
        if os.path.exists(output_file):
            log.info(f"Audio with video already exists for {output_file}")
            return

        audio_streams = self.get_media_audio_streams(src_file)
        if len(audio_streams) != len(languages) -1:
            log.error(f"Number of audio streams ({len(audio_streams)}) does not match number of languages ({len(languages)})")
            raise Exception(f"Number of audio streams ({len(audio_streams)}) does not match number of languages ({len(languages)})")

        args = []
        for i, language in enumerate(languages):
            args.append(f'-metadata:s:a:{i}')
            args.append(f'language={language}')

        cmd = [
            'ffmpeg', '-i', src_file, '-i', audio_file, '-c:v', 'copy', '-c:a', 'copy', '-map', '0', '-map', '1', *args, '-y', output_file
        ]
        subprocess.run(cmd, check=True, capture_output=True)