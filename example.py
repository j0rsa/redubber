import os
from redubber import Redubber
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    source = "CATEPXAN"
    target = "dest"
    os.makedirs(target, exist_ok=True)

    redubber = Redubber()
    for root, _dirs, files in os.walk(source):
        for file in files:
            src_file = os.path.join(root, file)
            filename = os.path.splitext(os.path.basename(src_file))[0]
            
            if redubber.can_redub(src_file) and '6' in filename:
                log.info(f"Redubbing {filename}")
                redubber.generate_subtitles(source, src_file, target)
                
                all_segments = redubber.get_text_and_segments(source, src_file)
                redubber.tts_segments(all_segments, source, src_file)
                redubbed_audio_path = redubber.assemble_audio(
                    all_segments, 
                    source, 
                    src_file, 
                    redubber.get_media_duration(src_file)
                )
                # mix audio with video and save to target
                final_video_path = os.path.join(target, filename + ".en.mp4")
                redubber.mix_audio_with_video(
                    source,
                    src_file,
                    redubbed_audio_path,
                    final_video_path,
                    ["zho", "eng"]
                )
                audio_streams = redubber.get_media_audio_streams(src_file)
                log.info(f"Original audio streams: {audio_streams}")
                audio_streams = redubber.get_media_audio_streams(final_video_path)
                log.info(f"Redubbed audio streams: {audio_streams}")
                # copy subs to target
                # break

if __name__ == "__main__":
    main()