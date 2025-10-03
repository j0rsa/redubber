import os
from redubber import Redubber
import sys
import logging
from reproj import Reproj
import shutil
from seg_postprocessor import postprocess_segments
from seg_postprocessor import word_count

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
source = "CATEPXAN"
target = "dest"


def get_env_str_or_raise(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise EnvironmentError(f"Environment variable '{key}' is not set.")
    return value


openai_token: str = get_env_str_or_raise("OPENAI_TOKEN")


def redub(file_filter: str | None = None, interactive: bool = False):
    os.makedirs(target, exist_ok=True)

    def file_condition(x: str) -> bool:
        if file_filter:
            return file_filter in x
        return True

    redubber = Redubber(openai_token, interactive=interactive)
    for root, _dirs, files in os.walk(source):
        for file in files:
            src_file = os.path.join(root, file)
            reproj = Reproj(source, src_file)

            if redubber.can_redub(src_file) and file_condition(reproj.filename):
                log.info(f"Redubbing {reproj.filename}")
                all_segments = redubber.get_text_and_segments(reproj)
                sbt_file = redubber.generate_subtitles(reproj, all_segments)
                shutil.copy(sbt_file, os.path.join(target, os.path.basename(sbt_file)))

                redubber.tts_segments(reproj, all_segments)
                redubbed_audio_path = redubber.assemble_long_audio(
                    all_segments, reproj, redubber.get_media_duration(src_file)
                )
                # mix audio with video and save to target
                final_video_path = os.path.join(target, reproj.filename + ".mp4")
                redubber.mix_audio_with_video(
                    reproj, redubbed_audio_path, final_video_path, ["zho", "eng"]
                )
                audio_streams = redubber.get_media_audio_streams(src_file)
                log.info(f"Original audio streams: {audio_streams}")
                audio_streams = redubber.get_media_audio_streams(final_video_path)
                log.info(f"Redubbed audio streams: {audio_streams}")
                # copy subs to target
                # break

def find_src_file(name: str, ext: str = ".mp4"):
    src_file = None
    first_folder = os.path.join(source, [folder for folder in os.listdir(source) if not folder.startswith(".") and os.path.isdir(os.path.join(source, folder))][0])
    for file in os.listdir(first_folder):
        if file.endswith(".mp4") and "62" in file:
            src_file = os.path.join(first_folder, file)
            break
    if src_file is None:
        raise Exception("No src file found")
    return src_file

def compress(interactive: bool = False):
    src_file = find_src_file("62")
    reproj = Reproj(source, src_file)
    redubber = Redubber(openai_token, interactive=interactive)
    all_segments_a = redubber.get_text_and_segments(reproj, compact=False)
    a_len = len(all_segments_a)
    a_word_count = sum([word_count(seg.text) for seg in all_segments_a])
    print(f"Before compacting:  segments: {a_len} words: {a_word_count}")
    redubber.write_srt(
        all_segments_a,
        os.path.join(
            source, os.path.splitext(os.path.basename(src_file))[0] + ".en.srt"
        ),
    )
    all_segments_b = postprocess_segments(all_segments_a)
    b_len = len(all_segments_b)
    b_word_count = sum([word_count(seg.text) for seg in all_segments_b])
    print(f"After compacting:  segments: {b_len} words: {b_word_count}")
    # print(all_segments)
    redubber.write_srt(
        all_segments_b,
        os.path.join(
            source, os.path.splitext(os.path.basename(src_file))[0] + ".compact.en.srt"
        ),
    )
    compression = (a_len - b_len) / a_len * 100
    print("Compression: ", round(compression, 2), "%")

    #compare word count
    print("Word count difference: ", a_word_count - b_word_count)
    


def join(interactive: bool = False):
    src_file = find_src_file("62")
    src_file = os.path.join(source, src_file)
    reproj = Reproj(source, src_file)
    redubber = Redubber(openai_token, interactive=interactive)
    redubber.mix_audio_with_video(
        reproj,
        "dest/62 Stone Stairs Production.en.mp3",
        "dest/62 Stone Stairs Production.mp4",
        ["zho", "eng"],
    )


def main():
    # print(sys.argv)
    if len(sys.argv) == 1:
        print("Running main function")
        redub("62", interactive=True)
        return
    case = sys.argv[1]
    match case:
        case "compress":
            compress()
        case "join":
            join()


if __name__ == "__main__":
    main()
