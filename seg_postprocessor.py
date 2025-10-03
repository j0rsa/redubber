from openai.types.audio.transcription_segment import TranscriptionSegment
from typing import List

def postprocess_segments(segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
    seg = segments
    seg = drop_empty_segments(seg)
    seg = join_segments_without_punctuation(seg)
    return seg

punctuation = (".", "?", "!", ":", ";", ",", "(", ")", "[", "]", "{", "}", "'", '"')

def drop_empty_segments(segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
    return [seg for seg in segments if seg.text.strip()]

def text_within_cps(start: float, end: float, text: str, cps: int = 10) -> bool:
    """
    Check if the text is within the cps limit.

    Characters per minute (CPM): Native speakers can read roughly 300 to 700 characters per minute for leisure reading.
    Resulting in 5 to 11 characters per second (CPS)

    Args:
        start: The start time of the text.
        end: The end time of the text.
        text: The text to check.
        cps: The cps limit.

    Returns:
        True if the text is within the cps limit, False otherwise.
    """
    if text == "":
        return True
    sub_text = text.strip()
    for p in punctuation:
        sub_text = sub_text.replace(p, "")
    sub_text = sub_text.replace(" ", "")
    chars_without_space_and_punctuation = len(sub_text)
    duration = end - start
    return chars_without_space_and_punctuation / duration <= cps

def join_segments_without_punctuation(
    segments: List[TranscriptionSegment]
) -> List[TranscriptionSegment]:
    if not segments:
        return []
    result = []
    current_segment = segments[0]
    current_added = False
    for seg in segments[1:]:
        if not seg.text.strip().endswith(punctuation) and \
        text_within_cps(current_segment.start, seg.end, seg.text.strip() + ' ' + current_segment.text):
            current_segment.text += ' ' + seg.text.strip()
            current_segment.end = seg.end
            current_added = False
        else:
            result.append(current_segment)
            current_added = True
            current_segment = seg
    if not current_added:
        result.append(current_segment)
    return result
