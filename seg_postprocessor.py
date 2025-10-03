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

def word_count(text: str) -> int:
    words = [s for s in text.split(' ') if len(s) > 0]
    words = [w for w in words if w not in punctuation]
    words = [w for w in words if w not in ["", " "] and w.isalpha()]
    return len(words)

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
    for seg in segments[1:]:
        seg_text = seg.text.strip()
        new_seg_text = current_segment.text + ' ' + seg_text
        if (not current_segment.text.strip().endswith(punctuation)) and \
        text_within_cps(current_segment.start, seg.end, new_seg_text) and \
        word_count(new_seg_text) <= 20:
            current_segment.text = new_seg_text
            current_segment.end = seg.end
        else:
            result.append(current_segment)
            current_segment = seg
    result.append(current_segment)
    return result
