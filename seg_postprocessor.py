from openai.types.audio.transcription_segment import TranscriptionSegment
from typing import List

def postprocess_segments(segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
    seg = merge_close_segments(segments)
    seg = join_segments_without_punctuation(seg)
    return seg


def join_segments_without_punctuation(
    segments: List[TranscriptionSegment]
) -> List[TranscriptionSegment]:
    if not segments:
        return []
    result = []
    current_segment = segments[0]
    current_added = False
    for seg in segments[1:]:
        if not seg.text.strip().endswith(
            (".", "?", "!", ":", ";", ",", "(", ")", "[", "]", "{", "}", "'", '"')
        ):
            current_segment.text += seg.text.strip()
            current_segment.end = seg.end
            current_added = False
        else:
            result.append(current_segment)
            current_added = True
            current_segment = seg
    if not current_added:
        result.append(current_segment)
    return result

def merge_close_segments(segments, max_gap=0.5):
    merged = []
    buffer = segments[0].copy()
    for seg in segments[1:]:
        gap = seg.start - buffer.end
        # Only merge if same speaker, gap is small, and buffer will not be too long
        if gap <= max_gap:
            buffer.end = seg.end
            buffer.text += ' ' + seg.text
        else:
            merged.append(buffer)
            buffer = seg.copy()
    merged.append(buffer)
    return merged
