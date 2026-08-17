export interface SubtitleReviewOriginal {
  chunk_url: string;
  chunk_name: string;
  seek_start: number;
  seek_end: number;
}

export interface SubtitleReviewSegment {
  index: number;
  start: number;
  end: number;
  duration: number;
  text: string;
  original: SubtitleReviewOriginal | null;
  tts_url: string | null;
}

export interface SubtitleReviewData {
  video_id: number;
  filename: string;
  srt_path: string;
  segments: SubtitleReviewSegment[];
  total: number;
  returned: number;
  has_chunks: boolean;
  has_tts: boolean;
}
