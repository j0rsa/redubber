export interface SubtitleReviewOriginal {
  chunk_url: string;
  chunk_name: string;
  seek_start: number;
  seek_end: number;
}

export interface SubtitleReviewFileOption {
  path: string;
  label: string;
  source: string;
}

export interface SubtitleQualityRule {
  id: string;
  label: string;
  scope: 'cue' | 'file';
}

export interface SubtitleQualityBreach {
  rule_id: string;
  message: string;
  segment_index: number | null;
}

/** @deprecated Use SubtitleQualityBreach — kept for backward compatibility */
export interface SubtitleReviewHallucinationWarning {
  code: string;
  message: string;
  segment_index: number | null;
}

export interface SubtitleReviewSegment {
  index: number;
  start: number;
  end: number;
  duration: number;
  text: string;
  original: SubtitleReviewOriginal | null;
  tts_url: string | null;
  breached_rule_count: number;
  breached_rules: string[];
}

export interface SubtitleReviewData {
  video_id: number;
  filename: string;
  srt_path: string;
  available_files: SubtitleReviewFileOption[];
  segments: SubtitleReviewSegment[];
  total: number;
  returned: number;
  has_chunks: boolean;
  has_tts: boolean;
  hallucination_warnings: SubtitleReviewHallucinationWarning[];
  quality_rules: SubtitleQualityRule[];
  quality_breaches: SubtitleQualityBreach[];
}
