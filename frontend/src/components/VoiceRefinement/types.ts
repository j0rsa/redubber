/**
 * Type definitions for Voice Refinement system
 */

export interface TranscriptionSegment {
  id: string;
  video_filename: string;
  start_time: number;
  end_time: number;
  duration: number;
  original_text: string;
  translated_text: string;
  audio_url: string;
}

export interface VoiceInstructions {
  text: string;
  detected_characteristics: {
    tone: string;
    pace: string;
    energy?: string;
    style: string;
    speaker_gender?: 'male' | 'female' | 'unknown';
  };
  generation_id: number;
}

export interface VoicePreview {
  voice: string;
  audio_url: string;
  duration_ms: number;
  cached: boolean;
}

export interface VoiceSettings {
  voice: string;
  voice_instructions: string;
  segment_used: string;
}

export interface SegmentFilter {
  search: string;
  minDuration: number; // seconds
  maxDuration: number; // seconds
}

export interface SegmentsResponse {
  segments: TranscriptionSegment[];
  total_candidates: number;
  total_matched: number;
  returned: number;
  has_more: boolean;
  sample_size: number;
}

export { AVAILABLE_VOICES } from '../../constants/voices';
export type VoiceId = 'alloy' | 'echo' | 'fable' | 'onyx' | 'nova' | 'shimmer';
