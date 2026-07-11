/**
 * Type definitions for Redubber v2.0
 */

export interface Project {
  id: number;
  path: string;
  name: string;
  created_at: string;
  updated_at: string;
  voice: string;
  voice_instructions: string;
  source_language_override: string;
  /** ISO 639-3 language code for the dubbing target language (e.g. "eng", "spa"). */
  target_language: string;
  working_directory: string;
}

export interface ProjectCreate {
  path: string;
}

export interface AudioStream {
  index: number;
  language: string;
  codec: string;
  channels: number | string;
  sample_rate: number | string;
}

export interface SubtitleInfo {
  language: string;
  embedded: boolean;
  path?: string;
  filename?: string;
}

export interface PipelineStatus {
  progress: number; // 0-100
  current_stage: string;
  is_complete: boolean;
  failed?: boolean;
  error?: string;
  replaced?: boolean;

  // Redubbing pipeline counters (7 steps)
  audio_chunks?: number;           // Stage 1: Extracted chunks
  transcripts?: number;            // Stage 2: Transcribed segments
  translated?: number;             // Stage 3: Translated segments
  tts_segments?: number;           // Stage 4: TTS files (current)
  tts_total?: number;              // Stage 4: TTS files (total)
  subtitles?: number;              // Stage 5: Subtitle files
  audio_assembled?: number;        // Stage 6: Audio chunks mixed (current)
  audio_assembled_total?: number;  // Stage 6: Total chunks to mix
  video_mixed?: boolean;           // Stage 7: Final video created

  // Validation & finalization (Boolean checks)
  output_validated?: boolean;      // Stage 8: Validation passed
  backup_created?: boolean;        // Stage 8: Backup created
  backup_location?: string;        // Stage 8: Backup path
  output_location?: string;        // Stage 8: New video path

  // Manual replacement
  file_replaced?: boolean;         // Stage 9: User decision
  replacement_status?: 'pending' | 'replaced' | 'kept_both' | 'cancelled';

  // Legacy
  has_external_subs?: boolean;
}

export interface VideoFile {
  id: number;
  filename: string;
  path: string;
  size_mb: number;
  duration_seconds: number;
  audio_streams: AudioStream[];
  subtitles: SubtitleInfo[];
  pipeline_status?: PipelineStatus;
}

export interface TaskStatus {
  task_id: string;
  video_path: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage: string;
  progress: number; // 0-100
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  // Pipeline stage counters
  audio_chunks?: number;
  transcripts?: number;
  tts_segments?: number;
  tts_total?: number;
  subtitles?: number;
  audio_assembled?: number;
  audio_assembled_total?: number;
  video_mixed?: boolean;
}

export interface RedubRequest {
  video_path: string;
  project_id: number;
}
