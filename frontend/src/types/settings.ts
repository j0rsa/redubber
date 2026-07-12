/**
 * Type definitions for the Settings screen.
 */

export type SttModel = 'whisper-1';
export type TtsModel = 'tts-1' | 'tts-1-hd' | 'gpt-4o-mini-tts';
export type VoiceAnalysisModel = 'gpt-4o' | 'gpt-4o-mini' | 'o3' | 'o4-mini';
export type VoiceAnalysisAudioModel = 'gpt-audio-mini' | 'gpt-audio-1';
export type DefaultVoice = 'alloy' | 'echo' | 'fable' | 'onyx' | 'nova' | 'shimmer';

export interface SettingsData {
  /** Masked API key: "sk-...xxxx" when saved, or "" when not yet configured. Never the full key. */
  openai_api_key: string;
  /** Custom OpenAI-compatible API base URL. Empty = use default openai.com endpoint. */
  openai_base_url: string;
  stt_model: SttModel;
  tts_model: TtsModel;
  voice_analysis_model: VoiceAnalysisModel;
  voice_analysis_audio_model: VoiceAnalysisAudioModel;
  default_voice: DefaultVoice;
  /** Starting directory for the file browser when creating a new project. */
  projects_root_path: string;
  /** Filesystem path where project folders are created. */
  working_directory: string;
  /** When true, all redub steps run automatically and the original file is overwritten. */
  auto_process: boolean;
  /** Speech synthesis playback rate. 1.25 helps dubs fit timing; 1.0 = natural pace. */
  tts_speed: number;
  /** Number of parallel TTS requests. Higher = faster but more API load. */
  tts_concurrency: number;
  /** Timeout in seconds per OpenAI API request. */
  openai_timeout: number;
  /** Number of retries on failed OpenAI API requests. */
  openai_retries: number;
  /** Audio chunk duration in seconds for Whisper transcription (max ~25 MB per chunk). */
  audio_chunk_duration: number;
}

export const DEFAULT_SETTINGS: SettingsData = {
  openai_api_key: '',
  openai_base_url: '',
  stt_model: 'whisper-1',
  tts_model: 'tts-1',
  voice_analysis_model: 'gpt-4o',
  voice_analysis_audio_model: 'gpt-audio-1',
  default_voice: 'nova',
  projects_root_path: '',
  working_directory: '',
  auto_process: false,
  tts_speed: 1.25,
  tts_concurrency: 20,
  openai_timeout: 60,
  openai_retries: 3,
  audio_chunk_duration: 900,
};
