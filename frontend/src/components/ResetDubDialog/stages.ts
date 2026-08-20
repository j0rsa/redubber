export const RESET_STAGES = [
  { id: 'start', label: 'Start' },
  { id: 'audio', label: 'Extract' },
  { id: 'stt', label: 'Transcribe' },
  { id: 'subtitles', label: 'Subtitles' },
  { id: 'tts', label: 'TTS' },
  { id: 'assemble', label: 'Assemble' },
  { id: 'mix', label: 'Mix' },
  { id: 'complete', label: 'Complete' },
] as const;

export type ResetStageId = (typeof RESET_STAGES)[number]['id'];
export type ResetToStageId = Exclude<ResetStageId, 'complete'>;

export function stageIndex(id: ResetStageId): number {
  return RESET_STAGES.findIndex((stage) => stage.id === id);
}
