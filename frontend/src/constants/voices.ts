/**
 * Voice configuration constants for OpenAI TTS
 */

export interface VoiceMetadata {
  id: string;
  name: string;
  description: string;
  icon: string;
  gender: 'male' | 'female' | 'neutral';
  characteristics?: string[];
}

/**
 * Available OpenAI TTS voices with metadata
 * Based on OpenAI API documentation
 */
export const AVAILABLE_VOICES: VoiceMetadata[] = [
  {
    id: 'alloy',
    name: 'Alloy',
    description: 'Neutral and balanced voice',
    icon: '🎭',
    gender: 'neutral',
    characteristics: ['versatile', 'clear', 'professional'],
  },
  {
    id: 'echo',
    name: 'Echo',
    description: 'Male voice with warm tone',
    icon: '🎤',
    gender: 'male',
    characteristics: ['warm', 'friendly', 'approachable'],
  },
  {
    id: 'fable',
    name: 'Fable',
    description: 'British male voice',
    icon: '📚',
    gender: 'male',
    characteristics: ['expressive', 'storytelling', 'British accent'],
  },
  {
    id: 'onyx',
    name: 'Onyx',
    description: 'Deep male voice',
    icon: '🎙️',
    gender: 'male',
    characteristics: ['deep', 'authoritative', 'confident'],
  },
  {
    id: 'nova',
    name: 'Nova',
    description: 'Energetic female voice',
    icon: '✨',
    gender: 'female',
    characteristics: ['energetic', 'youthful', 'bright'],
  },
  {
    id: 'shimmer',
    name: 'Shimmer',
    description: 'Soft female voice',
    icon: '🌟',
    gender: 'female',
    characteristics: ['soft', 'gentle', 'soothing'],
  },
];

/**
 * Default voice when none is specified
 */
export const DEFAULT_VOICE = 'alloy';

/**
 * Get voice metadata by voice ID
 */
export function getVoiceByName(voiceId: string): VoiceMetadata | undefined {
  return AVAILABLE_VOICES.find((v) => v.id === voiceId);
}

/**
 * Get voice icon emoji by voice ID
 */
export function getVoiceIcon(voiceId: string): string {
  const voice = getVoiceByName(voiceId);
  return voice?.icon || '🎵';
}

/**
 * Get voice display name by voice ID
 */
export function getVoiceDisplayName(voiceId: string): string {
  const voice = getVoiceByName(voiceId);
  return voice?.name || voiceId;
}
