import type { Meta, StoryObj } from '@storybook/react-vite';
import { VoiceRefinementView } from './VoiceRefinementView';
import type { VoiceRefinementViewProps } from './VoiceRefinementView';
import type { TranscriptionSegment, VoiceInstructions, VoicePreview, SegmentFilter } from './types';

const meta: Meta<typeof VoiceRefinementView> = {
  title: 'Components/VoiceRefinement',
  component: VoiceRefinementView,
  parameters: {
    // fullscreen so the fixed-position modal overlay fills the canvas naturally
    layout: 'fullscreen',
    backgrounds: { default: 'dark' },
    docs: {
      description: {
        component:
          'Voice refinement modal: select a transcription segment, analyze voice characteristics with AI, generate TTS previews for all voices, and save the best match.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof VoiceRefinementView>;

// ─── Mock data ───────────────────────────────────────────────────────────────

const seg = (
  id: string,
  file: string,
  start: number,
  end: number,
  original: string,
  translated: string
): TranscriptionSegment => ({
  id,
  video_filename: file,
  start_time: start,
  end_time: end,
  duration: end - start,
  original_text: original,
  translated_text: translated,
  audio_url: `https://example.com/audio/${id}.mp3`,
});

const mockSegments: TranscriptionSegment[] = [
  seg('seg_001', 'tutorial_intro.mp4', 0, 8.5,
    'Welcome to this comprehensive tutorial on advanced programming techniques.',
    'Bienvenido a este tutorial completo sobre técnicas avanzadas de programación.'),
  seg('seg_002', 'tutorial_intro.mp4', 8.5, 15.2,
    "Today, we'll explore the fundamental concepts every developer should know.",
    'Hoy exploraremos los conceptos fundamentales que todo desarrollador debe conocer.'),
  seg('seg_003', 'tutorial_intro.mp4', 15.2, 22.8,
    "Let's begin by understanding the core principles of clean code architecture.",
    'Comencemos por entender los principios básicos de la arquitectura de código limpio.'),
  seg('seg_004', 'tutorial_demo.mp4', 0, 12.3,
    "I'll show you how to implement these patterns in a real-world application.",
    'Te mostraré cómo implementar estos patrones en una aplicación del mundo real.'),
  seg('seg_005', 'tutorial_demo.mp4', 12.3, 20.1,
    'Pay close attention to the error handling strategy we use here.',
    'Presta mucha atención a la estrategia de manejo de errores que usamos aquí.'),
];

const mockInstructions: VoiceInstructions = {
  text: 'Speak in a professional, clear, and engaging tone. Use a moderate pace with emphasis on key technical terms. Maintain an educational and authoritative style suitable for tutorial content. Express slight enthusiasm when introducing new concepts, and pause naturally between sections.',
  detected_characteristics: {
    tone: 'Professional and educational',
    pace: 'Moderate, clear articulation',
    style: 'Tutorial presenter',
  },
  generation_id: 1,
};

const mockPreviews: VoicePreview[] = [
  { voice: 'alloy',   audio_url: 'https://example.com/preview/alloy.mp3',   duration_ms: 8500, cached: true  },
  { voice: 'echo',    audio_url: 'https://example.com/preview/echo.mp3',    duration_ms: 8500, cached: true  },
  { voice: 'fable',   audio_url: 'https://example.com/preview/fable.mp3',   duration_ms: 8700, cached: false },
  { voice: 'onyx',    audio_url: 'https://example.com/preview/onyx.mp3',    duration_ms: 8300, cached: false },
  { voice: 'nova',    audio_url: 'https://example.com/preview/nova.mp3',    duration_ms: 8600, cached: false },
  { voice: 'shimmer', audio_url: 'https://example.com/preview/shimmer.mp3', duration_ms: 8400, cached: false },
];

const defaultFilter: SegmentFilter = { search: '', minDuration: 0, maxDuration: 0 };

// ─── Shared no-op actions ─────────────────────────────────────────────────────

const baseActions: Pick<
  VoiceRefinementViewProps,
  | 'onSelectSegment' | 'onFilterChange' | 'onLoadMore'
  | 'onAnalyze' | 'onRegenerate' | 'onUpdateInstructions'
  | 'onSelectVoice' | 'onGeneratePreviews' | 'onSave' | 'onClose'
> = {
  onSelectSegment: (s) => console.log('Segment selected:', s.id),
  onFilterChange: (f) => console.log('Filter changed:', f),
  onLoadMore: () => console.log('Load more clicked'),
  onAnalyze: async () => console.log('Analyze clicked'),
  onRegenerate: async (fb) => console.log('Regenerate:', fb),
  onUpdateInstructions: () => console.log('Instructions updated'),
  onSelectVoice: (v) => console.log('Voice selected:', v),
  onGeneratePreviews: async () => console.log('Generate previews clicked'),
  onSave: () => console.log('Save clicked'),
  onClose: () => console.log('Close clicked'),
};

// ─── Shared segment-step props ────────────────────────────────────────────────

const segmentBase = {
  filter: defaultFilter,
  totalCandidates: 5,
  hasMore: false,
};

const segmentLarge = {
  filter: defaultFilter,
  totalCandidates: 1247,
  hasMore: true,
};

// ─── Stories ─────────────────────────────────────────────────────────────────

export const Step1LoadingSegments: Story = {
  args: {
    ...baseActions,
    ...segmentBase,
    segments: [],
    selectedSegment: null,
    loadingSegments: true,
    voiceInstructions: '',
    voiceInstructionsData: null,
    analyzingVoice: false,
    previews: [],
    selectedVoice: null,
    generatingPreviews: false,
    saving: false,
    error: null,
  },
};

export const Step1SegmentsLoaded: Story = {
  args: {
    ...baseActions,
    ...segmentBase,
    segments: mockSegments,
    selectedSegment: null,
    loadingSegments: false,
    voiceInstructions: '',
    voiceInstructionsData: null,
    analyzingVoice: false,
    previews: [],
    selectedVoice: null,
    generatingPreviews: false,
    saving: false,
    error: null,
  },
};

export const Step1LargeProject: Story = {
  args: {
    ...baseActions,
    ...segmentLarge,
    segments: mockSegments,
    selectedSegment: null,
    loadingSegments: false,
    voiceInstructions: '',
    voiceInstructionsData: null,
    analyzingVoice: false,
    previews: [],
    selectedVoice: null,
    generatingPreviews: false,
    saving: false,
    error: null,
  },
};

export const Step2SegmentSelected: Story = {
  args: {
    ...baseActions,
    ...segmentBase,
    segments: mockSegments,
    selectedSegment: mockSegments[0],
    loadingSegments: false,
    voiceInstructions: '',
    voiceInstructionsData: null,
    analyzingVoice: false,
    previews: [],
    selectedVoice: null,
    generatingPreviews: false,
    saving: false,
    error: null,
  },
};

export const Step2Analyzing: Story = {
  args: {
    ...baseActions,
    ...segmentBase,
    segments: mockSegments,
    selectedSegment: mockSegments[0],
    loadingSegments: false,
    voiceInstructions: '',
    voiceInstructionsData: null,
    analyzingVoice: true,
    previews: [],
    selectedVoice: null,
    generatingPreviews: false,
    saving: false,
    error: null,
  },
};

export const Step2InstructionsReady: Story = {
  args: {
    ...baseActions,
    ...segmentBase,
    segments: mockSegments,
    selectedSegment: mockSegments[0],
    loadingSegments: false,
    voiceInstructions: mockInstructions.text,
    voiceInstructionsData: mockInstructions,
    analyzingVoice: false,
    previews: [],
    selectedVoice: null,
    generatingPreviews: false,
    saving: false,
    error: null,
  },
};

export const Step3GeneratingPreviews: Story = {
  args: {
    ...baseActions,
    ...segmentBase,
    segments: mockSegments,
    selectedSegment: mockSegments[0],
    loadingSegments: false,
    voiceInstructions: mockInstructions.text,
    voiceInstructionsData: mockInstructions,
    analyzingVoice: false,
    previews: [],
    selectedVoice: null,
    generatingPreviews: true,
    saving: false,
    error: null,
  },
};

export const Step3PreviewsReady: Story = {
  args: {
    ...baseActions,
    ...segmentBase,
    segments: mockSegments,
    selectedSegment: mockSegments[0],
    loadingSegments: false,
    voiceInstructions: mockInstructions.text,
    voiceInstructionsData: mockInstructions,
    analyzingVoice: false,
    previews: mockPreviews,
    selectedVoice: null,
    generatingPreviews: false,
    saving: false,
    error: null,
  },
};

export const Step4VoiceSelected: Story = {
  args: {
    ...baseActions,
    ...segmentBase,
    segments: mockSegments,
    selectedSegment: mockSegments[0],
    loadingSegments: false,
    voiceInstructions: mockInstructions.text,
    voiceInstructionsData: mockInstructions,
    analyzingVoice: false,
    previews: mockPreviews,
    selectedVoice: 'nova',
    generatingPreviews: false,
    saving: false,
    error: null,
  },
};

export const Saving: Story = {
  args: {
    ...baseActions,
    ...segmentBase,
    segments: mockSegments,
    selectedSegment: mockSegments[0],
    loadingSegments: false,
    voiceInstructions: mockInstructions.text,
    voiceInstructionsData: mockInstructions,
    analyzingVoice: false,
    previews: mockPreviews,
    selectedVoice: 'nova',
    generatingPreviews: false,
    saving: true,
    error: null,
  },
};

export const WithError: Story = {
  args: {
    ...baseActions,
    ...segmentBase,
    segments: mockSegments,
    selectedSegment: mockSegments[0],
    loadingSegments: false,
    voiceInstructions: '',
    voiceInstructionsData: null,
    analyzingVoice: false,
    previews: [],
    selectedVoice: null,
    generatingPreviews: false,
    saving: false,
    error: 'Failed to connect to the AI service. Please check your API key and try again.',
  },
};

export const EmptyProject: Story = {
  args: {
    ...baseActions,
    filter: defaultFilter,
    totalCandidates: 0,
    hasMore: false,
    segments: [],
    selectedSegment: null,
    loadingSegments: false,
    voiceInstructions: '',
    voiceInstructionsData: null,
    analyzingVoice: false,
    previews: [],
    selectedVoice: null,
    generatingPreviews: false,
    saving: false,
    error: null,
  },
};
