import type { Meta, StoryObj } from '@storybook/react-vite';
import { VoiceRefinementView } from './VoiceRefinementView';
import type { VoiceRefinementViewProps } from './VoiceRefinementView';
import type { TranscriptionSegment, VoiceInstructions, VoicePreview, SegmentFilter } from './types';

const meta: Meta<typeof VoiceRefinementView> = {
  title: 'Components/VoiceRefinement',
  component: VoiceRefinementView,
  parameters: {
    layout: 'fullscreen',
    backgrounds: { default: 'dark' },
    docs: {
      description: {
        component:
          'Voice customization modal with two tabs:\n\n' +
          '**Interactive** — 3-step flow: pick a segment → AI-analyze voice → preview all voices.\n\n' +
          '**Manual** — directly type instructions and pick a voice, with an optional preview step.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof VoiceRefinementView>;

// ─── Mock data ────────────────────────────────────────────────────────────────

const seg = (
  id: string, file: string, start: number, end: number,
  original: string, translated: string
): TranscriptionSegment => ({
  id, video_filename: file, start_time: start, end_time: end,
  duration: end - start, original_text: original, translated_text: translated,
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
  text: 'Speak in a professional, clear, and engaging tone. Use a moderate pace with emphasis on key technical terms. Maintain an educational and authoritative style suitable for tutorial content. Express slight enthusiasm when introducing new concepts.',
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
  onUpdateInstructions: (i) => console.log('Instructions updated:', i),
  onSelectVoice: (v) => console.log('Voice selected:', v),
  onGeneratePreviews: async () => console.log('Generate previews clicked'),
  onSave: () => console.log('Save clicked'),
  onClose: () => console.log('Close clicked'),
};

const segmentBase = { filter: defaultFilter, totalCandidates: 5, hasMore: false };

// ─── Interactive tab stories ──────────────────────────────────────────────────

export const Interactive_LoadingSegments: Story = {
  name: 'Interactive — Loading segments',
  args: {
    ...baseActions, ...segmentBase,
    segments: [], selectedSegment: null, loadingSegments: true,
    voiceInstructions: '', voiceInstructionsData: null, analyzingVoice: false,
    previews: [], selectedVoice: null, generatingPreviews: false,
    saving: false, error: null,
  },
};

export const Interactive_SegmentsReady: Story = {
  name: 'Interactive — Segments ready',
  args: {
    ...baseActions, ...segmentBase,
    segments: mockSegments, selectedSegment: null, loadingSegments: false,
    voiceInstructions: '', voiceInstructionsData: null, analyzingVoice: false,
    previews: [], selectedVoice: null, generatingPreviews: false,
    saving: false, error: null,
  },
};

export const Interactive_Analyzing: Story = {
  name: 'Interactive — Analyzing voice',
  args: {
    ...baseActions, ...segmentBase,
    segments: mockSegments, selectedSegment: mockSegments[1], loadingSegments: false,
    voiceInstructions: '', voiceInstructionsData: null, analyzingVoice: true,
    previews: [], selectedVoice: null, generatingPreviews: false,
    saving: false, error: null,
  },
};

export const Interactive_InstructionsReady: Story = {
  name: 'Interactive — Instructions ready',
  args: {
    ...baseActions, ...segmentBase,
    segments: mockSegments, selectedSegment: mockSegments[1], loadingSegments: false,
    voiceInstructions: mockInstructions.text, voiceInstructionsData: mockInstructions,
    analyzingVoice: false,
    previews: [], selectedVoice: null, generatingPreviews: false,
    saving: false, error: null,
  },
};

export const Interactive_PreviewsReady: Story = {
  name: 'Interactive — Previews ready',
  args: {
    ...baseActions, ...segmentBase,
    segments: mockSegments, selectedSegment: mockSegments[1], loadingSegments: false,
    voiceInstructions: mockInstructions.text, voiceInstructionsData: mockInstructions,
    analyzingVoice: false,
    previews: mockPreviews, selectedVoice: 'nova', generatingPreviews: false,
    saving: false, error: null,
  },
};

export const Interactive_EmptyProject: Story = {
  name: 'Interactive — No segments (transcription needed)',
  args: {
    ...baseActions, filter: defaultFilter, totalCandidates: 0, hasMore: false,
    segments: [], selectedSegment: null, loadingSegments: false,
    voiceInstructions: '', voiceInstructionsData: null, analyzingVoice: false,
    previews: [], selectedVoice: null, generatingPreviews: false,
    saving: false, error: null,
    onTranscribe: async () => console.log('Transcribe clicked'),
  },
};

// ─── Manual tab stories ───────────────────────────────────────────────────────
// These open on the Interactive tab by default; the tab switcher is interactive.
// Storybook can't directly control internal useState, but docs show the states.

export const Manual_VoiceAndInstructions: Story = {
  name: 'Manual — Voice + instructions filled',
  args: {
    ...baseActions, ...segmentBase,
    segments: [], selectedSegment: null, loadingSegments: false,
    voiceInstructions: mockInstructions.text,
    voiceInstructionsData: null,
    analyzingVoice: false,
    previews: [], selectedVoice: 'nova', generatingPreviews: false,
    saving: false, error: null,
  },
};

export const Manual_PreviewsGenerated: Story = {
  name: 'Manual — Previews generated',
  args: {
    ...baseActions, ...segmentBase,
    segments: [], selectedSegment: null, loadingSegments: false,
    voiceInstructions: mockInstructions.text,
    voiceInstructionsData: null,
    analyzingVoice: false,
    previews: mockPreviews, selectedVoice: 'shimmer', generatingPreviews: false,
    saving: false, error: null,
  },
};

// ─── Shared states ────────────────────────────────────────────────────────────

export const Saving: Story = {
  name: 'Saving',
  args: {
    ...baseActions, ...segmentBase,
    segments: mockSegments, selectedSegment: mockSegments[0], loadingSegments: false,
    voiceInstructions: mockInstructions.text, voiceInstructionsData: mockInstructions,
    analyzingVoice: false,
    previews: mockPreviews, selectedVoice: 'nova', generatingPreviews: false,
    saving: true, error: null,
  },
};

export const WithError: Story = {
  name: 'Error state',
  args: {
    ...baseActions, ...segmentBase,
    segments: mockSegments, selectedSegment: null, loadingSegments: false,
    voiceInstructions: '', voiceInstructionsData: null, analyzingVoice: false,
    previews: [], selectedVoice: null, generatingPreviews: false,
    saving: false,
    error: 'Failed to connect to the AI service. Please check your API key and try again.',
  },
};
