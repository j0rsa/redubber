import type { Meta, StoryObj } from '@storybook/react-vite';
import { VoicePreviewGrid } from './VoicePreviewGrid';
import type { VoicePreview } from './types';

const meta: Meta<typeof VoicePreviewGrid> = {
  title: 'Components/VoiceRefinement/VoicePreviewGrid',
  component: VoicePreviewGrid,
  parameters: {
    layout: 'padded',
    backgrounds: {
      default: 'light-gray',
    },
    docs: {
      description: {
        component: 'Grid of voice preview cards with audio playback for comparing different voice options.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof VoicePreviewGrid>;

// Mock preview data
const allPreviewsMock: VoicePreview[] = [
  {
    voice: 'alloy',
    audio_url: 'https://example.com/preview/alloy.mp3',
    duration_ms: 8500,
    cached: true,
  },
  {
    voice: 'echo',
    audio_url: 'https://example.com/preview/echo.mp3',
    duration_ms: 8500,
    cached: true,
  },
  {
    voice: 'fable',
    audio_url: 'https://example.com/preview/fable.mp3',
    duration_ms: 8500,
    cached: false,
  },
  {
    voice: 'onyx',
    audio_url: 'https://example.com/preview/onyx.mp3',
    duration_ms: 8500,
    cached: false,
  },
  {
    voice: 'nova',
    audio_url: 'https://example.com/preview/nova.mp3',
    duration_ms: 8500,
    cached: false,
  },
  {
    voice: 'shimmer',
    audio_url: 'https://example.com/preview/shimmer.mp3',
    duration_ms: 8500,
    cached: false,
  },
];

// Default: All voices available, none selected
export const Default: Story = {
  args: {
    previews: allPreviewsMock,
    selectedVoice: null,
    onSelectVoice: (voice) => console.log('Voice selected:', voice),
    onGeneratePreviews: async () => {
      console.log('Generate previews clicked');
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    isGenerating: false,
    hasInstructions: true,
  },
};

// One voice selected
export const VoiceSelected: Story = {
  args: {
    previews: allPreviewsMock,
    selectedVoice: 'nova',
    onSelectVoice: (voice) => console.log('Voice selected:', voice),
    onGeneratePreviews: async () => console.log('Generate previews'),
    hasInstructions: true,
  },
};

// Some previews cached (faster loading)
export const SomeCached: Story = {
  args: {
    previews: [
      { voice: 'alloy', audio_url: 'https://example.com/preview/alloy.mp3', duration_ms: 8500, cached: true },
      { voice: 'echo', audio_url: 'https://example.com/preview/echo.mp3', duration_ms: 8500, cached: true },
      { voice: 'fable', audio_url: 'https://example.com/preview/fable.mp3', duration_ms: 8500, cached: true },
      { voice: 'onyx', audio_url: 'https://example.com/preview/onyx.mp3', duration_ms: 8500, cached: false },
      { voice: 'nova', audio_url: 'https://example.com/preview/nova.mp3', duration_ms: 8500, cached: false },
      { voice: 'shimmer', audio_url: 'https://example.com/preview/shimmer.mp3', duration_ms: 8500, cached: false },
    ],
    selectedVoice: null,
    onSelectVoice: (voice) => console.log('Voice selected:', voice),
    onGeneratePreviews: async () => console.log('Generate previews'),
    hasInstructions: true,
  },
};

// All cached
export const AllCached: Story = {
  args: {
    previews: allPreviewsMock.map(p => ({ ...p, cached: true })),
    selectedVoice: null,
    onSelectVoice: (voice) => console.log('Voice selected:', voice),
    onGeneratePreviews: async () => console.log('Generate previews'),
    hasInstructions: true,
  },
};

// Loading state: Generating previews
export const LoadingPreviews: Story = {
  args: {
    previews: [],
    selectedVoice: null,
    onSelectVoice: (voice) => console.log('Voice selected:', voice),
    onGeneratePreviews: async () => console.log('Generate previews'),
    isGenerating: true,
    hasInstructions: true,
  },
};

// Partial loading: Some previews generated
export const PartiallyLoaded: Story = {
  args: {
    previews: allPreviewsMock.slice(0, 3),
    selectedVoice: null,
    onSelectVoice: (voice) => console.log('Voice selected:', voice),
    onGeneratePreviews: async () => console.log('Generate previews'),
    isGenerating: true,
    hasInstructions: true,
  },
};

// No instructions yet (disabled state)
export const NoInstructions: Story = {
  args: {
    previews: [],
    selectedVoice: null,
    onSelectVoice: (voice) => console.log('Voice selected:', voice),
    onGeneratePreviews: async () => console.log('Generate previews'),
    hasInstructions: false,
  },
};

// Empty state: No previews generated yet (but has instructions)
export const NoPreviews: Story = {
  args: {
    previews: [],
    selectedVoice: null,
    onSelectVoice: (voice) => console.log('Voice selected:', voice),
    onGeneratePreviews: async () => console.log('Generate previews'),
    hasInstructions: true,
  },
};

// Subset of voices (only 3)
export const ThreeVoices: Story = {
  args: {
    previews: allPreviewsMock.slice(0, 3),
    selectedVoice: null,
    onSelectVoice: (voice) => console.log('Voice selected:', voice),
    onGeneratePreviews: async () => console.log('Generate previews'),
    hasInstructions: true,
  },
};

// Various durations
export const VariousDurations: Story = {
  args: {
    previews: [
      { voice: 'alloy', audio_url: 'https://example.com/preview/alloy.mp3', duration_ms: 5000, cached: true },
      { voice: 'echo', audio_url: 'https://example.com/preview/echo.mp3', duration_ms: 8500, cached: true },
      { voice: 'fable', audio_url: 'https://example.com/preview/fable.mp3', duration_ms: 12000, cached: false },
      { voice: 'onyx', audio_url: 'https://example.com/preview/onyx.mp3', duration_ms: 15500, cached: false },
      { voice: 'nova', audio_url: 'https://example.com/preview/nova.mp3', duration_ms: 7200, cached: false },
      { voice: 'shimmer', audio_url: 'https://example.com/preview/shimmer.mp3', duration_ms: 9800, cached: false },
    ],
    selectedVoice: null,
    onSelectVoice: (voice) => console.log('Voice selected:', voice),
    onGeneratePreviews: async () => console.log('Generate previews'),
    hasInstructions: true,
  },
};

// Dark background
export const DarkBackground: Story = {
  args: {
    previews: allPreviewsMock,
    selectedVoice: 'fable',
    onSelectVoice: (voice) => console.log('Voice selected:', voice),
    onGeneratePreviews: async () => console.log('Generate previews'),
    hasInstructions: true,
  },
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
};
