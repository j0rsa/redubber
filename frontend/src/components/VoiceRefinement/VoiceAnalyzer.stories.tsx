import type { Meta, StoryObj } from '@storybook/react-vite';
import { VoiceAnalyzer } from './VoiceAnalyzer';
import type { TranscriptionSegment, VoiceInstructions } from './types';

const meta: Meta<typeof VoiceAnalyzer> = {
  title: 'Components/VoiceRefinement/VoiceAnalyzer',
  component: VoiceAnalyzer,
  parameters: {
    layout: 'padded',
    backgrounds: {
      default: 'light-gray',
    },
    docs: {
      description: {
        component: 'Voice analyzer component for generating and editing voice instructions using AI analysis.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof VoiceAnalyzer>;

// Mock segment data
const mockSegment: TranscriptionSegment = {
  id: 'seg_001',
  video_filename: 'tutorial_intro.mp4',
  start_time: 0.0,
  end_time: 8.5,
  duration: 8.5,
  original_text: 'Welcome to this comprehensive tutorial on advanced programming techniques.',
  translated_text: 'Bienvenido a este tutorial completo sobre técnicas avanzadas de programación.',
  audio_url: 'https://example.com/audio/seg_001.mp3',
};

const mockInstructions = 'Professional, clear, and engaging tone. Moderate pace with emphasis on key technical terms. Maintain an educational and authoritative style suitable for tutorial content. Slight enthusiasm when introducing new concepts.';

const mockInstructionsData: VoiceInstructions = {
  text: mockInstructions,
  detected_characteristics: {
    tone: 'Professional and educational',
    pace: 'Moderate, clear articulation',
    emotion: 'Engaged and enthusiastic',
    style: 'Tutorial presenter',
  },
  generation_id: 1,
};

const longInstructions = mockInstructions.repeat(3); // Over 500 characters

// Default: No segment selected (disabled state)
export const Default: Story = {
  args: {
    selectedSegment: null,
    voiceInstructions: '',
    voiceInstructionsData: null,
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
};

// Segment selected, ready to analyze
export const ReadyToAnalyze: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: '',
    voiceInstructionsData: null,
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
};

// Analyzing (loading state)
export const Analyzing: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: '',
    voiceInstructionsData: null,
    isAnalyzing: true,
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
};

// Instructions generated (with characteristics)
export const InstructionsGenerated: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: mockInstructions,
    voiceInstructionsData: mockInstructionsData,
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
};

// Instructions without characteristics data
export const InstructionsWithoutCharacteristics: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: mockInstructions,
    voiceInstructionsData: null,
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
};

// Editing mode
export const EditingInstructions: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: mockInstructions,
    voiceInstructionsData: mockInstructionsData,
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
  play: async ({ canvasElement }) => {
    // Simulate clicking edit button
    const editButton = canvasElement.querySelector('button[class*="secondaryButton"]');
    if (editButton) {
      (editButton as HTMLElement).click();
    }
  },
};

// Long instructions (over 500 characters - shows warning)
export const LongInstructions: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: longInstructions,
    voiceInstructionsData: {
      ...mockInstructionsData,
      text: longInstructions,
    },
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
};

// Regenerating with feedback (loading)
export const RegeneratingWithFeedback: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: mockInstructions,
    voiceInstructionsData: mockInstructionsData,
    isAnalyzing: true,
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
};

// Short instructions
export const ShortInstructions: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: 'Clear, professional tone with moderate pace.',
    voiceInstructionsData: {
      text: 'Clear, professional tone with moderate pace.',
      detected_characteristics: {
        tone: 'Professional',
        pace: 'Moderate',
        emotion: 'Neutral',
        style: 'Clear',
      },
      generation_id: 1,
    },
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
};

// Many characteristics
export const ManyCharacteristics: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: mockInstructions,
    voiceInstructionsData: {
      text: mockInstructions,
      detected_characteristics: {
        tone: 'Professional and educational',
        pace: 'Moderate, clear articulation',
        emotion: 'Engaged and enthusiastic',
        style: 'Tutorial presenter',
      },
      generation_id: 1,
    },
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
};

// Few characteristics
export const FewCharacteristics: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: mockInstructions,
    voiceInstructionsData: {
      text: mockInstructions,
      detected_characteristics: {
        tone: 'Professional',
        pace: 'Moderate',
        emotion: 'Neutral',
        style: 'Informative',
      },
      generation_id: 1,
    },
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
};

// Different segment characteristics - Energetic
export const EnergeticVoice: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: 'High energy, upbeat delivery with fast pace and enthusiastic tone. Perfect for motivational content and exciting announcements.',
    voiceInstructionsData: {
      text: 'High energy, upbeat delivery with fast pace and enthusiastic tone. Perfect for motivational content and exciting announcements.',
      detected_characteristics: {
        tone: 'Enthusiastic and upbeat',
        pace: 'Fast, dynamic',
        emotion: 'Excited and motivated',
        style: 'Energetic presenter',
      },
      generation_id: 2,
    },
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
};

// Different segment characteristics - Calm
export const CalmVoice: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: 'Slow, soothing pace with gentle, calming tone. Ideal for meditation guides and relaxation content.',
    voiceInstructionsData: {
      text: 'Slow, soothing pace with gentle, calming tone. Ideal for meditation guides and relaxation content.',
      detected_characteristics: {
        tone: 'Gentle and soothing',
        pace: 'Slow, deliberate',
        emotion: 'Calm and peaceful',
        style: 'Meditation guide',
      },
      generation_id: 3,
    },
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
};

// Mobile viewport
export const Mobile: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: mockInstructions,
    voiceInstructionsData: mockInstructionsData,
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
  parameters: {
    viewport: {
      defaultViewport: 'mobile1',
    },
  },
};

// Dark background
export const DarkBackground: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: mockInstructions,
    voiceInstructionsData: mockInstructionsData,
    onAnalyze: async () => {
      console.log('Analyze clicked');
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    onUpdateInstructions: (instructions) => console.log('Update instructions:', instructions),
  },
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
};
