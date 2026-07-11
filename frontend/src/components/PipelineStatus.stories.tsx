import type { Meta, StoryObj } from '@storybook/react-vite';
import { PipelineStatus } from './PipelineStatus';
import type { PipelineStatus as PipelineStatusType } from '../types';

const meta: Meta<typeof PipelineStatus> = {
  title: 'Components/PipelineStatus',
  component: PipelineStatus,
  parameters: {
    docs: {
      description: {
        component: 'Real-time pipeline progress indicator with 7-stage redubbing workflow and all counters.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof PipelineStatus>;

// Stage 1: Extract Audio
export const Stage1_ExtractingAudio: Story = {
  args: {
    status: {
      progress: 15,
      current_stage: 'Extracting audio',
      is_complete: false,
      audio_chunks: 8,
    } as PipelineStatusType,
  },
};

// Stage 2: Transcribe
export const Stage2_Transcribing: Story = {
  args: {
    status: {
      progress: 30,
      current_stage: 'Transcribing',
      is_complete: false,
      audio_chunks: 8,
      transcripts: 120,
    } as PipelineStatusType,
  },
};

// Stage 3: Translate
export const Stage3_Translating: Story = {
  args: {
    status: {
      progress: 40,
      current_stage: 'Translating',
      is_complete: false,
      audio_chunks: 8,
      transcripts: 120,
      translated: 120,
    } as PipelineStatusType,
  },
};

// Stage 4: TTS Generation (Early)
export const Stage4_GeneratingTTS_Early: Story = {
  args: {
    status: {
      progress: 45,
      current_stage: 'Generating TTS',
      is_complete: false,
      audio_chunks: 8,
      transcripts: 120,
      translated: 120,
      tts_segments: 24,
      tts_total: 120,
    } as PipelineStatusType,
  },
};

// Stage 4: TTS Generation (Mid)
export const Stage4_GeneratingTTS_Mid: Story = {
  args: {
    status: {
      progress: 55,
      current_stage: 'Generating TTS',
      is_complete: false,
      audio_chunks: 8,
      transcripts: 120,
      translated: 120,
      tts_segments: 60,
      tts_total: 120,
    } as PipelineStatusType,
  },
};

// Stage 4: TTS Generation (Late)
export const Stage4_GeneratingTTS_Late: Story = {
  args: {
    status: {
      progress: 68,
      current_stage: 'Generating TTS',
      is_complete: false,
      audio_chunks: 8,
      transcripts: 120,
      translated: 120,
      tts_segments: 108,
      tts_total: 120,
    } as PipelineStatusType,
  },
};

// Stage 5: Generate Subtitles
export const Stage5_GeneratingSubtitles: Story = {
  args: {
    status: {
      progress: 72,
      current_stage: 'Generating subtitles',
      is_complete: false,
      audio_chunks: 8,
      transcripts: 120,
      translated: 120,
      tts_segments: 120,
      tts_total: 120,
      subtitles: 1,
    } as PipelineStatusType,
  },
};

// Stage 6: Assemble Audio (In Progress)
export const Stage6_AssemblingAudio_InProgress: Story = {
  args: {
    status: {
      progress: 78,
      current_stage: 'Assembling audio',
      is_complete: false,
      audio_chunks: 8,
      transcripts: 120,
      translated: 120,
      tts_segments: 120,
      tts_total: 120,
      subtitles: 1,
      audio_assembled: 8,
      audio_assembled_total: 12,
    } as PipelineStatusType,
  },
};

// Stage 6: Assemble Audio (Complete)
export const Stage6_AssemblingAudio_Complete: Story = {
  args: {
    status: {
      progress: 80,
      current_stage: 'Assembling audio',
      is_complete: false,
      audio_chunks: 8,
      transcripts: 120,
      translated: 120,
      tts_segments: 120,
      tts_total: 120,
      subtitles: 1,
      audio_assembled: 12,
      audio_assembled_total: 12,
    } as PipelineStatusType,
  },
};

// Stage 7: Mix Video
export const Stage7_MixingVideo: Story = {
  args: {
    status: {
      progress: 85,
      current_stage: 'Mixing video',
      is_complete: false,
      audio_chunks: 8,
      transcripts: 120,
      translated: 120,
      tts_segments: 120,
      tts_total: 120,
      subtitles: 1,
      audio_assembled: 12,
      audio_assembled_total: 12,
      video_mixed: true,
    } as PipelineStatusType,
  },
};

// Stage 8: Validation & Backup
export const Stage8_ValidatingAndBackup: Story = {
  args: {
    status: {
      progress: 95,
      current_stage: 'Ready for replacement',
      is_complete: false,
      audio_chunks: 8,
      transcripts: 120,
      translated: 120,
      tts_segments: 120,
      tts_total: 120,
      subtitles: 1,
      audio_assembled: 12,
      audio_assembled_total: 12,
      video_mixed: true,
      output_validated: true,
      backup_created: true,
      replacement_status: 'pending',
    } as PipelineStatusType,
  },
};

// Stage 9: Complete
export const Stage9_Complete: Story = {
  args: {
    status: {
      progress: 100,
      current_stage: 'Complete',
      is_complete: true,
      audio_chunks: 8,
      transcripts: 120,
      translated: 120,
      tts_segments: 120,
      tts_total: 120,
      subtitles: 1,
      audio_assembled: 12,
      audio_assembled_total: 12,
      video_mixed: true,
      output_validated: true,
      backup_created: true,
      file_replaced: true,
      replacement_status: 'replaced',
    } as PipelineStatusType,
  },
};
