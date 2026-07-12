import type { Meta, StoryObj } from '@storybook/react-vite';
import { FileGrid } from './FileGrid';
import type { VideoFile } from '../types';

const meta: Meta<typeof FileGrid> = {
  title: 'Components/FileGrid',
  component: FileGrid,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Displays video files with selection checkboxes, metadata, audio streams, pipeline status, and running-job indicators. Bulk actions are driven by the parent (ProjectDetail); this component is a pure view.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof FileGrid>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const createMockVideo = (overrides: Partial<VideoFile> = {}): VideoFile => ({
  id: 1,
  filename: 'tutorial.mp4',
  path: '/videos/tutorial.mp4',
  size_mb: 150.5,
  duration_seconds: 1200,
  audio_streams: [{ index: 0, language: 'rus', codec: 'aac', channels: 2, sample_rate: 48000 }],
  subtitles: [{ language: 'rus', embedded: false, path: '/videos/tutorial.ru.srt' }],
  pipeline_status: { progress: 0, current_stage: '', is_complete: false },
  ...overrides,
});

const fiveVideos: VideoFile[] = [
  createMockVideo({ id: 1, filename: 'intro.mp4', path: '/videos/intro.mp4', size_mb: 120, duration_seconds: 600 }),
  createMockVideo({ id: 2, filename: 'chapter_01.mp4', path: '/videos/chapter_01.mp4', size_mb: 280, duration_seconds: 2400 }),
  createMockVideo({ id: 3, filename: 'chapter_02.mp4', path: '/videos/chapter_02.mp4', size_mb: 340, duration_seconds: 3000 }),
  createMockVideo({ id: 4, filename: 'chapter_03.mp4', path: '/videos/chapter_03.mp4', size_mb: 210, duration_seconds: 1800 }),
  createMockVideo({ id: 5, filename: 'outro.mp4', path: '/videos/outro.mp4', size_mb: 95, duration_seconds: 420 }),
];

// ---------------------------------------------------------------------------
// Stories
// ---------------------------------------------------------------------------

/** Five videos, nothing selected, no running jobs. */
export const Default: Story = {
  args: {
    videos: fiveVideos,
    selectedIds: new Set<number>(),
    onSelectionChange: (ids) => console.log('Selection changed:', [...ids]),
  },
};

/** Two of five rows pre-selected. */
export const SomeSelected: Story = {
  args: {
    videos: fiveVideos,
    selectedIds: new Set<number>([2, 4]),
    onSelectionChange: (ids) => console.log('Selection changed:', [...ids]),
  },
};

/** Every row selected — header checkbox shows checked (not indeterminate). */
export const AllSelected: Story = {
  args: {
    videos: fiveVideos,
    selectedIds: new Set<number>([1, 2, 3, 4, 5]),
    onSelectionChange: (ids) => console.log('Selection changed:', [...ids]),
  },
};

/** Video id=3 has an active job — shows pulsing dot and "▶ View Job" link. */
export const WithRunningJobs: Story = {
  args: {
    videos: fiveVideos,
    selectedIds: new Set<number>(),
    onSelectionChange: (ids) => console.log('Selection changed:', [...ids]),
    runningJobIds: new Map<number, string>([[3, 'task-abc-123']]),
  },
};

/** Videos at various completed pipeline stages. */
export const WithPipelineStatus: Story = {
  args: {
    videos: [
      createMockVideo({
        id: 1,
        filename: 'stage_transcribing.mp4',
        pipeline_status: { progress: 30, current_stage: 'Transcribing', is_complete: false, audio_chunks: 8, transcripts: 40 },
      }),
      createMockVideo({
        id: 2,
        filename: 'stage_tts.mp4',
        pipeline_status: { progress: 55, current_stage: 'Generating TTS', is_complete: false, audio_chunks: 8, transcripts: 120, translated: 120, tts_segments: 60, tts_total: 120 },
      }),
      createMockVideo({
        id: 3,
        filename: 'stage_assembling.mp4',
        pipeline_status: { progress: 78, current_stage: 'Assembling audio', is_complete: false, audio_chunks: 8, transcripts: 120, translated: 120, tts_segments: 120, tts_total: 120, subtitles: 1, audio_assembled: 8, audio_assembled_total: 12 },
      }),
      createMockVideo({
        id: 4,
        filename: 'stage_complete.mp4',
        pipeline_status: { progress: 100, current_stage: 'Complete', is_complete: true, audio_chunks: 8, transcripts: 120, translated: 120, tts_segments: 120, tts_total: 120, subtitles: 1, audio_assembled: 12, audio_assembled_total: 12, video_mixed: true, output_validated: true, backup_created: true, file_replaced: true },
      }),
    ],
    selectedIds: new Set<number>(),
    onSelectionChange: (ids) => console.log('Selection changed:', [...ids]),
  },
};

/** No videos — table renders with header but empty body. */
export const Empty: Story = {
  args: {
    videos: [],
    selectedIds: new Set<number>(),
    onSelectionChange: (ids) => console.log('Selection changed:', [...ids]),
  },
};
