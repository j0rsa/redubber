import type { Meta, StoryObj } from '@storybook/react-vite';
import { useState } from 'react';
import { expect, fn, userEvent, within } from 'storybook/test';
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

const mixedCohortVideos: VideoFile[] = [
  createMockVideo({ id: 1, filename: 'unfinished-one.mp4', path: '/videos/unfinished-one.mp4' }),
  createMockVideo({ id: 2, filename: 'unfinished-two.mp4', path: '/videos/unfinished-two.mp4' }),
  createMockVideo({
    id: 3,
    filename: 'finished.mp4',
    path: '/videos/finished.mp4',
    pipeline_status: {
      progress: 100,
      current_stage: 'Complete',
      is_complete: true,
      replaced: true,
    },
  }),
];

const InteractiveMixedCohorts = () => {
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  return (
    <FileGrid
      videos={mixedCohortVideos}
      selectedIds={selectedIds}
      onSelectionChange={setSelectedIds}
      targetLanguage="eng"
    />
  );
};

const groupedCohortVideos: VideoFile[] = [
  createMockVideo({
    id: 11,
    filename: 'unfinished-a.mp4',
    path: '/projects/course/section-a/unfinished-a.mp4',
  }),
  createMockVideo({
    id: 12,
    filename: 'finished-a.mp4',
    path: '/projects/course/section-a/finished-a.mp4',
    pipeline_status: {
      progress: 100,
      current_stage: 'Complete',
      is_complete: true,
      replaced: true,
    },
  }),
  createMockVideo({
    id: 13,
    filename: 'unfinished-b.mp4',
    path: '/projects/course/section-b/unfinished-b.mp4',
  }),
  createMockVideo({
    id: 14,
    filename: 'unfinished-c.mp4',
    path: '/projects/course/section-b/unfinished-c.mp4',
  }),
];

const InteractiveGroupedCohorts = () => {
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  return (
    <FileGrid
      projectPath="/projects/course"
      videos={groupedCohortVideos}
      selectedIds={selectedIds}
      onSelectionChange={setSelectedIds}
      targetLanguage="eng"
    />
  );
};

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

/** Video id=4 is finalized — shows "Reset redub" instead of Redub. */
export const WithReplacedVideo: Story = {
  args: {
    videos: [
      ...fiveVideos.slice(0, 3),
      createMockVideo({
        id: 4,
        filename: 'chapter_03.mp4',
        path: '/videos/chapter_03.mp4',
        size_mb: 210,
        duration_seconds: 1800,
        audio_streams: [
          { index: 0, language: 'eng', codec: 'aac', channels: 2, sample_rate: 48000 },
          { index: 1, language: 'rus', codec: 'aac', channels: 2, sample_rate: 48000 },
        ],
        subtitles: [
          {
            language: 'eng',
            embedded: false,
            path: '/videos/chapter_03.en.srt',
            quality_issue_count: 1,
            quality_issues: [
              {
                rule_id: 'known_hallucination_phrase',
                label: 'Known STT phrase',
                message: "contains common STT hallucination phrase(s): 'thank you for watching'",
                segment_index: 3,
              },
            ],
          },
        ],
        pipeline_status: {
          progress: 100,
          current_stage: 'Complete',
          is_complete: true,
          replaced: true,
        },
      }),
    ],
    selectedIds: new Set<number>(),
    onSelectionChange: (ids) => console.log('Selection changed:', [...ids]),
    onResetDub: (id) => console.log('Reset dub:', id),
    targetLanguage: 'eng',
  },
};

/** Finalized dub without pipeline_status.replaced — still shows Reset redub. */
export const TargetStateWithoutReplacedFlag: Story = {
  args: {
    videos: [
      createMockVideo({
        id: 4,
        filename: 'chapter_03.mp4',
        path: '/videos/chapter_03.mp4',
        size_mb: 210,
        duration_seconds: 1800,
        audio_streams: [
          { index: 0, language: 'en', codec: 'aac', channels: 2, sample_rate: 48000 },
          { index: 1, language: 'rus', codec: 'aac', channels: 2, sample_rate: 48000 },
        ],
        subtitles: [{ language: 'eng', embedded: false, path: '/videos/chapter_03.en.srt' }],
        pipeline_status: {
          progress: 100,
          current_stage: 'Complete',
          is_complete: true,
        },
      }),
    ],
    selectedIds: new Set<number>(),
    onSelectionChange: (ids) => console.log('Selection changed:', [...ids]),
    onResetDub: (id) => console.log('Reset dub:', id),
    onRedubSingle: (path) => console.log('Redub:', path),
    targetLanguage: 'eng',
  },
};

/** Failed extract — Redub stays available so the row is not soft-locked. */
export const FailedExtract: Story = {
  args: {
    videos: [
      createMockVideo({
        id: 6,
        filename: 'hairstyles.mp4',
        path: '/videos/hairstyles.mp4',
        duration_seconds: 0,
        audio_streams: [],
        subtitles: [{ language: '', embedded: false, path: '/videos/hairstyles.sub' }],
        pipeline_status: {
          progress: 0,
          current_stage: 'Extract Audio',
          is_complete: false,
          failed: true,
          error: "ffprobe could not read duration for /videos/hairstyles.mp4: Invalid data found",
          replaced: false,
        },
      }),
    ],
    selectedIds: new Set<number>(),
    onSelectionChange: (ids) => console.log('Selection changed:', [...ids]),
    onRedubSingle: (path) => console.log('Redub:', path),
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

/** Generated subs available — shows Review subs next to Redub. */
export const WithGeneratedSubs: Story = {
  args: {
    videos: [
      createMockVideo({
        id: 1,
        filename: 'with_subs.mp4',
        subtitles: [{ language: 'eng', embedded: false, path: '/videos/with_subs.en.srt' }],
        pipeline_status: {
          progress: 100,
          current_stage: 'Complete',
          is_complete: true,
          transcripts: 12,
          subtitles: 1,
        },
      }),
    ],
    selectedIds: new Set<number>(),
    onSelectionChange: (ids) => console.log('Selection changed:', [...ids]),
    onRedubSingle: (path) => console.log('redub', path),
    onReviewSubs: (id) => console.log('review', id),
  },
};

/** Quality-rule badge next to a subtitle that has hallucination / density issues. */
export const WithSubtitleQualityWarnings: Story = {
  args: {
    videos: [
      createMockVideo({
        id: 1,
        filename: 'clean.mp4',
        subtitles: [
          { language: 'rus', embedded: false, path: '/videos/clean.ru.srt' },
          { language: 'eng', embedded: false, path: '/videos/clean.en.srt' },
        ],
      }),
      createMockVideo({
        id: 2,
        filename: 'hallucinated.mp4',
        subtitles: [
          { language: 'rus', embedded: false, path: '/videos/hallucinated.ru.srt' },
          {
            language: 'eng',
            embedded: false,
            path: '/videos/hallucinated.en.srt',
            quality_issue_count: 2,
            quality_issues: [
              {
                rule_id: 'known_hallucination_phrase',
                label: 'Known STT phrase',
                message: "contains common STT hallucination phrase(s): 'thank you for watching'",
                segment_index: 3,
              },
              {
                rule_id: 'excessive_cps',
                label: 'Text too dense',
                message: '45.0 chars/s (>40.0) — text too dense for duration',
                segment_index: 3,
              },
            ],
          },
        ],
      }),
    ],
    selectedIds: new Set<number>(),
    onSelectionChange: (ids) => console.log('Selection changed:', [...ids]),
    onReviewSubs: (id) => console.log('review', id),
  },
};

/** Videos grouped by subfolder — folders sort before filenames (001 before 999). */
export const GroupedByFolder: Story = {
  args: {
    projectPath: '/projects/course',
    videos: [
      createMockVideo({ id: 1, filename: '01.intro.mp4', path: '/projects/course/999/01.intro.mp4' }),
      createMockVideo({ id: 2, filename: '02.main.mp4', path: '/projects/course/999/02.main.mp4' }),
      createMockVideo({ id: 3, filename: '01.lesson.mp4', path: '/projects/course/001/01.lesson.mp4' }),
      createMockVideo({ id: 4, filename: 'trailer.mp4', path: '/projects/course/trailer.mp4' }),
    ],
    selectedIds: new Set<number>(),
    onSelectionChange: (ids) => console.log('Selection changed:', [...ids]),
  },
};

export const SelectAllUsesUnfinishedCohort: Story = {
  render: () => <InteractiveMixedCohorts />,
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const selectAll = canvas.getByRole('checkbox', { name: 'Select all videos' });
    const unfinishedOne = canvas.getByRole('checkbox', { name: 'Select unfinished-one.mp4' });
    const unfinishedTwo = canvas.getByRole('checkbox', { name: 'Select unfinished-two.mp4' });
    const finished = canvas.getByRole('checkbox', { name: 'Select finished.mp4' });

    await expect(finished).toBeEnabled();
    await userEvent.click(selectAll);
    await expect(unfinishedOne).toBeChecked();
    await expect(unfinishedTwo).toBeChecked();
    await expect(finished).not.toBeChecked();
    await expect(finished).toBeDisabled();
  },
};

export const FinishedSelectionLocksUnfinishedCohort: Story = {
  render: () => <InteractiveMixedCohorts />,
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const selectAll = canvas.getByRole('checkbox', { name: 'Select all videos' });
    const unfinishedOne = canvas.getByRole('checkbox', { name: 'Select unfinished-one.mp4' });
    const unfinishedTwo = canvas.getByRole('checkbox', { name: 'Select unfinished-two.mp4' });
    const finished = canvas.getByRole('checkbox', { name: 'Select finished.mp4' });

    await userEvent.click(finished);
    await expect(finished).toBeChecked();
    await expect(unfinishedOne).toBeDisabled();
    await expect(unfinishedTwo).toBeDisabled();
    await expect(selectAll).toBeDisabled();
  },
};

export const SelectUnfinishedByDirectory: Story = {
  render: () => <InteractiveGroupedCohorts />,
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const sectionA = canvas.getByRole('checkbox', {
      name: 'Select unfinished videos in section-a',
    });
    const sectionB = canvas.getByRole('checkbox', {
      name: 'Select unfinished videos in section-b',
    });
    const unfinishedA = canvas.getByRole('checkbox', {
      name: 'Select unfinished-a.mp4',
    });
    const finishedA = canvas.getByRole('checkbox', {
      name: 'Select finished-a.mp4',
    });
    const unfinishedB = canvas.getByRole('checkbox', {
      name: 'Select unfinished-b.mp4',
    });
    const unfinishedC = canvas.getByRole('checkbox', {
      name: 'Select unfinished-c.mp4',
    });

    await userEvent.click(sectionA);
    await expect(unfinishedA).toBeChecked();
    await expect(finishedA).not.toBeChecked();
    await expect(finishedA).toBeDisabled();
    await expect(unfinishedB).not.toBeChecked();
    await expect(sectionA).toBeChecked();

    await userEvent.click(sectionB);
    await expect(unfinishedA).toBeChecked();
    await expect(unfinishedB).toBeChecked();
    await expect(unfinishedC).toBeChecked();
    await expect(sectionB).toBeChecked();
  },
};

export const GeneratedSubtitleWarningsHoldPipeline: Story = {
  args: {
    videos: [
      createMockVideo({
        id: 21,
        filename: 'held-for-review.mp4',
        path: '/videos/held-for-review.mp4',
        subtitles: [
          {
            language: 'eng',
            embedded: false,
            path: '/work/held-for-review.en.srt',
          },
        ],
      }),
    ],
    selectedIds: new Set<number>(),
    onSelectionChange: fn(),
    onResolveSubtitleWarnings: fn(),
    liveTaskStatuses: new Map([
      [
        21,
        {
          task_id: 'held-task',
          video_path: '/videos/held-for-review.mp4',
          status: 'awaiting_subtitle_review',
          stage: 'Subtitle review required',
          progress: 38,
          created_at: '2026-08-21T15:00:00Z',
          quality_issue_count: 1,
          quality_issues: [
            {
              rule_id: 'known_hallucination_phrase',
              label: 'Known STT phrase',
              message: 'contains a known hallucination phrase',
              segment_index: 0,
            },
          ],
        },
      ],
    ]),
  },
  play: async ({ args, canvasElement }) => {
    const canvas = within(canvasElement);
    await expect(canvas.getByText('Review needed')).toBeVisible();
    await expect(
      canvas.getByRole('checkbox', { name: 'Select held-for-review.mp4' }),
    ).toBeDisabled();
    await userEvent.click(
      canvas.getByRole('button', { name: 'Resolve warnings' }),
    );
    await expect(args.onResolveSubtitleWarnings).toHaveBeenCalledOnce();
  },
};
