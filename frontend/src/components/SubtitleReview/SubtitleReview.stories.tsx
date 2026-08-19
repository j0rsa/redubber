import type { Meta, StoryObj } from '@storybook/react-vite';
import { SubtitleReview } from './SubtitleReview';
import type { SubtitleReviewData } from './types';

const meta: Meta<typeof SubtitleReview> = {
  title: 'Components/SubtitleReview',
  component: SubtitleReview,
  parameters: { layout: 'fullscreen' },
};

export default meta;
type Story = StoryObj<typeof SubtitleReview>;

const qualityRules = [
  { id: 'known_hallucination_phrase', label: 'Known STT phrase', scope: 'cue' as const },
  { id: 'excessive_cps', label: 'Text too dense', scope: 'cue' as const },
  { id: 'shared_phrase_run', label: 'Shared phrase run', scope: 'cue' as const },
];

const sample: SubtitleReviewData = {
  video_id: 1,
  filename: 'lesson.mp4',
  srt_path: '/videos/lesson.en.srt',
  available_files: [
    { path: '/videos/lesson.en.srt', label: 'lesson.en.srt', source: 'generated' },
  ],
  total: 4,
  returned: 4,
  has_chunks: true,
  has_tts: true,
  hallucination_warnings: [],
  quality_rules: qualityRules,
  quality_breaches: [],
  segments: [
    {
      index: 0,
      start: 0,
      end: 4.2,
      duration: 4.2,
      text: 'Welcome to this lesson on the structure of the chest area.',
      original: { chunk_url: '/orig/0', chunk_name: 'lesson_001.m4a', seek_start: 0, seek_end: 4.2 },
      tts_url: '/tts/0',
      breached_rule_count: 0,
      breached_rules: [],
    },
    {
      index: 1,
      start: 4.5,
      end: 9.1,
      duration: 4.6,
      text: 'Today we will look at the ribs, the sternum, and how they protect the organs inside.',
      original: { chunk_url: '/orig/0', chunk_name: 'lesson_001.m4a', seek_start: 4.5, seek_end: 9.1 },
      tts_url: '/tts/1',
      breached_rule_count: 0,
      breached_rules: [],
    },
    {
      index: 2,
      start: 9.4,
      end: 11.0,
      duration: 1.6,
      text: 'Let us begin.',
      original: { chunk_url: '/orig/0', chunk_name: 'lesson_001.m4a', seek_start: 9.4, seek_end: 11.0 },
      tts_url: null,
      breached_rule_count: 0,
      breached_rules: [],
    },
    {
      index: 3,
      start: 12.0,
      end: 28.5,
      duration: 16.5,
      text: 'Pay attention to this long explanation of how the diaphragm moves during breathing and why that matters for the next chapter.',
      original: { chunk_url: '/orig/0', chunk_name: 'lesson_001.m4a', seek_start: 12, seek_end: 28.5 },
      tts_url: '/tts/3',
      breached_rule_count: 0,
      breached_rules: [],
    },
  ],
};

export const Default: Story = {
  args: {
    isOpen: true,
    onClose: () => console.log('close'),
    data: sample,
    loading: false,
    error: null,
  },
};

export const Loading: Story = {
  args: {
    isOpen: true,
    onClose: () => console.log('close'),
    filename: 'lesson.mp4',
    data: null,
    loading: true,
    error: null,
  },
};

export const MissingArtefacts: Story = {
  args: {
    isOpen: true,
    onClose: () => console.log('close'),
    data: {
      ...sample,
      has_chunks: false,
      has_tts: false,
      segments: sample.segments.map((s) => ({ ...s, original: null, tts_url: null })),
    },
    loading: false,
    error: null,
  },
};

export const MultipleFiles: Story = {
  args: {
    isOpen: true,
    onClose: () => console.log('close'),
    data: {
      ...sample,
      available_files: [
        { path: '/videos/lesson.en.srt', label: 'lesson.en.srt', source: 'generated' },
        { path: '/videos/lesson.ru.srt', label: 'lesson.ru.srt', source: 'sidecar' },
      ],
    },
    loading: false,
    error: null,
  },
};

export const WithHallucinationWarnings: Story = {
  args: {
    isOpen: true,
    onClose: () => console.log('close'),
    data: {
      ...sample,
      quality_rules: qualityRules,
      quality_breaches: [
        {
          rule_id: 'known_hallucination_phrase',
          message: "contains common STT hallucination phrase(s): 'thank you for watching'",
          segment_index: 3,
        },
        {
          rule_id: 'excessive_cps',
          message: '45.0 chars/s (>40.0) — text too dense for duration',
          segment_index: 3,
        },
      ],
      hallucination_warnings: [
        {
          code: 'known_hallucination_phrase',
          message: "contains common STT hallucination phrase(s): 'thank you for watching'",
          segment_index: 3,
        },
        {
          code: 'excessive_cps',
          message: '45.0 chars/s (>40.0) — text too dense for duration',
          segment_index: 3,
        },
      ],
      segments: sample.segments.map((segment) =>
        segment.index === 3
          ? {
              ...segment,
              breached_rule_count: 2,
              breached_rules: ['known_hallucination_phrase', 'excessive_cps'],
            }
          : segment,
      ),
    },
    loading: false,
    error: null,
  },
};
