import type { Meta, StoryObj } from '@storybook/react-vite';
import { SegmentSelector } from './SegmentSelector';
import type { TranscriptionSegment, SegmentFilter } from './types';

const meta: Meta<typeof SegmentSelector> = {
  title: 'Components/VoiceRefinement/SegmentSelector',
  component: SegmentSelector,
  parameters: {
    layout: 'padded',
    backgrounds: {
      default: 'light-gray',
    },
    docs: {
      description: {
        component:
          'Segment selector for choosing a representative audio sample to analyze voice characteristics. Supports smart sampling, search and duration filtering, and paginated load-more.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof SegmentSelector>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const createSegment = (
  id: string,
  videoFilename: string,
  startTime: number,
  endTime: number,
  originalText: string,
  translatedText: string = ''
): TranscriptionSegment => ({
  id,
  video_filename: videoFilename,
  start_time: startTime,
  end_time: endTime,
  duration: endTime - startTime,
  original_text: originalText,
  translated_text: translatedText || `[Translation of: ${originalText.slice(0, 40)}…]`,
  audio_url: `https://example.com/audio/${id}.mp3`,
});

const DEFAULT_FILTER: SegmentFilter = { search: '', minDuration: 0, maxDuration: 0 };

const noop = () => undefined;

const mockSegments: TranscriptionSegment[] = [
  createSegment('seg_001', 'tutorial_intro.mp4', 0.0, 8.5,
    'Welcome to this comprehensive tutorial on advanced programming techniques.'),
  createSegment('seg_002', 'tutorial_intro.mp4', 8.5, 15.2,
    'Today, we will explore the fundamental concepts that every developer should know.'),
  createSegment('seg_003', 'tutorial_intro.mp4', 15.2, 22.8,
    "Let's begin by understanding the core principles of clean code architecture."),
  createSegment('seg_004', 'tutorial_demo.mp4', 0.0, 12.3,
    'In this demonstration, I will show you how to implement these patterns in a real-world application.'),
  createSegment('seg_005', 'tutorial_demo.mp4', 12.3, 20.1,
    'Pay close attention to the error handling strategy we use here.'),
];

// 20-segment dataset for large/paginated stories
const largeSegments: TranscriptionSegment[] = Array.from({ length: 20 }, (_, i) =>
  createSegment(
    `seg_large_${String(i + 1).padStart(3, '0')}`,
    `video_part${Math.floor(i / 5) + 1}.mp4`,
    i * 12.0,
    (i + 1) * 12.0,
    `Segment ${i + 1}: ${['Introduction to the topic.', 'Detailed explanation follows.', 'Here is a worked example.', 'Summary of key points.'][i % 4]}`
  )
);

// ---------------------------------------------------------------------------
// Stories
// ---------------------------------------------------------------------------

/** Five segments, no filter active, nothing selected. */
export const Default: Story = {
  args: {
    segments: mockSegments,
    selectedSegment: null,
    onSelectSegment: (segment) => console.log('Selected:', segment.id),
    isLoading: false,
    filter: DEFAULT_FILTER,
    totalCandidates: undefined,
    hasMore: false,
    onFilterChange: noop,
    onLoadMore: noop,
  },
};

/** Search filter active, 2 results shown from total_matched=2. */
export const WithSearchFilter: Story = {
  args: {
    segments: [mockSegments[0], mockSegments[2]],
    selectedSegment: null,
    onSelectSegment: (segment) => console.log('Selected:', segment.id),
    isLoading: false,
    filter: { search: 'hello', minDuration: 0, maxDuration: 0 },
    totalCandidates: undefined,
    hasMore: false,
    onFilterChange: noop,
    onLoadMore: noop,
  },
};

/** Duration filter active (5–15 s), 12 results visible. */
export const WithDurationFilter: Story = {
  args: {
    segments: largeSegments.slice(0, 12),
    selectedSegment: null,
    onSelectSegment: (segment) => console.log('Selected:', segment.id),
    isLoading: false,
    filter: { search: '', minDuration: 5, maxDuration: 15 },
    totalCandidates: 12,
    hasMore: false,
    onFilterChange: noop,
    onLoadMore: noop,
  },
};

/** has_more=true — Load more button is visible at the bottom. */
export const HasMore: Story = {
  args: {
    segments: mockSegments,
    selectedSegment: null,
    onSelectSegment: (segment) => console.log('Selected:', segment.id),
    isLoading: false,
    filter: DEFAULT_FILTER,
    totalCandidates: 847,
    hasMore: true,
    onFilterChange: noop,
    onLoadMore: () => console.log('Load more clicked'),
  },
};

/** Initial load spinner — no segments yet. */
export const Loading: Story = {
  args: {
    segments: [],
    selectedSegment: null,
    onSelectSegment: noop,
    isLoading: true,
    filter: DEFAULT_FILTER,
    totalCandidates: undefined,
    hasMore: false,
    onFilterChange: noop,
    onLoadMore: noop,
  },
};

/** Appending more results — existing segments visible, Load more button disabled. */
export const LoadingMore: Story = {
  args: {
    segments: mockSegments,
    selectedSegment: null,
    onSelectSegment: (segment) => console.log('Selected:', segment.id),
    isLoading: true,
    filter: DEFAULT_FILTER,
    totalCandidates: 847,
    hasMore: true,
    onFilterChange: noop,
    onLoadMore: noop,
  },
};

/** 20 segments shown from a 1247-segment corpus (evenly sampled). */
export const LargeDataset: Story = {
  args: {
    segments: largeSegments,
    selectedSegment: null,
    onSelectSegment: (segment) => console.log('Selected:', segment.id),
    isLoading: false,
    filter: DEFAULT_FILTER,
    totalCandidates: 1247,
    hasMore: true,
    onFilterChange: noop,
    onLoadMore: () => console.log('Load more clicked'),
  },
};

/** Search with 0 matches. */
export const NoResults: Story = {
  args: {
    segments: [],
    selectedSegment: null,
    onSelectSegment: noop,
    isLoading: false,
    filter: { search: 'xyzzy not found', minDuration: 0, maxDuration: 0 },
    totalCandidates: 0,
    hasMore: false,
    onFilterChange: noop,
    onLoadMore: noop,
  },
};

/** One segment highlighted as selected. */
export const Selected: Story = {
  args: {
    segments: mockSegments,
    selectedSegment: mockSegments[1],
    onSelectSegment: (segment) => console.log('Selected:', segment.id),
    isLoading: false,
    filter: DEFAULT_FILTER,
    totalCandidates: 847,
    hasMore: false,
    onFilterChange: noop,
    onLoadMore: noop,
  },
};

/** Dark background decorator to verify contrast. */
export const DarkBackground: Story = {
  args: {
    segments: mockSegments,
    selectedSegment: mockSegments[0],
    onSelectSegment: (segment) => console.log('Selected:', segment.id),
    isLoading: false,
    filter: DEFAULT_FILTER,
    totalCandidates: 847,
    hasMore: false,
    onFilterChange: noop,
    onLoadMore: noop,
  },
  parameters: {
    backgrounds: { default: 'dark' },
  },
};
