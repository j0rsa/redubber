import type { Meta, StoryObj } from '@storybook/react-vite';
import { JobMonitorView } from './JobMonitor';
import type { TaskStatus } from '../types';

const meta: Meta<typeof JobMonitorView> = {
  title: 'Pages/JobMonitor',
  component: JobMonitorView,
  parameters: {
    layout: 'fullscreen',
    backgrounds: { default: 'light-gray' },
    docs: {
      description: {
        component: 'Real-time job monitoring page showing task progress, status, and cancellation.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof JobMonitorView>;

// ─── Mock data helpers ────────────────────────────────────────────────────────

const task = (overrides: Partial<TaskStatus> = {}): TaskStatus => ({
  task_id: 'task_abc123',
  video_path: '/Users/jane/Videos/Meetings/q4_review.mp4',
  status: 'running',
  stage: 'Generating TTS',
  progress: 65,
  created_at: new Date(Date.now() - 300_000).toISOString(),
  started_at: new Date(Date.now() - 240_000).toISOString(),
  ...overrides,
});

const noop = () => console.log('clicked');

// ─── Stories ─────────────────────────────────────────────────────────────────

export const Running: Story = {
  args: {
    task: task({ stage: 'Generating TTS', progress: 65 }),
    isLoading: false, isCanceling: false, cancelError: null,
    onBack: noop, onCancel: noop,
  },
};

export const Queued: Story = {
  args: {
    task: task({ status: 'queued', stage: '', progress: 0, started_at: undefined }),
    isLoading: false, isCanceling: false, cancelError: null,
    onBack: noop, onCancel: noop,
  },
};

export const ExtractingAudio: Story = {
  args: {
    task: task({ stage: 'Extracting audio', progress: 10 }),
    isLoading: false, isCanceling: false, cancelError: null,
    onBack: noop, onCancel: noop,
  },
};

export const Transcribing: Story = {
  args: {
    task: task({ stage: 'Transcribing', progress: 35 }),
    isLoading: false, isCanceling: false, cancelError: null,
    onBack: noop, onCancel: noop,
  },
};

export const AlmostComplete: Story = {
  args: {
    task: task({ stage: 'Finalizing', progress: 95 }),
    isLoading: false, isCanceling: false, cancelError: null,
    onBack: noop, onCancel: noop,
  },
};

export const Completed: Story = {
  args: {
    task: task({
      status: 'completed', stage: 'Complete', progress: 100,
      completed_at: new Date(Date.now() - 10_000).toISOString(),
    }),
    isLoading: false, isCanceling: false, cancelError: null,
    onBack: noop, onCancel: noop,
  },
};

export const Failed: Story = {
  args: {
    task: task({
      status: 'failed', stage: 'Generating TTS', progress: 65,
      error: 'OpenAI API rate limit exceeded. Please try again in a few minutes.',
      completed_at: new Date().toISOString(),
    }),
    isLoading: false, isCanceling: false, cancelError: null,
    onBack: noop, onCancel: noop,
  },
};

export const FailedLongError: Story = {
  args: {
    task: task({
      status: 'failed', stage: 'Extracting audio', progress: 5,
      error: 'FFmpeg error: Invalid video codec. The input file may be corrupted or in an unsupported format. Supported formats include: mp4, mkv, avi, mov, webm. Please check the video file and try again.',
      completed_at: new Date().toISOString(),
    }),
    isLoading: false, isCanceling: false, cancelError: null,
    onBack: noop, onCancel: noop,
  },
};

export const Canceling: Story = {
  args: {
    task: task({ stage: 'Generating TTS', progress: 65 }),
    isLoading: false, isCanceling: true, cancelError: null,
    onBack: noop, onCancel: noop,
  },
};

export const CancelError: Story = {
  args: {
    task: task({ stage: 'Generating TTS', progress: 65 }),
    isLoading: false, isCanceling: false,
    cancelError: 'Server returned 500: Internal Server Error',
    onBack: noop, onCancel: noop,
  },
};

export const Loading: Story = {
  args: {
    task: undefined, isLoading: true, isCanceling: false, cancelError: null,
    onBack: noop, onCancel: noop,
  },
};

export const TaskNotFound: Story = {
  args: {
    task: null, isLoading: false, isCanceling: false, cancelError: null,
    onBack: noop, onCancel: noop,
  },
};

export const LongVideoPath: Story = {
  args: {
    task: task({
      video_path: '/Users/jane/Documents/Work/Projects/Q4-2025/Marketing/Videos/Tutorials/Complete-Guide-to-Product-Features-Final-Version-Approved-December-2025.mp4',
      stage: 'Generating TTS', progress: 50,
    }),
    isLoading: false, isCanceling: false, cancelError: null,
    onBack: noop, onCancel: noop,
  },
};
