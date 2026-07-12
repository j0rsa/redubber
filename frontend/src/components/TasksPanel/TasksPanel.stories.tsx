import type { Meta, StoryObj } from '@storybook/react-vite';
import { useState } from 'react';
import { TasksPanel, TasksIndicator } from './TasksPanel';
import type { TaskStatus } from '../../types';

// ── Shared mock data ────────────────────────────────────────────────────────

const runningTask1: TaskStatus = {
  task_id: 'task-001',
  video_path: '/videos/projects/2024/meeting.mp4',
  status: 'running',
  stage: 'Generating TTS',
  progress: 65,
  created_at: '2024-07-10T10:00:00Z',
  started_at: '2024-07-10T10:01:00Z',
};

const queuedTask: TaskStatus = {
  task_id: 'task-002',
  video_path: '/videos/projects/2024/lecture_01.mp4',
  status: 'queued',
  stage: '',
  progress: 0,
  created_at: '2024-07-10T10:05:00Z',
};

const runningTask2: TaskStatus = {
  task_id: 'task-003',
  video_path: '/videos/projects/2024/tutorial_intro.mp4',
  status: 'running',
  stage: 'Transcribing',
  progress: 35,
  created_at: '2024-07-10T09:50:00Z',
  started_at: '2024-07-10T09:51:00Z',
};

const singleRunningTask: TaskStatus = {
  task_id: 'task-004',
  video_path: '/home/user/movies/documentary_final_cut.mp4',
  status: 'running',
  stage: 'Translating',
  progress: 48,
  created_at: '2024-07-10T11:00:00Z',
  started_at: '2024-07-10T11:00:30Z',
};

// ── Panel meta ──────────────────────────────────────────────────────────────

const panelMeta: Meta<typeof TasksPanel> = {
  title: 'Components/TasksPanel/TasksPanel',
  component: TasksPanel,
  parameters: {
    docs: {
      description: {
        component:
          'Slide-in side panel showing all active (queued/running) tasks. Pure presentational — no routing or API calls.',
      },
    },
    layout: 'fullscreen',
  },
};

export default panelMeta;
type PanelStory = StoryObj<typeof TasksPanel>;

const noop = () => {};

/** Three active tasks: two running, one queued */
export const Default: PanelStory = {
  args: {
    tasks: [runningTask1, queuedTask, runningTask2],
    isOpen: true,
    onClose: noop,
    onViewJob: noop,
  },
};

/** Single running task */
export const SingleTask: PanelStory = {
  args: {
    tasks: [singleRunningTask],
    isOpen: true,
    onClose: noop,
    onViewJob: noop,
  },
};

/** No active tasks — shows empty state */
export const Empty: PanelStory = {
  args: {
    tasks: [],
    isOpen: true,
    onClose: noop,
    onViewJob: noop,
  },
};

/** Panel is closed — renders nothing */
export const Closed: PanelStory = {
  args: {
    tasks: [runningTask1],
    isOpen: false,
    onClose: noop,
    onViewJob: noop,
  },
};

/** Narrow viewport — panel should fill full width */
export const MobilePanel: PanelStory = {
  args: {
    tasks: [runningTask1, queuedTask],
    isOpen: true,
    onClose: noop,
    onViewJob: noop,
  },
  decorators: [
    (Story) => (
      <div style={{ width: '375px', height: '812px', overflow: 'hidden', position: 'relative' }}>
        <Story />
      </div>
    ),
  ],
};

// ── Indicator meta ──────────────────────────────────────────────────────────

const indicatorMeta: Meta<typeof TasksIndicator> = {
  title: 'Components/TasksPanel/TasksIndicator',
  component: TasksIndicator,
  parameters: {
    docs: {
      description: {
        component:
          'Fixed-position circular trigger button with badge count and pulse animation when tasks are active.',
      },
    },
    layout: 'centered',
  },
};

// Indicator stories live under the same `export default` so we need a second file
// Storybook supports one default export per file; split indicator stories here using
// a separate named export trick via a wrapper component.

type IndicatorStory = StoryObj<typeof TasksIndicator>;

/** Re-export the indicator meta in a dedicated story file. */
export const IndicatorWithBadge: IndicatorStory & { __storybookMeta?: typeof indicatorMeta } = {
  render: () => {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const [isOpen, setIsOpen] = useState(false);
    return (
      <div style={{ position: 'relative', width: '100px', height: '100px' }}>
        <TasksIndicator activeCount={3} isOpen={isOpen} onClick={() => setIsOpen((p) => !p)} />
      </div>
    );
  },
};

export const IndicatorNoBadge: IndicatorStory = {
  render: () => {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const [isOpen, setIsOpen] = useState(true); // open=true keeps it visible even with count=0
    return (
      <div style={{ position: 'relative', width: '100px', height: '100px' }}>
        <TasksIndicator activeCount={0} isOpen={isOpen} onClick={() => setIsOpen((p) => !p)} />
      </div>
    );
  },
};
