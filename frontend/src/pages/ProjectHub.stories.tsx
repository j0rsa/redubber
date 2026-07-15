import type { Meta, StoryObj } from '@storybook/react-vite';
import type { Project } from '../types';
import { ProjectCard } from './ProjectHub';
import styles from './ProjectHub.module.css';

// ── Pure view (no routing/query deps) ─────────────────────────────────────

interface ProjectHubViewProps {
  projects: Project[];
  isLoading?: boolean;
  isError?: boolean;
  onNewProject?: () => void;
  onSelectProject?: (id: number) => void;
}

const ProjectHubView = ({
  projects,
  isLoading = false,
  isError = false,
  onNewProject = () => console.log('new project'),
  onSelectProject = (id) => console.log('open project', id),
}: ProjectHubViewProps) => (
  <div className={styles.page}>
    <header className={styles.header}>
      <div className={styles.headerContent}>
        <div>
          <h1 className={styles.title}>Redubber</h1>
          <p className={styles.subtitle}>AI-powered video dubbing</p>
        </div>
        <button className={styles.newProjectButton} onClick={onNewProject} type="button">
          + New Project
        </button>
      </div>
    </header>

    <main className={styles.content}>
      {isLoading && (
        <div className={styles.grid}>
          {[1, 2, 3, 4].map((n) => <div key={n} className={styles.skeletonCard} />)}
        </div>
      )}

      {isError && (
        <div className={styles.errorState}>
          <p>Could not load projects.</p>
          <p className={styles.errorDetail}>Network request failed</p>
        </div>
      )}

      {!isLoading && !isError && (
        projects.length > 0 ? (
          <div className={styles.grid}>
            {projects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                onClick={() => onSelectProject(project.id)}
              />
            ))}
          </div>
        ) : (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>📁</div>
            <h2 className={styles.emptyTitle}>No projects yet</h2>
            <p className={styles.emptyText}>
              Create your first project to start dubbing videos.
            </p>
            <button className={styles.newProjectButton} onClick={onNewProject} type="button">
              + Create First Project
            </button>
          </div>
        )
      )}
    </main>
  </div>
);

// ─── Meta ─────────────────────────────────────────────────────────────────

const meta: Meta<typeof ProjectHubView> = {
  title: 'Pages/ProjectHub',
  component: ProjectHubView,
  parameters: {
    layout: 'fullscreen',
    backgrounds: { default: 'light-gray' },
    docs: {
      description: {
        component:
          'Project hub listing with readiness progress bars showing replaced/total video counts.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof ProjectHubView>;

// ─── Mock data ─────────────────────────────────────────────────────────────

const baseProject = {
  voice: 'nova',
  voice_instructions: '',
  source_language_override: '',
  target_language: 'eng',
} satisfies Partial<Project>;

const mockProjects: Project[] = [
  {
    ...baseProject,
    id: 1,
    name: 'Tutorials',
    path: '/Users/jane/Videos/Tutorials',
    created_at: '2026-01-15T10:30:00Z',
    updated_at: '2026-06-20T14:00:00Z',
    source_language_override: 'rus',
    total_videos: 12,
    replaced_videos: 8,
  },
  {
    ...baseProject,
    id: 2,
    name: 'Meetings',
    path: '/Users/jane/Videos/Meetings',
    created_at: '2026-02-10T14:20:00Z',
    updated_at: '2026-07-01T09:00:00Z',
    voice: 'alloy',
    target_language: 'spa',
    total_videos: 5,
    replaced_videos: 0,
  },
  {
    ...baseProject,
    id: 3,
    name: 'Conference Talks',
    path: '/Users/jane/Videos/Conferences',
    created_at: '2026-03-05T09:15:00Z',
    updated_at: '2026-07-10T11:00:00Z',
    voice: 'echo',
    target_language: 'fra',
    total_videos: 3,
    replaced_videos: 3,
  },
];

// ─── Stories ──────────────────────────────────────────────────────────────

export const Default: Story = {
  args: { projects: mockProjects },
};

export const Empty: Story = {
  args: { projects: [] },
};

export const Loading: Story = {
  args: { projects: [], isLoading: true },
};

export const Error: Story = {
  args: { projects: [], isError: true },
};

export const SingleProject: Story = {
  args: { projects: [mockProjects[0]] },
};

export const ManyProjects: Story = {
  args: {
    projects: Array.from({ length: 10 }, (_, i) => ({
      ...baseProject,
      id: i + 1,
      name: `Project ${String(i + 1).padStart(2, '0')}`,
      path: `/Users/jane/Videos/Project_${i + 1}`,
      created_at: new Date(2026, 0, i + 1).toISOString(),
      updated_at: new Date(2026, 5, i + 1).toISOString(),
      voice: ['alloy', 'nova', 'echo', 'fable', 'onyx', 'shimmer'][i % 6],
      total_videos: (i + 1) * 2,
      replaced_videos: i,
    })),
  },
};

/** Side-by-side cards showing each progress bar state. */
export const ProgressBarStates: Story = {
  name: 'Progress Bar States',
  args: {
    projects: [
      {
        ...baseProject,
        id: 101,
        name: 'Not started',
        path: '/Videos/not-started',
        created_at: '2026-01-01T00:00:00Z',
        updated_at: '2026-07-01T00:00:00Z',
        total_videos: 10,
        replaced_videos: 0,
      },
      {
        ...baseProject,
        id: 102,
        name: 'In progress (67%)',
        path: '/Videos/in-progress',
        created_at: '2026-01-01T00:00:00Z',
        updated_at: '2026-07-02T00:00:00Z',
        total_videos: 12,
        replaced_videos: 8,
      },
      {
        ...baseProject,
        id: 103,
        name: 'Complete',
        path: '/Videos/complete',
        created_at: '2026-01-01T00:00:00Z',
        updated_at: '2026-07-03T00:00:00Z',
        total_videos: 6,
        replaced_videos: 6,
      },
      {
        ...baseProject,
        id: 104,
        name: 'No bar (not scanned yet)',
        path: '/Videos/legacy-project',
        created_at: '2025-06-01T00:00:00Z',
        updated_at: '2025-06-01T00:00:00Z',
        total_videos: 0,
        replaced_videos: 0,
      },
    ],
  },
  parameters: {
    docs: {
      description: {
        story:
          'Progress bars appear when total_videos > 0. The label shows replaced/total counts; fill width reflects the percentage.',
      },
    },
  },
};

/** Isolated card for Storybook controls on progress values. */
export const ProgressBarCard: StoryObj<typeof ProjectCard> = {
  name: 'Progress Bar Card',
  render: (args) => (
    <div className={styles.page}>
      <main className={styles.content}>
        <div className={styles.grid} style={{ maxWidth: 420 }}>
          <ProjectCard {...args} />
        </div>
      </main>
    </div>
  ),
  args: {
    project: mockProjects[0],
    onClick: () => console.log('open project'),
  },
  argTypes: {
    project: { control: 'object' },
    onClick: { action: 'clicked' },
  },
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        story: 'Single project card — tweak total_videos and replaced_videos in Controls.',
      },
    },
  },
};
