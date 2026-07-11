import type { Meta, StoryObj } from '@storybook/react-vite';
import type { Project } from '../types';
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
            {projects.map((p) => (
              <button
                key={p.id}
                className={styles.projectCard}
                onClick={() => onSelectProject(p.id)}
                type="button"
              >
                <span className={styles.projectIcon}>🎬</span>
                <div className={styles.projectInfo}>
                  <span className={styles.projectName}>{p.name}</span>
                  <span className={styles.projectPath}>{p.path}</span>
                </div>
                <span className={styles.projectDate}>
                  {new Date(p.updated_at).toLocaleDateString(undefined, {
                    year: 'numeric', month: 'short', day: 'numeric',
                  })}
                </span>
              </button>
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
        component: 'Project hub: list of existing projects + button to create a new one.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof ProjectHubView>;

// ─── Mock data ─────────────────────────────────────────────────────────────

const mockProjects: Project[] = [
  {
    id: 1, name: 'Tutorials', path: '/Users/jane/Videos/Tutorials',
    created_at: '2026-01-15T10:30:00Z', updated_at: '2026-06-20T14:00:00Z',
    voice: 'nova', voice_instructions: '', source_language_override: 'rus', target_language: 'eng',
  },
  {
    id: 2, name: 'Meetings', path: '/Users/jane/Videos/Meetings',
    created_at: '2026-02-10T14:20:00Z', updated_at: '2026-07-01T09:00:00Z',
    voice: 'alloy', voice_instructions: '', source_language_override: '', target_language: 'spa',
  },
  {
    id: 3, name: 'Conference Talks', path: '/Users/jane/Videos/Conferences',
    created_at: '2026-03-05T09:15:00Z', updated_at: '2026-07-10T11:00:00Z',
    voice: 'echo', voice_instructions: '', source_language_override: 'zho', target_language: 'fra',
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
      id: i + 1,
      name: `Project ${String(i + 1).padStart(2, '0')}`,
      path: `/Users/jane/Videos/Project_${i + 1}`,
      created_at: new Date(2026, 0, i + 1).toISOString(),
      updated_at: new Date(2026, 5, i + 1).toISOString(),
      voice: ['alloy', 'nova', 'echo', 'fable', 'onyx', 'shimmer'][i % 6],
      voice_instructions: '',
      source_language_override: '',
      target_language: 'eng',
    })),
  },
};
