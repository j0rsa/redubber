import type { Meta, StoryObj } from '@storybook/react-vite';
import type { FileNode } from '../components/FileBrowser';
import { ProjectCreation } from '../components/ProjectCreation';
import styles from './NewProject.module.css';

// ── Pure view ──────────────────────────────────────────────────────────────

interface NewProjectViewProps {
  currentPath: string;
  nodes: FileNode[];
  isLoading?: boolean;
  error?: string | null;
  onBack?: () => void;
  onLoadDirectory?: (path: string) => void;
  onCreateProject?: (path: string, name: string) => void;
  onCancel?: () => void;
}

const NewProjectView = ({
  currentPath,
  nodes,
  isLoading = false,
  error = null,
  onBack = () => console.log('back'),
  onLoadDirectory = (p) => console.log('browse', p),
  onCreateProject = (p, n) => console.log('create', { p, n }),
  onCancel = () => console.log('cancel'),
}: NewProjectViewProps) => (
  <div className={styles.page}>
    <header className={styles.header}>
      <button className={styles.backButton} onClick={onBack} type="button">
        ← Back
      </button>
      <h1 className={styles.title}>New Project</h1>
    </header>

    <main className={styles.content}>
      {error && (
        <div className={styles.errorBanner}>{error}</div>
      )}
      <ProjectCreation
        initialPath={currentPath}
        nodes={nodes}
        isLoading={isLoading}
        onLoadDirectory={onLoadDirectory}
        onCreateProject={onCreateProject}
        onCancel={onCancel}
      />
    </main>
  </div>
);

// ─── Meta ──────────────────────────────────────────────────────────────────

const meta: Meta<typeof NewProjectView> = {
  title: 'Pages/NewProject',
  component: NewProjectView,
  decorators: [
    (Story) => (
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', width: '100%' }}>
        <Story />
      </div>
    ),
  ],
  parameters: {
    layout: 'fullscreen',
    backgrounds: { default: 'light-gray' },
    docs: {
      description: {
        component: 'New project page: file browser to select a folder and name the project.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof NewProjectView>;

// ─── Mock data ─────────────────────────────────────────────────────────────

const homeNodes: FileNode[] = [
  { name: 'Documents', path: '/Users/jane/Documents', type: 'directory' },
  { name: 'Videos', path: '/Users/jane/Videos', type: 'directory' },
  { name: 'Downloads', path: '/Users/jane/Downloads', type: 'directory' },
  { name: 'Desktop', path: '/Users/jane/Desktop', type: 'directory' },
];

const videosNodes: FileNode[] = [
  { name: 'Tutorials', path: '/Users/jane/Videos/Tutorials', type: 'directory' },
  { name: 'Meetings', path: '/Users/jane/Videos/Meetings', type: 'directory' },
  { name: 'Presentations', path: '/Users/jane/Videos/Presentations', type: 'directory' },
  { name: 'intro.mp4', path: '/Users/jane/Videos/intro.mp4', type: 'file', size: 524288000 },
  { name: 'demo.mp4', path: '/Users/jane/Videos/demo.mp4', type: 'file', size: 209715200 },
];

const mixedNodes: FileNode[] = [
  { name: 'lecture_01.mp4', path: '/path/lecture_01.mp4', type: 'file', size: 314572800 },
  { name: 'lecture_02.mp4', path: '/path/lecture_02.mp4', type: 'file', size: 471859200 },
  { name: 'lecture_03.mp4', path: '/path/lecture_03.mp4', type: 'file', size: 262144000 },
  { name: 'lecture_01.srt', path: '/path/lecture_01.srt', type: 'file', size: 8192 },
  { name: 'Assets', path: '/path/Assets', type: 'directory' },
];

// ─── Stories ──────────────────────────────────────────────────────────────

/** Home directory — initial state on page open. */
export const Default: Story = {
  args: {
    currentPath: '/Users/jane',
    nodes: homeNodes,
  },
};

/** Browsed into a videos folder. */
export const InsideVideosFolder: Story = {
  args: {
    currentPath: '/Users/jane/Videos',
    nodes: videosNodes,
  },
};

/** Folder with video files ready to become a project. */
export const FolderWithVideos: Story = {
  args: {
    currentPath: '/Users/jane/Videos/Lectures',
    nodes: mixedNodes,
  },
};

/** Loading while browsing a directory. */
export const BrowserLoading: Story = {
  args: {
    currentPath: '/Users/jane/Videos',
    nodes: [],
    isLoading: true,
  },
};

/** Creating in progress (project being saved). */
export const Creating: Story = {
  args: {
    currentPath: '/Users/jane/Videos/Lectures',
    nodes: mixedNodes,
    isLoading: true,
  },
};

/** Empty directory. */
export const EmptyDirectory: Story = {
  args: {
    currentPath: '/Users/jane/EmptyFolder',
    nodes: [],
  },
};

/** Error banner visible. */
export const WithError: Story = {
  args: {
    currentPath: '/Users/jane/Videos',
    nodes: videosNodes,
    error: 'Failed to create project: path does not exist on server.',
  },
};
