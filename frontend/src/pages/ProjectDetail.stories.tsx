import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { FileGrid } from '../components/FileGrid';
import { ProjectSettingsPanel } from '../components/ProjectSettingsPanel/ProjectSettingsPanel';
import type { Project, VideoFile } from '../types';
import styles from './ProjectDetail.module.css';

// ── Shared story meta ─────────────────────────────────────────────────────────

const meta: Meta = {
  title: 'Pages/ProjectDetail',
  parameters: {
    layout: 'fullscreen',
    backgrounds: { default: 'light-gray' },
    docs: {
      description: {
        component: 'Project detail page: video list with bulk redub actions and project settings.',
      },
    },
  },
};

export default meta;

// ── Mock data ─────────────────────────────────────────────────────────────────

const mockProject: Project = {
  id: 1,
  path: '/Users/jane/Videos/Tutorials',
  name: 'Tutorials',
  created_at: '2026-01-15T10:30:00Z',
  updated_at: '2026-07-06T15:45:00Z',
  voice: 'nova',
  voice_instructions: 'Clear and professional tone',
  source_language_override: 'rus',
  target_language: 'eng',
  total_videos: 6,
  replaced_videos: 2,
};

const mockVideos: VideoFile[] = [
  {
    id: 1,
    filename: 'intro_tutorial.mp4',
    path: '/Users/jane/Videos/Tutorials/intro_tutorial.mp4',
    size_mb: 125.5,
    duration_seconds: 900,
    audio_streams: [{ index: 0, language: 'rus', codec: 'aac', channels: 2, sample_rate: 48000 }],
    subtitles: [{ language: 'rus', embedded: false, path: '/Users/jane/Videos/Tutorials/intro_tutorial.ru.srt' }],
    pipeline_status: { progress: 0, current_stage: '', is_complete: false },
  },
  {
    id: 2,
    filename: 'advanced_features.mp4',
    path: '/Users/jane/Videos/Tutorials/advanced_features.mp4',
    size_mb: 320.8,
    duration_seconds: 1800,
    audio_streams: [{ index: 0, language: 'rus', codec: 'aac', channels: 2, sample_rate: 48000 }],
    subtitles: [],
    pipeline_status: { progress: 45, current_stage: 'Transcribing', is_complete: false, audio_chunks: 90, transcripts: 40 },
  },
  {
    id: 3,
    filename: 'complete_demo.mp4',
    path: '/Users/jane/Videos/Tutorials/complete_demo.mp4',
    size_mb: 450.2,
    duration_seconds: 2400,
    audio_streams: [
      { index: 0, language: 'rus', codec: 'aac', channels: 2, sample_rate: 48000 },
      { index: 1, language: 'eng', codec: 'aac', channels: 2, sample_rate: 48000 },
    ],
    subtitles: [{ language: 'eng', embedded: true }, { language: 'rus', embedded: true }],
    pipeline_status: { progress: 100, current_stage: 'Complete', is_complete: true },
  },
];

// ── Pure view for stories ─────────────────────────────────────────────────────

interface ProjectDetailViewProps {
  project: Project;
  videos: VideoFile[];
  isLoading?: boolean;
  videosLoading?: boolean;
  scanPending?: boolean;
  submitError?: string | null;
  batchProgressText?: string | null;
}

const ProjectDetailView = ({
  project,
  videos,
  isLoading = false,
  videosLoading = false,
  scanPending = false,
  submitError = null,
  batchProgressText = null,
}: ProjectDetailViewProps) => {
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());

  if (isLoading) {
    return (
      <div className={styles.centered}>
        <p className={styles.loadingText}>Loading project…</p>
      </div>
    );
  }

  const hasVideos = videos.length > 0;
  const selectedCount = selectedIds.size;

  return (
    <div className={styles.page}>
      <div className={styles.inner}>

        {/* Header */}
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <button className={styles.backButton} onClick={() => console.log('back')}>
              ← Back
            </button>
            <h1 className={styles.projectName}>{project.name}</h1>
            <p className={styles.projectPath}>{project.path}</p>
          </div>
          <div className={styles.headerActions}>
            <button
              className={styles.scanButton}
              disabled={scanPending}
              onClick={() => console.log('scan')}
            >
              {scanPending ? 'Scanning…' : 'Scan for Videos'}
            </button>
          </div>
        </div>

        {/* Error banner */}
        {submitError && (
          <div className={styles.errorBanner}>{submitError}</div>
        )}

        {/* Project settings panel */}
        <ProjectSettingsPanel
          project={project}
          onOpenVoiceRefinement={() => console.log('open voice refinement')}
          onUpdateSourceLanguage={async (lang) => console.log('set source lang:', lang)}
          onUpdateTargetLanguage={async (lang) => console.log('set target lang:', lang)}
        />

        {/* Videos */}
        <div className={styles.videosSection}>
          <div className={styles.videosSectionHeader}>
            <h2 className={styles.videosSectionTitle}>Video Files</h2>
          </div>

          {hasVideos && (
            <div className={styles.bulkBar}>
              <span className={styles.bulkBarInfo}>
                {batchProgressText ?? (selectedCount > 0 ? `${selectedCount} selected` : 'No selection')}
              </span>
              <button
                className={styles.bulkButtonPrimary}
                disabled={selectedCount === 0}
                onClick={() => console.log('redub selected')}
              >
                Redub Selected{selectedCount > 0 ? ` (${selectedCount})` : ''}
              </button>
              <button
                className={styles.bulkButtonOutline}
                onClick={() => console.log('redub all')}
              >
                Redub All ({videos.length})
              </button>
            </div>
          )}

          {videosLoading ? (
            <p className={styles.loadingText}>Loading videos…</p>
          ) : hasVideos ? (
            <FileGrid
              videos={videos}
              selectedIds={selectedIds}
              onSelectionChange={setSelectedIds}
              onRedubSingle={(p) => console.log('single redub:', p)}
            />
          ) : (
            <p className={styles.emptyText}>
              No videos found. Click "Scan for Videos" to search the project directory.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

// ── Stories ───────────────────────────────────────────────────────────────────

type Story = StoryObj<typeof ProjectDetailView>;

export const Default: Story = {
  render: () => <ProjectDetailView project={mockProject} videos={mockVideos} />,
};

export const Loading: Story = {
  render: () => <ProjectDetailView project={mockProject} videos={[]} isLoading />,
};

export const EmptyProject: Story = {
  render: () => <ProjectDetailView project={mockProject} videos={[]} />,
};

export const SingleVideo: Story = {
  render: () => <ProjectDetailView project={mockProject} videos={[mockVideos[0]]} />,
};

export const MixedProgress: Story = {
  render: () => <ProjectDetailView project={mockProject} videos={mockVideos} />,
};

export const ScanPending: Story = {
  render: () => <ProjectDetailView project={mockProject} videos={mockVideos} scanPending />,
};

export const BatchInProgress: Story = {
  render: () => (
    <ProjectDetailView
      project={mockProject}
      videos={mockVideos}
      batchProgressText="Submitting 1/3…"
    />
  ),
};

export const WithError: Story = {
  render: () => (
    <ProjectDetailView
      project={mockProject}
      videos={mockVideos}
      submitError="Failed to submit redub: OpenAI API rate limit exceeded."
    />
  ),
};

export const ManyVideos: Story = {
  render: () => (
    <ProjectDetailView
      project={mockProject}
      videos={Array.from({ length: 15 }, (_, i) => ({
        id: i + 1,
        filename: `tutorial_${String(i + 1).padStart(2, '0')}.mp4`,
        path: `/Users/jane/Videos/Tutorials/tutorial_${i + 1}.mp4`,
        size_mb: 50 + i * 30,
        duration_seconds: 300 + i * 120,
        audio_streams: [{ index: 0, language: 'rus', codec: 'aac', channels: 2, sample_rate: 48000 }],
        subtitles: [],
        pipeline_status: {
          progress: (i * 7) % 100,
          current_stage: ['Extracting audio', 'Transcribing', 'Translating', 'Generating TTS', 'Complete'][i % 5],
          is_complete: i % 5 === 4,
        },
      }))}
    />
  ),
};
