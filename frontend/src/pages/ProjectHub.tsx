import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { useProjects } from '../hooks/useProjects';
import { useUIStore } from '../stores/uiStore';
import { InstallPrompt } from '../components/InstallPrompt';
import type { Project } from '../types';
import styles from './ProjectHub.module.css';
import { useState } from 'react';
import { Settings } from '../components/Settings/Settings';
import { useSettings } from '../hooks/useSettings';
import { apiClient } from '../api/client';

export const ProjectHub = () => {
  const navigate = useNavigate();
  const { data: projects, isLoading, error } = useProjects();
  const setCurrentProjectId = useUIStore((state) => state.setCurrentProjectId);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { settings, isSaving, error: settingsError, successMessage, saveSettings } = useSettings();
  const { data: health } = useQuery<{ version: string }>({
    queryKey: ['health'],
    queryFn: async () => (await apiClient.get('/health')).data,
    staleTime: Infinity,
  });

  const handleSelectProject = (id: number) => {
    setCurrentProjectId(id);
    navigate(`/project/${id}`);
  };

  return (
    <>
      <InstallPrompt />
      <div className={styles.page}>
        <header className={styles.header}>
          <div className={styles.headerContent}>
            <div className={styles.brand}>
              <h1 className={styles.title}>
                Redubber
                {health?.version && (
                  <sup className={styles.version}>v{health.version}</sup>
                )}
              </h1>
              <p className={styles.subtitle}>AI-powered video dubbing</p>
            </div>
            <div className={styles.headerActions}>
              <button
                className={styles.settingsButton}
                onClick={() => setSettingsOpen(true)}
                type="button"
                title="Settings"
              >
                ⚙
              </button>
              <button
                className={styles.newProjectButton}
                onClick={() => navigate('/new-project')}
                type="button"
              >
                + New Project
              </button>
            </div>
          </div>
        </header>
        {settingsOpen && (
          <div className={styles.settingsOverlay} onClick={() => setSettingsOpen(false)}>
            <div className={styles.settingsPanel} onClick={(e) => e.stopPropagation()}>
              <div className={styles.settingsPanelHeader}>
                <h2 className={styles.settingsPanelTitle}>Settings</h2>
                <button className={styles.settingsCloseButton} onClick={() => setSettingsOpen(false)}>✕</button>
              </div>
              <Settings
                settings={settings}
                isSaving={isSaving}
                error={settingsError}
                successMessage={successMessage}
                onSave={saveSettings}
              />
            </div>
          </div>
        )}

        <main className={styles.content}>
          {isLoading && (
            <div className={styles.grid}>
              {[1, 2, 3, 4].map((n) => (
                <div key={n} className={styles.skeletonCard} />
              ))}
            </div>
          )}

          {error && (
            <div className={styles.errorState}>
              <p>Could not load projects.</p>
              <p className={styles.errorDetail}>{(error as Error).message}</p>
            </div>
          )}

          {!isLoading && !error && (
            <>
              {projects && projects.length > 0 ? (
                <div className={styles.grid}>
                  {projects.map((project) => (
                    <ProjectCard
                      key={project.id}
                      project={project}
                      onClick={() => handleSelectProject(project.id)}
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
                  <button
                    className={styles.newProjectButton}
                    onClick={() => navigate('/new-project')}
                    type="button"
                  >
                    + Create First Project
                  </button>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </>
  );
};

// ── Project card ──────────────────────────────────────────────────────────────

interface ProjectCardProps {
  project: Project;
  onClick: () => void;
}

const ProjectCard = ({ project, onClick }: ProjectCardProps) => {
  const date = new Date(project.updated_at).toLocaleDateString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
  });

  const total = project.total_videos ?? 0;
  const replaced = project.replaced_videos ?? 0;
  const pct = total > 0 ? Math.round((replaced / total) * 100) : 0;

  return (
    <button className={styles.projectCard} onClick={onClick} type="button">
      <span className={styles.projectIcon}>🎬</span>
      <div className={styles.projectInfo}>
        <span className={styles.projectName}>{project.name}</span>
        <span className={styles.projectPath}>{project.path}</span>
        {total > 0 && (
          <div className={styles.progressRow}>
            <div className={styles.progressBar}>
              <div className={styles.progressFill} style={{ width: `${pct}%` }} />
            </div>
            <span className={styles.progressLabel}>{replaced}/{total}</span>
          </div>
        )}
      </div>
      <span className={styles.projectDate}>{date}</span>
    </button>
  );
};
