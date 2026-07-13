import { useState, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { useProject, useDeleteProject } from '../hooks/useProjects';
import { useVideos, useScanVideos } from '../hooks/useVideos';
import { useSubmitRedub } from '../hooks/useTasks';
import { useActiveTasks } from '../hooks/useActiveTasks';
import { FileGrid } from '../components/FileGrid';
import { ProjectSettingsPanel } from '../components/ProjectSettingsPanel/ProjectSettingsPanel';
import { VoiceRefinement } from '../components/VoiceRefinement/VoiceRefinement';
import { useUIStore } from '../stores/uiStore';
import { apiClient } from '../api/client';
import type { VideoFile, TaskStatus } from '../types';
import styles from './ProjectDetail.module.css';

export const ProjectDetail = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const projectId = id ? parseInt(id, 10) : null;
  const queryClient = useQueryClient();

  const { data: project, isLoading: projectLoading } = useProject(projectId);

  // activeTasks polls every 3s when jobs are running — use it as the source of truth
  const { activeTasks } = useActiveTasks();
  const hasRunningJobs = activeTasks.length > 0;
  const { data: videos, isLoading: videosLoading } = useVideos(projectId, hasRunningJobs);

  // Derive runningJobs from activeTasks by matching video_path to the loaded video list
  // This works even after page reload — no client state needed
  const runningJobs = new Map<number, string>();
  if (videos) {
    for (const task of activeTasks) {
      const video = videos.find((v) => v.path === task.video_path);
      if (video) runningJobs.set(video.id, task.task_id);
    }
  }

  const prevActiveTaskIds = useRef<Set<string>>(new Set());

  useEffect(() => {
    const currentIds = new Set(activeTasks.map((t) => t.task_id));
    let anyCompleted = false;

    // Detect tasks that just dropped out of activeTasks (they completed or failed)
    for (const id of prevActiveTaskIds.current) {
      if (!currentIds.has(id)) {
        anyCompleted = true;
      }
    }

    if (anyCompleted) {
      queryClient.invalidateQueries({ queryKey: ['videos', projectId] });
    }

    prevActiveTaskIds.current = currentIds;
  }, [activeTasks, projectId, queryClient]);

  // Build videoId → TaskStatus for live progress overlay
  const taskStatusByVideoId = new Map<number, TaskStatus>();
  if (videos) {
    for (const video of videos) {
      const runningTaskId = runningJobs.get(video.id);
      if (runningTaskId) {
        const ts = activeTasks.find((t) => t.task_id === runningTaskId);
        if (ts) taskStatusByVideoId.set(video.id, ts);
      }
    }
  }
  const scanVideos = useScanVideos();
  const submitRedub = useSubmitRedub();

  const setCurrentProjectId = useUIStore((state) => state.setCurrentProjectId);
  const hideCompleted = useUIStore((state) => projectId ? (state.hideCompletedByProject[projectId] ?? false) : false);
  const setHideCompleted = useUIStore((state) => state.setHideCompleted);

  const [isVoiceRefinementOpen, setIsVoiceRefinementOpen] = useState(false);
  const [targetLangSaving, setTargetLangSaving] = useState(false);
  const [sourceLangSaving, setSourceLangSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const deleteProject = useDeleteProject();

  const handleDeleteProject = async () => {
    if (!projectId) return;
    try {
      await deleteProject.mutateAsync(projectId);
      navigate('/');
    } catch (err) {
      console.error('Failed to delete project:', err);
    }
  };

  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [batchProgress, setBatchProgress] = useState<{ submitted: number; total: number } | null>(null);
  const [finalizingIds, setFinalizingIds] = useState<Set<number>>(new Set());
  const [generatingSubsIds, setGeneratingSubsIds] = useState<Set<number>>(new Set());

  const handleScan = async () => {
    if (!projectId) return;
    try { await scanVideos.mutateAsync(projectId); }
    catch (err) { console.error('Failed to scan videos:', err); }
  };

  const handleBatchRedub = async (videoFiles: VideoFile[]) => {
    if (!projectId) return;
    setBatchProgress({ submitted: 0, total: videoFiles.length });
    for (const video of videoFiles) {
      try {
        await submitRedub.mutateAsync({ video_path: video.path, project_id: projectId });
        setBatchProgress((prev) => prev ? { ...prev, submitted: prev.submitted + 1 } : null);
      } catch (err) {
        console.error(`Failed to submit ${video.filename}:`, err);
      }
    }
    setBatchProgress(null);
    setSelectedIds(new Set());
  };

  const handleRedubSelected = () => {
    if (!videos) return;
    void handleBatchRedub(videos.filter((v) => selectedIds.has(v.id)));
  };

  const handleRedubAll = () => {
    if (!videos) return;
    void handleBatchRedub(
      videos.filter((v) => !v.pipeline_status?.replaced && !runningJobs.has(v.id))
    );
  };

  const handleRedubSingle = async (videoPath: string) => {
    if (!projectId) return;
    try {
      await submitRedub.mutateAsync({ video_path: videoPath, project_id: projectId });
    } catch (err) {
      console.error('Failed to submit redub:', err);
    }
  };

  const handleFinalize = async (videoId: number) => {
    if (!projectId) return;
    setFinalizingIds((prev) => new Set(prev).add(videoId));
    try {
      await apiClient.post(`/projects/${projectId}/videos/${videoId}/finalize`);
      queryClient.invalidateQueries({ queryKey: ['videos', projectId] });
    } catch (err) {
      console.error('Finalize failed:', err);
    } finally {
      setFinalizingIds((prev) => { const s = new Set(prev); s.delete(videoId); return s; });
    }
  };

  const handleGenerateSubs = async (videoId: number) => {
    if (!projectId) return;
    setGeneratingSubsIds((prev) => new Set(prev).add(videoId));
    try {
      await apiClient.post(`/projects/${projectId}/videos/${videoId}/generate-subtitles`);
      queryClient.invalidateQueries({ queryKey: ['videos', projectId] });
    } catch (err) {
      console.error('Generate subs failed:', err);
    } finally {
      setGeneratingSubsIds((prev) => { const s = new Set(prev); s.delete(videoId); return s; });
    }
  };

  const handleBack = () => {
    setCurrentProjectId(null);
    navigate('/');
  };

  const handleSourceLanguageUpdate = async (lang: string): Promise<void> => {
    if (!projectId) return;
    setSourceLangSaving(true);
    try {
      await apiClient.put(`/projects/${projectId}/source-language`, { source_language: lang });
      queryClient.invalidateQueries({ queryKey: ['project', projectId] });
    } catch (err) {
      console.error('Failed to update source language:', err);
    } finally {
      setSourceLangSaving(false);
    }
  };

  const handleTargetLanguageUpdate = async (lang: string): Promise<void> => {
    if (!projectId) return;
    setTargetLangSaving(true);
    try {
      await apiClient.put(`/projects/${projectId}/target-language`, { target_language: lang });
      queryClient.invalidateQueries({ queryKey: ['project', projectId] });
    } catch (err) {
      console.error('Failed to update target language:', err);
    } finally {
      setTargetLangSaving(false);
    }
  };

  if (projectLoading) {
    return (
      <div className={styles.centered}>
        <p className={styles.loadingText}>Loading project…</p>
      </div>
    );
  }

  if (!project) {
    return (
      <div className={styles.centered}>
        <p className={styles.notFoundText}>Project not found</p>
        <button className={styles.backButton} onClick={handleBack}>
          Back to Projects
        </button>
      </div>
    );
  }

  const hasVideos = videos && videos.length > 0;
  const selectedCount = selectedIds.size;
  const totalCount = videos?.filter((v) => !v.pipeline_status?.replaced && !runningJobs.has(v.id)).length ?? 0;

  return (
    <div className={styles.page}>
      <div className={styles.inner}>

        {/* ── Header ── */}
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <button className={styles.backButton} onClick={handleBack}>
              ← Back
            </button>
            <h1 className={styles.projectName}>{project.name}</h1>
            <p className={styles.projectPath}>{project.path}</p>
            {project.working_directory && (
              <p className={styles.projectPath} title="Working directory for artefacts">
                ↳ {project.working_directory}
              </p>
            )}
          </div>
          <div className={styles.headerActions}>
            <button
              className={styles.scanButton}
              onClick={handleScan}
              disabled={scanVideos.isPending}
            >
              {scanVideos.isPending ? 'Scanning…' : 'Scan for Videos'}
            </button>
            <button
              className={styles.deleteButton}
              onClick={() => setConfirmDelete(true)}
              title="Delete project"
            >
              Delete
            </button>
          </div>
        </div>

        {/* ── Delete confirmation dialog ── */}
        {confirmDelete && (
          <div className={styles.confirmOverlay}>
            <div className={styles.confirmDialog}>
              <h2 className={styles.confirmTitle}>Delete project?</h2>
              <p className={styles.confirmBody}>
                This removes <strong>{project.name}</strong> from Redubber. Your video files are not deleted.
              </p>
              <div className={styles.confirmActions}>
                <button
                  className={styles.confirmDeleteButton}
                  onClick={handleDeleteProject}
                  disabled={deleteProject.isPending}
                >
                  {deleteProject.isPending ? 'Deleting…' : 'Delete'}
                </button>
                <button
                  className={styles.confirmCancelButton}
                  onClick={() => setConfirmDelete(false)}
                  disabled={deleteProject.isPending}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── Error banners ── */}
        {scanVideos.isError && (
          <div className={styles.errorBanner}>
            Failed to scan: {(scanVideos.error as Error).message}
          </div>
        )}
        {submitRedub.isError && (
          <div className={styles.errorBanner}>
            Failed to submit redub: {(submitRedub.error as Error).message}
          </div>
        )}

        {/* ── Project Settings (language + voice, collapsible) ── */}
        <ProjectSettingsPanel
          project={project}
          onOpenVoiceRefinement={() => setIsVoiceRefinementOpen(true)}
          onUpdateSourceLanguage={handleSourceLanguageUpdate}
          onUpdateTargetLanguage={handleTargetLanguageUpdate}
          isSavingSource={sourceLangSaving}
          isSavingTarget={targetLangSaving}
        />

        {/* ── Videos ── */}
        <div className={styles.videosSection}>
          <div className={styles.videosSectionHeader}>
            <h2 className={styles.videosSectionTitle}>Video Files</h2>
            {hasVideos && videos?.some((v) => v.pipeline_status?.replaced) && (
              <button
                className={`${styles.toggleButton} ${hideCompleted ? styles.toggleButtonActive : ''}`}
                onClick={() => projectId && setHideCompleted(projectId, !hideCompleted)}
                title={hideCompleted ? 'Show completed files' : 'Hide completed files'}
              >
                {hideCompleted ? '👁 Show completed' : '✓ Hide completed'}
              </button>
            )}
          </div>

          {hasVideos && (
            <div className={styles.bulkBar}>
              <span className={styles.bulkBarInfo}>
                {batchProgress
                  ? `Submitting ${batchProgress.submitted}/${batchProgress.total}…`
                  : selectedCount > 0
                  ? `${selectedCount} selected`
                  : 'No selection'}
              </span>
              <button
                className={styles.bulkButtonPrimary}
                onClick={handleRedubSelected}
                disabled={selectedCount === 0 || batchProgress !== null}
              >
                Redub Selected{selectedCount > 0 ? ` (${selectedCount})` : ''}
              </button>
              <button
                className={styles.bulkButtonOutline}
                onClick={handleRedubAll}
                disabled={batchProgress !== null}
              >
                Redub All ({totalCount})
              </button>
            </div>
          )}

          {videosLoading ? (
            <p className={styles.loadingText}>Loading videos…</p>
          ) : hasVideos ? (
            <FileGrid
              videos={hideCompleted ? (videos?.filter((v) => !v.pipeline_status?.replaced) ?? []) : (videos ?? [])}
              selectedIds={selectedIds}
              onSelectionChange={setSelectedIds}
              runningJobIds={runningJobs}
              onRedubSingle={handleRedubSingle}
              onFinalize={handleFinalize}
              finalizingIds={finalizingIds}
              onGenerateSubs={handleGenerateSubs}
              generatingSubsIds={generatingSubsIds}
              liveTaskStatuses={taskStatusByVideoId}
              activeTasks={activeTasks}
            />
          ) : (
            <p className={styles.emptyText}>
              No videos found. Click "Scan for Videos" to search the project directory.
            </p>
          )}
        </div>

        {/* ── Voice Refinement modal ── */}
        {projectId && (
          <VoiceRefinement
            projectId={projectId}
            isOpen={isVoiceRefinementOpen}
            onClose={() => setIsVoiceRefinementOpen(false)}
            onSave={() => {
              setIsVoiceRefinementOpen(false);
              queryClient.invalidateQueries({ queryKey: ['project', projectId] });
            }}
            firstVideoPath={videos?.find((v) => !v.pipeline_status?.replaced)?.path}
          />
        )}
      </div>
    </div>
  );
};
