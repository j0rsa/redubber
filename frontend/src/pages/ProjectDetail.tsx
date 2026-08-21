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
import { SubtitleReview } from '../components/SubtitleReview/SubtitleReview';
import { RetryRedubDialog } from '../components/RetryRedubDialog/RetryRedubDialog';
import { ResetDubDialog } from '../components/ResetDubDialog/ResetDubDialog';
import type { ResetToStageId } from '../components/ResetDubDialog/stages';
import { useSettings } from '../hooks/useSettings';
import { useUIStore } from '../stores/uiStore';
import { apiClient } from '../api/client';
import { formatDuration } from '../utils/format';
import { getApiErrorMessage } from '../utils/apiError';
import { useSubtitleReview } from '../hooks/useSubtitleReview';
import { isVideoInTargetState } from '../utils/language';
import type { VideoFile, TaskStatus } from '../types';
import styles from './ProjectDetail.module.css';

export const ProjectDetail = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const projectId = id ? parseInt(id, 10) : null;
  const queryClient = useQueryClient();

  const { data: project, isLoading: projectLoading } = useProject(projectId);

  const isFinalized = (video: VideoFile) =>
    Boolean(video.pipeline_status?.replaced)
    || isVideoInTargetState(
      video.audio_streams,
      video.subtitles,
      project?.target_language ?? '',
    );

  // activeTasks polls every 3s when jobs are running — use it as the source of truth
  const { activeTasks, hasActive } = useActiveTasks();
  const hasRunningJobs = hasActive;
  const { data: videos, isLoading: videosLoading } = useVideos(projectId, hasRunningJobs);

  // Derive runningJobs from activeTasks by matching video_path to the loaded video list
  // This works even after page reload — no client state needed
  const runningJobs = new Map<number, string>();
  if (videos) {
    for (const task of activeTasks) {
      if (task.status !== 'queued' && task.status !== 'running') continue;
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
      queryClient.invalidateQueries({ queryKey: ['project', projectId] });
      queryClient.invalidateQueries({ queryKey: ['projects'] });
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
        continue;
      }
      const failedTask = activeTasks.find(
        (t) => t.video_path === video.path && t.status === 'failed',
      );
      if (failedTask) taskStatusByVideoId.set(video.id, failedTask);
    }
  }
  const scanVideos = useScanVideos();
  const submitRedub = useSubmitRedub();
  const { settings } = useSettings();

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
  const [reviewVideoId, setReviewVideoId] = useState<number | null>(null);
  const [reviewSrtPath, setReviewSrtPath] = useState<string | null>(null);
  const subtitleReview = useSubtitleReview({
    projectId,
    videoId: reviewVideoId,
    srtPath: reviewSrtPath,
  });

  useEffect(() => {
    setReviewSrtPath(null);
  }, [reviewVideoId]);
  const [resettingDubIds, setResettingDubIds] = useState<Set<number>>(new Set());
  const [confirmResetVideo, setConfirmResetVideo] = useState<VideoFile | null>(null);
  const [resetDubError, setResetDubError] = useState<string | null>(null);
  const [lastResetTo, setLastResetTo] = useState<ResetToStageId>('start');
  const [retryVideo, setRetryVideo] = useState<VideoFile | null>(null);

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
    if (!videos || !project) return;
    void handleBatchRedub(
      videos.filter((v) => !isFinalized(v) && !runningJobs.has(v.id))
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

  const handleRetryResetDub = async (video: VideoFile) => {
    if (!projectId) return;
    setResettingDubIds((prev) => new Set(prev).add(video.id));
    try {
      await apiClient.post(
        `/projects/${projectId}/videos/${video.id}/reset-dub`,
        null,
        { params: { reset_to: lastResetTo } },
      );
      await queryClient.invalidateQueries({ queryKey: ['tasks'] });
    } catch (err) {
      const message = getApiErrorMessage(err, 'Failed to retry reset redub');
      setResetDubError(message);
      console.error('Failed to retry reset redub:', err);
    } finally {
      setResettingDubIds((prev) => {
        const s = new Set(prev);
        s.delete(video.id);
        return s;
      });
    }
  };

  const handleRetryFailed = (video: VideoFile) => {
    setRetryVideo(video);
  };

  const handleRetryConfirm = async (audioChunkDuration: number) => {
    if (!projectId || !retryVideo) return;
    try {
      const result = await submitRedub.mutateAsync({
        video_path: retryVideo.path,
        project_id: projectId,
        audio_chunk_duration: audioChunkDuration,
      });
      setRetryVideo(null);
      if (result?.task_id) {
        navigate(`/job/${result.task_id}`);
      }
      await queryClient.invalidateQueries({ queryKey: ['tasks'] });
    } catch (err) {
      console.error('Failed to retry redub:', err);
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

  const handleResetDub = async (videoId: number, resetTo: ResetToStageId) => {
    if (!projectId) return;
    setResetDubError(null);
    setLastResetTo(resetTo);
    setResettingDubIds((prev) => new Set(prev).add(videoId));
    try {
      await apiClient.post(
        `/projects/${projectId}/videos/${videoId}/reset-dub`,
        null,
        { params: { reset_to: resetTo } },
      );
      setConfirmResetVideo(null);
      setResetDubError(null);
      await queryClient.invalidateQueries({ queryKey: ['tasks'] });
    } catch (err) {
      const message = getApiErrorMessage(err, 'Failed to reset redub');
      setResetDubError(message);
      await queryClient.invalidateQueries({ queryKey: ['videos', projectId] });
      await queryClient.invalidateQueries({ queryKey: ['project', projectId] });
    } finally {
      setResettingDubIds((prev) => {
        const s = new Set(prev);
        s.delete(videoId);
        return s;
      });
    }
  };

  // While a reset-dub job runs, FileGrid shows "View Job" via runningJobs.

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
  const selectedDurationSeconds = videos
    ? videos
        .filter((v) => selectedIds.has(v.id))
        .reduce((sum, v) => sum + (v.duration_seconds || 0), 0)
    : 0;
  const totalCount = videos?.filter((v) => !isFinalized(v) && !runningJobs.has(v.id)).length ?? 0;

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

        {/* ── Reset redub dialog ── */}
        {confirmResetVideo && (
          <ResetDubDialog
            videoFilename={confirmResetVideo.filename}
            currentStage="complete"
            isSubmitting={resettingDubIds.has(confirmResetVideo.id)}
            errorMessage={resetDubError}
            onCancel={() => {
              setConfirmResetVideo(null);
              setResetDubError(null);
            }}
            onConfirm={(resetTo) => void handleResetDub(confirmResetVideo.id, resetTo)}
          />
        )}

        {retryVideo && (
          <RetryRedubDialog
            videoFilename={retryVideo.filename}
            errorMessage={
              retryVideo.pipeline_status?.error
              ?? taskStatusByVideoId.get(retryVideo.id)?.error
              ?? undefined
            }
            defaultChunkDuration={settings.audio_chunk_duration}
            isSubmitting={submitRedub.isPending}
            onCancel={() => setRetryVideo(null)}
            onConfirm={(audioChunkDuration) => void handleRetryConfirm(audioChunkDuration)}
          />
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
            {hasVideos && videos?.some((v) => isFinalized(v)) && (
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
                  ? `${selectedCount} selected · ${formatDuration(selectedDurationSeconds)}`
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
              videos={hideCompleted ? (videos?.filter((v) => !isFinalized(v)) ?? []) : (videos ?? [])}
              projectPath={project.path}
              selectedIds={selectedIds}
              onSelectionChange={setSelectedIds}
              runningJobIds={runningJobs}
              onRedubSingle={handleRedubSingle}
              onRetryFailed={handleRetryFailed}
              onRetryResetDub={handleRetryResetDub}
              onFinalize={handleFinalize}
              finalizingIds={finalizingIds}
              onGenerateSubs={handleGenerateSubs}
              generatingSubsIds={generatingSubsIds}
              onReviewSubs={setReviewVideoId}
              onResetDub={(videoId) => {
                setResetDubError(null);
                const video = videos?.find((v) => v.id === videoId) ?? null;
                setConfirmResetVideo(video);
              }}
              resettingDubIds={resettingDubIds}
              liveTaskStatuses={taskStatusByVideoId}
              activeTasks={activeTasks}
              targetLanguage={project.target_language}
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
            firstVideoPath={videos?.find((v) => !isFinalized(v))?.path}
          />
        )}

        {projectId && (
          <SubtitleReview
            isOpen={reviewVideoId !== null}
            onClose={() => setReviewVideoId(null)}
            filename={videos?.find((v) => v.id === reviewVideoId)?.filename}
            data={subtitleReview.data}
            loading={subtitleReview.loading}
            error={subtitleReview.error}
            selectedSrtPath={reviewSrtPath}
            onSrtPathChange={setReviewSrtPath}
            onSaveCue={subtitleReview.saveCue}
            savingCueIndex={subtitleReview.savingCueIndex}
            onDeleteCue={subtitleReview.deleteCue}
            deletingCueIndex={subtitleReview.deletingCueIndex}
          />
        )}
      </div>
    </div>
  );
};
