import { useParams, useNavigate } from 'react-router-dom';
import { useRef, useEffect, useState } from 'react';
import { useTask, useCancelTask, useSubmitRedub } from '../hooks/useTasks';
import { useNotifications } from '../hooks/useNotifications';
import { useSettings } from '../hooks/useSettings';
import { RetryRedubDialog } from '../components/RetryRedubDialog/RetryRedubDialog';
import { formatPipelineError } from '../utils/formatError';
import type { TaskStatus } from '../types';
import { PipelineStatus } from '../components/PipelineStatus';
import styles from './JobMonitor.module.css';

// ── Pure view ─────────────────────────────────────────────────────────────────

export interface JobMonitorViewProps {
  task: TaskStatus | null | undefined;
  isLoading: boolean;
  isCanceling: boolean;
  cancelError: string | null;
  onBack: () => void;
  onCancel: () => void;
  onRetry?: () => void;
  onResolve?: () => void;
  isRetrying?: boolean;
  retryError?: string | null;
}

export const JobMonitorView = ({
  task,
  isLoading,
  isCanceling,
  cancelError,
  onBack,
  onCancel,
  onRetry,
  onResolve,
  isRetrying = false,
  retryError = null,
}: JobMonitorViewProps) => {
  if (isLoading) {
    return (
      <div className={styles.centered}>
        <p className={styles.loadingText}>Loading task status…</p>
      </div>
    );
  }

  if (!task) {
    return (
      <div className={styles.centered}>
        <p className={styles.notFoundText}>Task not found</p>
        <button className={styles.backButton} onClick={onBack}>Back</button>
      </div>
    );
  }

  const badgeClass: Record<string, string> = {
    running:   styles.badgeRunning,
    queued:    styles.badgeQueued,
    completed: styles.badgeCompleted,
    failed:    styles.badgeFailed,
    awaiting_subtitle_review: styles.badgeQueued,
  };
  const statusLabel: Record<string, string> = {
    completed: 'Completed',
    failed: 'Failed',
    running: 'Running',
    queued: 'Queued',
    awaiting_subtitle_review: 'Review needed',
  };

  const pipelineStatus = {
    progress: task.progress,
    current_stage: task.stage,
    is_complete: task.status === 'completed',
    awaiting_subtitle_review: task.status === 'awaiting_subtitle_review',
  };

  return (
    <div className={styles.page}>
      <div className={styles.inner}>
        <div className={styles.header}>
          <button className={styles.backButton} onClick={onBack}>← Back</button>
          <h1 className={styles.title}>Task Monitor</h1>
          <p className={styles.taskId}>ID: {task.task_id}</p>
        </div>

        <div className={styles.card}>
          <div className={styles.statusRow}>
            <h2 className={styles.statusLabel}>Status</h2>
            <span className={`${styles.badge} ${badgeClass[task.status] ?? styles.badgeQueued}`}>
              {statusLabel[task.status] ?? task.status}
            </span>
          </div>

          <p className={styles.videoLabel}>Video</p>
          <p className={styles.videoPath}>{task.video_path}</p>

          {(task.status === 'running' || task.status === 'awaiting_subtitle_review') && (
            <PipelineStatus status={pipelineStatus} />
          )}

          {task.error && (
            <div className={styles.errorBox}>
              <p className={styles.errorBoxTitle}>Error</p>
              <p className={styles.errorBoxBody}>{formatPipelineError(task.error)}</p>
            </div>
          )}

          <div className={styles.timestamps}>
            <span>Created: {new Date(task.created_at).toLocaleString()}</span>
            {task.started_at && <span>Started: {new Date(task.started_at).toLocaleString()}</span>}
            {task.completed_at && <span>Completed: {new Date(task.completed_at).toLocaleString()}</span>}
          </div>

          {(task.status === 'queued' || task.status === 'running') && (
            <div className={styles.actions}>
              <button
                className={styles.cancelButton}
                onClick={onCancel}
                disabled={isCanceling}
              >
                {isCanceling ? 'Canceling…' : 'Cancel Task'}
              </button>
            </div>
          )}

          {task.status === 'failed' && onRetry && (
            <div className={styles.actions}>
              <button
                className={styles.retryButton}
                onClick={onRetry}
                disabled={isRetrying}
              >
                {isRetrying ? 'Retrying…' : 'Retry redub'}
              </button>
            </div>
          )}
          {task.status === 'awaiting_subtitle_review' && onResolve && (
            <div className={styles.actions}>
              <button className={styles.retryButton} onClick={onResolve}>
                Resolve subtitle warnings
              </button>
            </div>
          )}

          {cancelError && (
            <div className={styles.errorBox}>
              Failed to cancel: {cancelError}
            </div>
          )}

          {retryError && (
            <div className={styles.errorBox}>
              Failed to retry: {retryError}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ── Connected container ───────────────────────────────────────────────────────

export const JobMonitor = () => {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();
  const { data: task, isLoading } = useTask(taskId || null);
  const cancelTask = useCancelTask();
  const submitRedub = useSubmitRedub();
  const { settings } = useSettings();
  const [showRetryDialog, setShowRetryDialog] = useState(false);
  const { showNotification, requestPermission, permission } = useNotifications();
  const previousStatus = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (permission === 'default') requestPermission();
  }, [permission, requestPermission]);

  useEffect(() => {
    if (!task || !previousStatus.current) {
      if (task) previousStatus.current = task.status;
      return;
    }
    const wasRunning = previousStatus.current === 'running';
    if (wasRunning && task.status === 'completed') {
      showNotification('Redubbing Complete', {
        body: `"${task.video_path}" has been redubbed successfully`,
        icon: '/pwa-192x192.png', badge: '/pwa-192x192.png', tag: task.task_id,
      });
    } else if (wasRunning && task.status === 'failed') {
      showNotification('Redubbing Failed', {
        body: `"${task.video_path}" failed: ${task.error ?? 'Unknown error'}`,
        icon: '/pwa-192x192.png', badge: '/pwa-192x192.png', tag: task.task_id,
      });
    }
    previousStatus.current = task.status;
  }, [task?.status, task?.video_path, task?.error, task?.task_id, showNotification]);

  const handleCancel = async () => {
    if (!taskId) return;
    try { await cancelTask.mutateAsync(taskId); }
    catch (err) { console.error('Failed to cancel task:', err); }
  };

  const handleRetry = async (audioChunkDuration: number) => {
    if (!task?.video_path || task.project_id == null) return;
    try {
      const result = await submitRedub.mutateAsync({
        video_path: task.video_path,
        project_id: task.project_id,
        audio_chunk_duration: audioChunkDuration,
      });
      setShowRetryDialog(false);
      if (result?.task_id) {
        navigate(`/job/${result.task_id}`, { replace: true });
      }
    } catch (err) {
      console.error('Failed to retry redub:', err);
    }
  };

  return (
    <>
      <JobMonitorView
        task={task}
        isLoading={isLoading}
        isCanceling={cancelTask.isPending}
        cancelError={cancelTask.isError ? (cancelTask.error as Error).message : null}
        onBack={() => navigate(-1)}
        onCancel={handleCancel}
        onRetry={task?.project_id != null && task.status === 'failed' ? () => setShowRetryDialog(true) : undefined}
        onResolve={
          task?.project_id != null
          && task.status === 'awaiting_subtitle_review'
            ? () => navigate(`/project/${task.project_id}`)
            : undefined
        }
        isRetrying={submitRedub.isPending}
        retryError={submitRedub.isError ? (submitRedub.error as Error).message : null}
      />
      {showRetryDialog && task && (
        <RetryRedubDialog
          videoFilename={task.video_path.split('/').pop() ?? task.video_path}
          errorMessage={task.error ?? undefined}
          defaultChunkDuration={settings.audio_chunk_duration}
          isSubmitting={submitRedub.isPending}
          onCancel={() => setShowRetryDialog(false)}
          onConfirm={(audioChunkDuration) => void handleRetry(audioChunkDuration)}
        />
      )}
    </>
  );
};
