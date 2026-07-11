import { useParams, useNavigate } from 'react-router-dom';
import { useRef, useEffect } from 'react';
import { useTask, useCancelTask } from '../hooks/useTasks';
import { useNotifications } from '../hooks/useNotifications';
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
}

export const JobMonitorView = ({
  task,
  isLoading,
  isCanceling,
  cancelError,
  onBack,
  onCancel,
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
  };
  const statusLabel: Record<string, string> = {
    completed: 'Completed', failed: 'Failed', running: 'Running', queued: 'Queued',
  };

  const pipelineStatus = {
    progress: task.progress,
    current_stage: task.stage,
    is_complete: task.status === 'completed',
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

          {task.status === 'running' && (
            <PipelineStatus status={pipelineStatus} />
          )}

          {task.error && (
            <div className={styles.errorBox}>
              <p className={styles.errorBoxTitle}>Error</p>
              <p className={styles.errorBoxBody}>{task.error}</p>
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

          {cancelError && (
            <div className={styles.errorBox}>
              Failed to cancel: {cancelError}
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

  return (
    <JobMonitorView
      task={task}
      isLoading={isLoading}
      isCanceling={cancelTask.isPending}
      cancelError={cancelTask.isError ? (cancelTask.error as Error).message : null}
      onBack={() => navigate(-1)}
      onCancel={handleCancel}
    />
  );
};
