import { basename } from './utils';
import type { TaskStatus } from '../../types';
import styles from './TasksPanel.module.css';

// ── TasksIndicator ─────────────────────────────────────────────────────────

export interface TasksIndicatorProps {
  activeCount: number;
  isOpen: boolean;
  onClick: () => void;
}

export const TasksIndicator = ({ activeCount, isOpen, onClick }: TasksIndicatorProps) => {
  return (
    <button
      className={[
        styles.indicator,
        activeCount === 0 && !isOpen ? styles.indicatorIdle : '',
        activeCount > 0 ? styles.indicatorPulsing : '',
      ].join(' ').trim()}
      onClick={onClick}
      aria-label={activeCount > 0 ? `${activeCount} active task${activeCount > 1 ? 's' : ''}` : 'Tasks'}
      title={activeCount > 0 ? `${activeCount} active task${activeCount > 1 ? 's' : ''}` : 'Tasks'}
    >
      {/* Queue icon: three stacked lines */}
      <span className={styles.indicatorIcon} aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
          <line x1="4" y1="7"  x2="20" y2="7" />
          <line x1="4" y1="12" x2="20" y2="12" />
          <line x1="4" y1="17" x2="20" y2="17" />
        </svg>
      </span>
      {activeCount > 0 && (
        <span className={styles.badge} aria-label={`${activeCount} active tasks`}>
          {activeCount > 9 ? '9+' : activeCount}
        </span>
      )}
    </button>
  );
};

// ── TaskCard ───────────────────────────────────────────────────────────────

const statusIcon: Record<TaskStatus['status'], string> = {
  running: '🔄',
  queued: '⏳',
  completed: '✅',
  failed: '❌',
};

interface TaskCardProps {
  task: TaskStatus;
  onViewJob: (taskId: string) => void;
}

const TaskCard = ({ task, onViewJob }: TaskCardProps) => {
  const icon = statusIcon[task.status] ?? '🔄';
  const name = basename(task.video_path);
  const showProgress = task.status === 'running';

  return (
    <div className={styles.taskCard}>
      <div className={styles.taskTop}>
        <span className={styles.taskIcon} aria-hidden="true">{icon}</span>
        <span className={styles.taskName} title={task.video_path}>{name}</span>
      </div>

      <div className={styles.taskMeta}>
        <span className={styles.taskStage}>
          {task.stage || (task.status === 'queued' ? 'Queued' : task.status)}
        </span>
        <span className={styles.taskPercent}>{task.progress}%</span>
      </div>

      {showProgress && (
        <div className={styles.progressRow}>
          <div className={styles.progressTrack}>
            <div
              className={styles.progressFill}
              style={{ width: `${task.progress}%` }}
              role="progressbar"
              aria-valuenow={task.progress}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          </div>
        </div>
      )}

      {task.status === 'failed' && task.error && (
        <p className={styles.taskError} title={task.error}>
          {task.error.length > 120 ? task.error.slice(0, 120) + '…' : task.error}
        </p>
      )}

      {task.status === 'running' && (
        <button
          className={styles.viewDetails}
          onClick={() => onViewJob(task.task_id)}
        >
          View Details →
        </button>
      )}
    </div>
  );
};

// ── TasksPanel ─────────────────────────────────────────────────────────────

export interface TasksPanelProps {
  tasks: TaskStatus[];
  isOpen: boolean;
  onClose: () => void;
  onViewJob: (taskId: string) => void;
}

export const TasksPanel = ({ tasks, isOpen, onClose, onViewJob }: TasksPanelProps) => {
  if (!isOpen) return null;

  return (
    <>
      <div
        className={styles.backdrop}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside className={styles.panel} role="dialog" aria-label="Running Tasks" aria-modal="true">
        <header className={styles.header}>
          <h2 className={styles.title}>
            {tasks.length > 0 ? `Active Tasks (${tasks.length})` : 'Tasks'}
          </h2>
          <button
            className={styles.closeButton}
            onClick={onClose}
            aria-label="Close panel"
          >
            ✕
          </button>
        </header>

        <div className={styles.taskList}>
          {tasks.length === 0 ? (
            <p className={styles.emptyState}>No active tasks</p>
          ) : (
            tasks.map((task) => (
              <TaskCard key={task.task_id} task={task} onViewJob={onViewJob} />
            ))
          )}
        </div>
      </aside>
    </>
  );
};
