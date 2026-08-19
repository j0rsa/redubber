import { type ChangeEvent, Fragment, useMemo } from 'react';
import type { VideoFile, TaskStatus } from '../types';
import { PipelineStatus } from './PipelineStatus';
import { formatDuration, formatSize } from '../utils/format';
import { isVideoInTargetState } from '../utils/language';
import { formatFolderLabel, groupVideosByFolder } from '../utils/groupVideosByFolder';
import styles from './FileGrid.module.css';

export interface FileGridProps {
  videos: VideoFile[];
  selectedIds: Set<number>;
  onSelectionChange: (ids: Set<number>) => void;
  /** Project root path — used to group videos by relative subfolder. */
  projectPath?: string;
  /** Maps videoId → taskId for in-flight jobs. */
  runningJobIds?: Map<number, string>;
  /** Called when the user clicks "Redub" on a row that has not failed. */
  onRedubSingle?: (videoPath: string) => void;
  /** Called when the user clicks "Retry" on a failed redub row. */
  onRetryFailed?: (video: VideoFile) => void;
  /** Called when the user clicks "Retry reset redub" on a failed reset-dub row. */
  onRetryResetDub?: (video: VideoFile) => void;
  /** Called when the user clicks "Replace Original" after the pipeline completes. */
  onFinalize?: (videoId: number) => void;
  /** Maps videoId → true while finalize is in progress. */
  finalizingIds?: Set<number>;
  /** Called when the user clicks "Generate Subs" to regenerate subtitles from existing segments. */
  onGenerateSubs?: (videoId: number) => void;
  /** Maps videoId → true while sub generation is in progress. */
  generatingSubsIds?: Set<number>;
  /** Open the generated-subtitle review screen for a video. */
  onReviewSubs?: (videoId: number) => void;
  /** Called when the user clicks "Reset redub" on a finalized video. */
  onResetDub?: (videoId: number) => void;
  /** Maps videoId → true while dub reset is in progress. */
  resettingDubIds?: Set<number>;
  /** Live task statuses keyed by videoId — used to show real-time progress while a job runs. */
  liveTaskStatuses?: Map<number, TaskStatus>;
  /** All active tasks — used to detect queued videos not yet in liveTaskStatuses. */
  activeTasks?: TaskStatus[];
  /** Project target language (ISO 639-2), used to detect finalized dubs. */
  targetLanguage?: string;
}

interface VideoRowProps {
  video: VideoFile;
  isSelected: boolean;
  isComplete: boolean;
  isRunning: boolean;
  taskId?: string;
  displayStatus: VideoFile['pipeline_status'];
  isReadyToReplace: boolean;
  isReplaced: boolean;
  isFailedResetDub: boolean;
  isFailedRedub: boolean;
  canReviewSubs: boolean;
  onRowSelect: (id: number, checked: boolean) => void;
  onRedubSingle?: (videoPath: string) => void;
  onRetryFailed?: (video: VideoFile) => void;
  onRetryResetDub?: (video: VideoFile) => void;
  onFinalize?: (videoId: number) => void;
  finalizingIds?: Set<number>;
  onGenerateSubs?: (videoId: number) => void;
  generatingSubsIds?: Set<number>;
  onReviewSubs?: (videoId: number) => void;
  onResetDub?: (videoId: number) => void;
  resettingDubIds?: Set<number>;
}

const VideoRow = ({
  video,
  isSelected,
  isComplete,
  isRunning,
  taskId,
  displayStatus,
  isReadyToReplace,
  isReplaced,
  isFailedResetDub,
  isFailedRedub,
  canReviewSubs,
  onRowSelect,
  onRedubSingle,
  onRetryFailed,
  onRetryResetDub,
  onFinalize,
  finalizingIds,
  onGenerateSubs,
  generatingSubsIds,
  onReviewSubs,
  onResetDub,
  resettingDubIds,
}: VideoRowProps) => (
  <tr
    className={`${styles.row} ${isSelected ? styles.rowSelected : ''} ${isComplete ? styles.rowComplete : ''}`}
  >
    <td className={styles.checkboxCell}>
      <input
        type="checkbox"
        checked={isSelected}
        onChange={(e) => onRowSelect(video.id, e.target.checked)}
        aria-label={`Select ${video.filename}`}
        disabled={isComplete}
      />
    </td>
    <td className={styles.cell} data-label="Filename">
      <span className={styles.filename}>
        {isRunning && <span className={styles.runningDot} aria-label="Job running" />}
        {video.filename}
      </span>
    </td>
    <td className={styles.cell} data-label="Duration">
      <span className={styles.duration}>{formatDuration(video.duration_seconds)}</span>
    </td>
    <td className={styles.cell} data-label="Size">
      <span className={styles.size}>{formatSize(video.size_mb)}</span>
    </td>
    <td className={styles.cell} data-label="Audio">
      {video.audio_streams.map((stream) => (
        <div key={stream.index} className={styles.audioStream}>
          <span className={styles.badge}>{stream.language}</span>
          <span style={{ color: '#757575', fontSize: '12px' }}>
            {stream.codec}
          </span>
        </div>
      ))}
    </td>
    <td className={styles.cell} data-label="Subtitles">
      {video.subtitles.map((sub, idx) => (
        <div key={idx} className={styles.subtitle}>
          <span className={styles.badge}>{sub.language}</span>
          <span style={{ color: '#757575', fontSize: '12px' }}>
            {sub.embedded ? 'embedded' : 'external'}
          </span>
        </div>
      ))}
    </td>
    <td className={styles.cell} data-label="Status">
      {displayStatus ? (
        <PipelineStatus status={displayStatus} />
      ) : (
        <span style={{ color: '#999' }}>Not started</span>
      )}
    </td>
    <td className={styles.cell} data-label="Actions">
      <div className={styles.actions}>
        {isRunning && taskId ? (
          <a href={`/job/${taskId}`} className={styles.viewJobLink}>
            ▶ View Job
          </a>
        ) : isReadyToReplace && onFinalize ? (
          <button
            onClick={() => onFinalize(video.id)}
            className={styles.finalizeButton}
            disabled={finalizingIds?.has(video.id)}
          >
            {finalizingIds?.has(video.id) ? 'Replacing…' : '🔁 Replace Original'}
          </button>
        ) : video.pipeline_status?.current_stage === 'Gen Subtitles' && onGenerateSubs ? (
          <button
            onClick={() => onGenerateSubs(video.id)}
            className={styles.actionButton}
            disabled={generatingSubsIds?.has(video.id)}
          >
            {generatingSubsIds?.has(video.id) ? 'Generating…' : '📝 Generate Subs'}
          </button>
        ) : isReplaced && onResetDub ? (
          <button
            onClick={() => onResetDub(video.id)}
            className={styles.resetDubButton}
            disabled={resettingDubIds?.has(video.id)}
            title="Reset the redub to an earlier pipeline step"
          >
            {resettingDubIds?.has(video.id) ? 'Resetting…' : 'Reset redub'}
          </button>
        ) : isFailedResetDub && onRetryResetDub ? (
          <button
            type="button"
            onClick={() => onRetryResetDub(video)}
            className={styles.retryButton}
            disabled={resettingDubIds?.has(video.id)}
          >
            {resettingDubIds?.has(video.id) ? 'Retrying…' : 'Retry reset redub'}
          </button>
        ) : isFailedRedub && onRetryFailed ? (
          <button
            type="button"
            onClick={() => onRetryFailed(video)}
            className={styles.retryButton}
          >
            Retry
          </button>
        ) : onRedubSingle ? (
          <button
            onClick={() => onRedubSingle(video.path)}
            className={styles.actionButton}
          >
            Redub
          </button>
        ) : null}
        {canReviewSubs && (
          <button
            type="button"
            className={styles.reviewButton}
            onClick={() => onReviewSubs?.(video.id)}
          >
            Review subs
          </button>
        )}
      </div>
    </td>
  </tr>
);

export const FileGrid = ({
  videos,
  selectedIds,
  onSelectionChange,
  projectPath,
  runningJobIds,
  onRedubSingle,
  onRetryFailed,
  onRetryResetDub,
  onFinalize,
  finalizingIds,
  onGenerateSubs,
  generatingSubsIds,
  onReviewSubs,
  onResetDub,
  resettingDubIds,
  liveTaskStatuses,
  activeTasks = [],
  targetLanguage = '',
}: FileGridProps) => {
  const selectableVideos = videos.filter((v) => !v.pipeline_status?.replaced);
  const allSelected = selectableVideos.length > 0 && selectableVideos.every((v) => selectedIds.has(v.id));
  const someSelected = selectableVideos.some((v) => selectedIds.has(v.id)) && !allSelected;

  const totalDuration = videos.reduce((sum, v) => sum + (v.duration_seconds || 0), 0);
  const totalSize = videos.reduce((sum, v) => sum + (v.size_mb || 0), 0);

  const folderGroups = useMemo(
    () => groupVideosByFolder(videos, projectPath),
    [videos, projectPath],
  );

  const showFolderHeaders = folderGroups.length > 1
    || (folderGroups.length === 1 && folderGroups[0]?.folder !== '.');

  const handleSelectAll = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.checked) {
      onSelectionChange(new Set(selectableVideos.map((v) => v.id)));
    } else {
      onSelectionChange(new Set());
    }
  };

  const handleRowSelect = (id: number, checked: boolean) => {
    const next = new Set(selectedIds);
    if (checked) {
      next.add(id);
    } else {
      next.delete(id);
    }
    onSelectionChange(next);
  };

  const renderVideoRow = (video: VideoFile) => {
    const liveTask = liveTaskStatuses?.get(video.id);
    const isRunning = (runningJobIds?.has(video.id) ?? false)
      && liveTask?.status !== 'failed';
    const taskId = runningJobIds?.get(video.id);
    const isSelected = selectedIds.has(video.id);
    const isReplaced = (video.pipeline_status?.replaced ?? false)
      || isVideoInTargetState(video.audio_streams, video.subtitles, targetLanguage);
    const isReadyToReplace = (video.pipeline_status?.is_complete ?? false) && !isReplaced;
    const isComplete = isReplaced;
    const canReviewSubs =
      Boolean(onReviewSubs) &&
      (
        video.subtitles.length > 0
        || (video.pipeline_status?.subtitles ?? 0) > 0
        || (video.pipeline_status?.transcripts ?? 0) > 0
      );
    const displayStatus = liveTask
      ? {
          ...(video.pipeline_status ?? {}),
          ...(liveTask.audio_chunks != null && { audio_chunks: liveTask.audio_chunks }),
          ...(liveTask.transcripts != null && { transcripts: liveTask.transcripts }),
          ...(liveTask.tts_segments != null && { tts_segments: liveTask.tts_segments }),
          ...(liveTask.tts_total != null && { tts_total: liveTask.tts_total }),
          ...(liveTask.subtitles != null && { subtitles: liveTask.subtitles }),
          ...(liveTask.audio_assembled != null && { audio_assembled: liveTask.audio_assembled }),
          ...(liveTask.audio_assembled_total != null && { audio_assembled_total: liveTask.audio_assembled_total }),
          ...(liveTask.video_mixed != null && { video_mixed: liveTask.video_mixed }),
          progress: liveTask.progress,
          current_stage: liveTask.stage || 'Running',
          is_complete: liveTask.status === 'completed',
          failed: liveTask.status === 'failed',
          error: liveTask.error,
          replaced: isReplaced,
        }
      : (liveTask === undefined && !video.pipeline_status && activeTasks.some(t => t.video_path === video.path && (t.status === 'queued' || t.status === 'running')))
        ? { progress: 0, current_stage: 'Queued', is_complete: false, failed: false, error: undefined, replaced: false }
        : video.pipeline_status;

    const isFailed = Boolean(displayStatus?.failed) && !isRunning;
    const failedTaskType = liveTask?.task_type;
    const isFailedResetDub =
      isFailed
      && (
        failedTaskType === 'reset_dub'
        || (failedTaskType === undefined && isReplaced)
      );
    const isFailedRedub = isFailed && !isFailedResetDub;

    return (
      <VideoRow
        key={video.id}
        video={video}
        isSelected={isSelected}
        isComplete={isComplete}
        isRunning={isRunning}
        taskId={taskId}
        displayStatus={displayStatus}
        isReadyToReplace={isReadyToReplace}
        isReplaced={isReplaced}
        isFailedResetDub={isFailedResetDub}
        isFailedRedub={isFailedRedub}
        canReviewSubs={canReviewSubs}
        onRowSelect={handleRowSelect}
        onRedubSingle={onRedubSingle}
        onRetryFailed={onRetryFailed}
        onRetryResetDub={onRetryResetDub}
        onFinalize={onFinalize}
        finalizingIds={finalizingIds}
        onGenerateSubs={onGenerateSubs}
        generatingSubsIds={generatingSubsIds}
        onReviewSubs={onReviewSubs}
        onResetDub={onResetDub}
        resettingDubIds={resettingDubIds}
      />
    );
  };

  return (
    <div className={styles.fileGrid}>
      <table className={styles.table}>
        <thead className={styles.header}>
          <tr>
            <th className={styles.checkboxCell}>
              <input
                type="checkbox"
                checked={allSelected}
                ref={(el) => {
                  if (el) el.indeterminate = someSelected;
                }}
                onChange={handleSelectAll}
                aria-label="Select all videos"
                disabled={videos.length === 0}
              />
            </th>
            <th>Filename</th>
            <th>Duration</th>
            <th>Size</th>
            <th>Audio Streams</th>
            <th>Subtitles</th>
            <th>Pipeline Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {folderGroups.map((group) => (
            <Fragment key={group.folder}>
              {showFolderHeaders && (
                <tr className={styles.folderHeaderRow}>
                  <td className={styles.folderHeaderCell} colSpan={8}>
                    {formatFolderLabel(group.folder)}
                  </td>
                </tr>
              )}
              {group.videos.map((video) => renderVideoRow(video))}
            </Fragment>
          ))}
        </tbody>
        {videos.length > 0 && (
          <tfoot>
            <tr className={styles.totalsRow}>
              <td className={styles.checkboxCell} />
              <td className={styles.cell} data-label="Filename">
                <span className={styles.totalsLabel}>Total</span>
              </td>
              <td className={styles.cell} data-label="Duration">
                <span className={styles.duration}>{formatDuration(totalDuration)}</span>
              </td>
              <td className={styles.cell} data-label="Size">
                <span className={styles.size}>{formatSize(totalSize)}</span>
              </td>
              <td className={styles.cell} colSpan={4} aria-hidden="true" />
            </tr>
          </tfoot>
        )}
      </table>
    </div>
  );
};
