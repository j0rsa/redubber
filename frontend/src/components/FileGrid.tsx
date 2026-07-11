import { type ChangeEvent } from 'react';
import type { VideoFile, TaskStatus } from '../types';
import { PipelineStatus } from './PipelineStatus';
import styles from './FileGrid.module.css';

export interface FileGridProps {
  videos: VideoFile[];
  selectedIds: Set<number>;
  onSelectionChange: (ids: Set<number>) => void;
  /** Maps videoId → taskId for in-flight jobs. */
  runningJobIds?: Map<number, string>;
  /** Called when the user clicks "▶ View Job" for a row that has no running job yet (single-file submit). */
  onRedubSingle?: (videoPath: string) => void;
  /** Called when the user clicks "Replace Original" after the pipeline completes. */
  onFinalize?: (videoId: number) => void;
  /** Maps videoId → true while finalize is in progress. */
  finalizingIds?: Set<number>;
  /** Called when the user clicks "Generate Subs" to regenerate subtitles from existing segments. */
  onGenerateSubs?: (videoId: number) => void;
  /** Maps videoId → true while sub generation is in progress. */
  generatingSubsIds?: Set<number>;
  /** Live task statuses keyed by videoId — used to show real-time progress while a job runs. */
  liveTaskStatuses?: Map<number, TaskStatus>;
  /** All active tasks — used to detect queued videos not yet in liveTaskStatuses. */
  activeTasks?: TaskStatus[];
}

export const FileGrid = ({
  videos,
  selectedIds,
  onSelectionChange,
  runningJobIds,
  onRedubSingle,
  onFinalize,
  finalizingIds,
  onGenerateSubs,
  generatingSubsIds,
  liveTaskStatuses,
  activeTasks = [],
}: FileGridProps) => {
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatSize = (mb: number) => {
    return mb >= 1000 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(1)} MB`;
  };

  const selectableVideos = videos.filter((v) => !v.pipeline_status?.replaced);
  const allSelected = selectableVideos.length > 0 && selectableVideos.every((v) => selectedIds.has(v.id));
  const someSelected = selectableVideos.some((v) => selectedIds.has(v.id)) && !allSelected;

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
          {videos.map((video) => {
            const isRunning = runningJobIds?.has(video.id) ?? false;
            const taskId = runningJobIds?.get(video.id);
            const isSelected = selectedIds.has(video.id);
            const isReplaced = video.pipeline_status?.replaced ?? false;
            const isReadyToReplace = (video.pipeline_status?.is_complete ?? false) && !isReplaced;
            const isComplete = isReplaced; // "done done" — disable selection
            const liveTask = liveTaskStatuses?.get(video.id);
            // Build displayStatus by layering three sources (later overrides earlier):
            // 1. disk-based pipeline_status  — baseline counters from completed stages
            // 2. live task counters          — up-to-date values while the task runs
            // 3. live task status fields     — progress, stage, completion, errors
            const displayStatus = liveTask
              ? {
                  ...(video.pipeline_status ?? {}),
                  // Live counters from task queue (override stale disk values)
                  ...(liveTask.audio_chunks    != null && { audio_chunks: liveTask.audio_chunks }),
                  ...(liveTask.transcripts     != null && { transcripts: liveTask.transcripts }),
                  ...(liveTask.tts_segments    != null && { tts_segments: liveTask.tts_segments }),
                  ...(liveTask.tts_total       != null && { tts_total: liveTask.tts_total }),
                  ...(liveTask.subtitles       != null && { subtitles: liveTask.subtitles }),
                  ...(liveTask.audio_assembled != null && { audio_assembled: liveTask.audio_assembled }),
                  ...(liveTask.audio_assembled_total != null && { audio_assembled_total: liveTask.audio_assembled_total }),
                  ...(liveTask.video_mixed     != null && { video_mixed: liveTask.video_mixed }),
                  // Live status overrides
                  progress: liveTask.progress,
                  current_stage: liveTask.stage || 'Running',
                  is_complete: liveTask.status === 'completed',
                  failed: liveTask.status === 'failed',
                  error: liveTask.error,
                  replaced: video.pipeline_status?.replaced ?? false,
                }
              : (liveTask === undefined && !video.pipeline_status && activeTasks.some(t => t.video_path === video.path))
                ? { progress: 0, current_stage: 'Queued', is_complete: false, failed: false, error: undefined, replaced: false }
                : video.pipeline_status;

            return (
              <tr
                key={video.id}
                className={`${styles.row} ${isSelected ? styles.rowSelected : ''} ${isComplete ? styles.rowComplete : ''}`}
              >
                <td className={styles.checkboxCell}>
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={(e) => handleRowSelect(video.id, e.target.checked)}
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
                  ) : onRedubSingle ? (
                    <button
                      onClick={() => onRedubSingle(video.path)}
                      className={styles.actionButton}
                    >
                      Redub
                    </button>
                  ) : null}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};
