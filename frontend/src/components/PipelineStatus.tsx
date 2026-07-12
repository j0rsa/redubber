import type { PipelineStatus as PipelineStatusType } from '../types';
import styles from './PipelineStatus.module.css';

interface PipelineStatusProps {
  status: PipelineStatusType;
}

export const PipelineStatus = ({ status }: PipelineStatusProps) => {
  const getStatusClass = () => {
    if (status.failed) return styles.failed;
    if (status.is_complete) return styles.completed;
    if (status.replacement_status === 'pending') return styles.ready;
    if (status.progress > 0) return styles.running;
    return styles.pending;
  };

  const getStatusLabel = () => {
    if (status.failed) return 'Failed';
    if (status.is_complete) return 'Complete';
    if (status.replacement_status === 'pending') return 'Ready';
    if (status.current_stage === 'Queued') return 'Queued';
    if (status.progress > 0) return 'Ongoing';
    return 'Pending';
  };

  const renderCounter = (
    emoji: string,
    value: number | boolean | undefined,
    total?: number,
    inProgress?: boolean
  ) => {
    if (value === undefined || value === null) {
      return null;
    }

    if (typeof value === 'boolean') {
      return (
        <span className={styles.stat}>
          <span className={styles.statIcon}>{emoji}</span>
          <span className={styles.statValue}>{value ? '✓' : '—'}</span>
        </span>
      );
    }

    const progressClass = inProgress ? styles.statProgress : styles.stat;
    const displayValue = total && value < total ? `${value}/${total}` : value;

    return (
      <span className={progressClass}>
        <span className={styles.statIcon}>{emoji}</span>
        <span className={styles.statValue}>{displayValue}</span>
      </span>
    );
  };

  return (
    <div className={styles.container}>
      <div className={styles.statusHeader}>
        <span className={`${styles.stageChip} ${getStatusClass()}`}>
          {getStatusLabel()}
        </span>
        {status.current_stage && !status.failed && (
          <span className={styles.stageName}>{status.current_stage}</span>
        )}
        {status.failed && status.error && (
          <span className={styles.errorText} title={status.error}>
            {status.error.length > 80 ? status.error.slice(0, 80) + '…' : status.error}
          </span>
        )}
      </div>

      <div className={styles.progressSection}>
        <div className={styles.progressBar}>
          <div
            className={styles.progressFill}
            style={{ width: `${status.progress}%` }}
          />
        </div>
        <span className={styles.percentage}>{status.progress.toFixed(0)}%</span>
      </div>

      <div className={styles.stats}>
        {/* Stage 1: Extract */}
        {renderCounter('✂️', status.audio_chunks)}

        {/* Stage 2: Transcribe */}
        {renderCounter('📝', status.transcripts)}

        {/* Stage 3: Subtitles */}
        {renderCounter('📑', status.subtitles)}

        {/* Stage 4: TTS (with progress) */}
        {status.tts_segments !== undefined &&
          renderCounter(
            '🎙️',
            status.tts_segments,
            status.tts_total,
            status.tts_segments < (status.tts_total || Infinity)
          )}

        {/* Stage 6: Audio assembly (with progress) */}
        {status.audio_assembled !== undefined &&
          renderCounter(
            '🎵',
            status.audio_assembled,
            status.audio_assembled_total,
            status.audio_assembled < (status.audio_assembled_total || Infinity)
          )}

        {/* Stage 7: Video mixed */}
        {renderCounter('🎬', status.video_mixed)}

        {/* Stage 8: Validation */}
        {renderCounter('✅', status.output_validated)}

        {/* Stage 8: Backup */}
        {renderCounter('💾', status.backup_created)}
      </div>
    </div>
  );
};
