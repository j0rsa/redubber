import { useMemo, useState } from 'react';
import { RESET_STAGES, stageIndex, type ResetStageId, type ResetToStageId } from './stages';
import styles from './ResetDubDialog.module.css';

function artifactsSummary(stageId: string, keepSubs = false): { deleted: string[]; kept: string[] } {
  const deleted: string[] = ['Dubbed video track'];
  const kept: string[] = ['Original video', 'External subtitle sidecars'];

  if (stageId === 'start') {
    deleted.push('Extracted audio chunks', 'Transcripts & STT cache', 'TTS audio segments', 'Mixed audio track');
    if (keepSubs) kept.push('Generated subtitles');
    else deleted.push('Generated subtitles');
  } else if (stageId === 'audio') {
    kept.push('Extracted audio chunks');
    deleted.push('Transcripts & STT cache', 'TTS audio segments', 'Mixed audio track');
    kept.push('Generated subtitles');
  } else if (stageId === 'stt') {
    kept.push('Extracted audio chunks', 'Transcripts & STT cache', 'Generated subtitles');
    deleted.push('TTS audio segments', 'Mixed audio track');
  } else if (stageId === 'subtitles') {
    kept.push('Extracted audio chunks', 'Transcripts & STT cache', 'Generated subtitles');
    deleted.push('TTS audio segments', 'Mixed audio track');
  } else if (stageId === 'tts') {
    kept.push('Extracted audio chunks', 'Transcripts & STT cache', 'Generated subtitles', 'TTS audio segments');
    deleted.push('Mixed audio track');
  } else {
    kept.push('Extracted audio chunks', 'Transcripts & STT cache', 'Generated subtitles', 'TTS audio segments', 'Mixed audio track');
  }

  return { deleted, kept };
}

export interface ResetDubDialogProps {
  videoFilename: string;
  selectionCount?: number;
  currentStage?: ResetStageId;
  isSubmitting?: boolean;
  errorMessage?: string | null;
  onCancel: () => void;
  onConfirm: (resetTo: ResetToStageId, keepSubtitles: boolean) => void;
}

export const ResetDubDialog = ({
  videoFilename,
  selectionCount,
  currentStage = 'complete',
  isSubmitting = false,
  errorMessage,
  onCancel,
  onConfirm,
}: ResetDubDialogProps) => {
  const currentIndex = Math.max(0, stageIndex(currentStage));
  const [selectedIndex, setSelectedIndex] = useState(currentIndex);
  const [keepSubtitles, setKeepSubtitles] = useState(false);

  const selected = RESET_STAGES[selectedIndex];
  const canSubmit = selectedIndex < currentIndex && selected.id !== 'complete';
  const showKeepSubs = selected.id === 'start';

  const artifacts = useMemo(
    () => artifactsSummary(selected.id, keepSubtitles),
    [selected.id, keepSubtitles],
  );

  const handleSubmit = () => {
    if (!canSubmit) return;
    onConfirm(selected.id, keepSubtitles);
  };

  return (
    <div className={styles.overlay} role="dialog" aria-modal="true" aria-labelledby="reset-dub-title">
      <div className={styles.dialog}>
        <h2 id="reset-dub-title" className={styles.title}>Reset redub</h2>
        <p className={styles.body}>
          Move back from the current step to choose how far to rewind{' '}
          <strong>
            {selectionCount
              ? `${selectionCount} selected video${selectionCount === 1 ? '' : 's'}`
              : videoFilename}
          </strong>
          . {selectionCount == null || selectionCount === 1
            ? 'The video file is'
            : 'Each video file is'} always reverted to the original-language track.
        </p>

        {errorMessage && (
          <div className={styles.errorBox} role="alert">
            {errorMessage}
          </div>
        )}

        <div className={styles.fieldLabel}>
          Reset to
        </div>
        <div
          className={styles.sliderWrap}
          style={{ ['--node-count' as string]: currentIndex + 1 }}
        >
          <div className={styles.track} aria-hidden="true">
            <div
              className={styles.trackFill}
              style={{
                width: currentIndex === 0
                  ? '0%'
                  : `${(selectedIndex / currentIndex) * 100}%`,
              }}
            />
          </div>
          <div className={styles.nodes}>
            {RESET_STAGES.slice(0, currentIndex + 1).map((stage, index) => {
              const isCurrent = index === currentIndex;
              const isSelected = index === selectedIndex;
              const isKept = index <= selectedIndex;
              const isDeleted = index > selectedIndex && index <= currentIndex;
              return (
                <button
                  key={stage.id}
                  type="button"
                  className={[
                    styles.node,
                    isSelected ? styles.nodeSelected : '',
                    isKept ? styles.nodeKept : '',
                    isDeleted ? styles.nodeDeleted : '',
                    isCurrent ? styles.nodeCurrent : '',
                  ].join(' ')}
                  disabled={isSubmitting}
                  onClick={() => setSelectedIndex(index)}
                  aria-label={stage.label}
                  aria-current={isSelected ? 'step' : undefined}
                >
                  <span className={styles.dot} />
                  <span className={styles.nodeLabel}>{stage.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {showKeepSubs && (
          <label className={styles.keepSubsToggle}>
            <input
              type="checkbox"
              checked={keepSubtitles}
              onChange={(e) => setKeepSubtitles(e.target.checked)}
              disabled={isSubmitting}
            />
            Keep generated subtitles
          </label>
        )}

        <div className={styles.artifacts}>
          <ul className={styles.artifactDeleted}>
            {artifacts.deleted.map((item) => <li key={item}>{item}</li>)}
          </ul>
          {artifacts.kept.length > 0 && (
            <ul className={styles.artifactKept}>
              {artifacts.kept.map((item) => <li key={item}>{item}</li>)}
            </ul>
          )}
        </div>

        <div className={styles.actions}>
          <button
            type="button"
            className={styles.cancelButton}
            onClick={onCancel}
            disabled={isSubmitting}
          >
            Cancel
          </button>
          <button
            type="button"
            className={styles.confirmButton}
            onClick={handleSubmit}
            disabled={isSubmitting || !canSubmit}
          >
            {isSubmitting ? 'Resetting…' : `Reset to ${selected.label}`}
          </button>
        </div>
      </div>
    </div>
  );
};
