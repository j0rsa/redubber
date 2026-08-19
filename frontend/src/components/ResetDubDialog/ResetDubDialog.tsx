import { useMemo, useState } from 'react';
import { RESET_STAGES, stageIndex, type ResetStageId, type ResetToStageId } from './stages';
import styles from './ResetDubDialog.module.css';

export interface ResetDubDialogProps {
  videoFilename: string;
  currentStage?: ResetStageId;
  isSubmitting?: boolean;
  errorMessage?: string | null;
  onCancel: () => void;
  onConfirm: (resetTo: ResetToStageId) => void;
}

export const ResetDubDialog = ({
  videoFilename,
  currentStage = 'complete',
  isSubmitting = false,
  errorMessage,
  onCancel,
  onConfirm,
}: ResetDubDialogProps) => {
  const currentIndex = Math.max(0, stageIndex(currentStage));
  const [selectedIndex, setSelectedIndex] = useState(currentIndex);

  const selected = RESET_STAGES[selectedIndex];
  const canSubmit = selectedIndex < currentIndex && selected.id !== 'complete';
  const deletesSubs = selected.id === 'start';

  const hint = useMemo(() => {
    if (deletesSubs) {
      return 'The dubbed audio track is removed and generated subtitles are deleted. The next redub starts from the beginning.';
    }
    return `The dubbed audio track is always removed. Subtitles are kept. Later artefacts after ${selected.label} are cleared.`;
  }, [deletesSubs, selected.label]);

  const handleSubmit = () => {
    if (!canSubmit) return;
    onConfirm(selected.id);
  };

  return (
    <div className={styles.overlay} role="dialog" aria-modal="true" aria-labelledby="reset-dub-title">
      <div className={styles.dialog}>
        <h2 id="reset-dub-title" className={styles.title}>Reset redub</h2>
        <p className={styles.body}>
          Move back from the current step to choose how far to rewind{' '}
          <strong>{videoFilename}</strong>. The video file is always reverted to the
          original-language track.
        </p>

        {errorMessage && (
          <div className={styles.errorBox} role="alert">
            {errorMessage}
          </div>
        )}

        <label className={styles.fieldLabel} htmlFor="reset-stage-slider">
          Reset to
        </label>
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
          <input
            id="reset-stage-slider"
            className={styles.slider}
            type="range"
            min={0}
            max={currentIndex}
            step={1}
            value={selectedIndex}
            disabled={isSubmitting || currentIndex === 0}
            aria-valuetext={selected.label}
            onChange={(event) => setSelectedIndex(Number(event.target.value))}
          />
          <div className={styles.nodes}>
            {RESET_STAGES.slice(0, currentIndex + 1).map((stage, index) => {
              const isCurrent = index === currentIndex;
              const isSelected = index === selectedIndex;
              const isKept = index <= selectedIndex;
              return (
                <button
                  key={stage.id}
                  type="button"
                  className={[
                    styles.node,
                    isSelected ? styles.nodeSelected : '',
                    isKept ? styles.nodeKept : '',
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

        <p className={styles.hint}>{hint}</p>

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
