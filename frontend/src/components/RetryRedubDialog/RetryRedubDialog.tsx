import { useState } from 'react';
import styles from './RetryRedubDialog.module.css';

const MIN_CHUNK_SECONDS = 60;
const MAX_CHUNK_SECONDS = 3600;

export interface RetryRedubDialogProps {
  videoFilename: string;
  errorMessage?: string;
  defaultChunkDuration: number;
  isSubmitting?: boolean;
  onCancel: () => void;
  onConfirm: (audioChunkDuration: number) => void;
}

export const RetryRedubDialog = ({
  videoFilename,
  errorMessage,
  defaultChunkDuration,
  isSubmitting = false,
  onCancel,
  onConfirm,
}: RetryRedubDialogProps) => {
  const [chunkDuration, setChunkDuration] = useState(defaultChunkDuration);

  const clampedValue = Number.isFinite(chunkDuration)
    ? Math.min(MAX_CHUNK_SECONDS, Math.max(MIN_CHUNK_SECONDS, chunkDuration))
    : defaultChunkDuration;

  const isValid =
    Number.isFinite(chunkDuration)
    && chunkDuration >= MIN_CHUNK_SECONDS
    && chunkDuration <= MAX_CHUNK_SECONDS;

  const handleSubmit = () => {
    if (!isValid) return;
    onConfirm(clampedValue);
  };

  return (
    <div className={styles.overlay} role="dialog" aria-modal="true" aria-labelledby="retry-redub-title">
      <div className={styles.dialog}>
        <h2 id="retry-redub-title" className={styles.title}>Retry redub</h2>
        <p className={styles.body}>
          Retry <strong>{videoFilename}</strong> with a different audio chunk size.
          Shorter chunks can help when transcription fails on long silent sections.
        </p>

        {errorMessage && (
          <div className={styles.errorBox} role="alert">
            {errorMessage}
          </div>
        )}

        <label className={styles.fieldLabel} htmlFor="retry-chunk-duration">
          Audio chunk duration
        </label>
        <p className={styles.hint}>
          Seconds of audio sent to Whisper per request (60–3600). Existing chunks and transcription artefacts will be cleared.
        </p>
        <div className={styles.inputRow}>
          <input
            id="retry-chunk-duration"
            type="number"
            className={styles.input}
            value={chunkDuration}
            min={MIN_CHUNK_SECONDS}
            max={MAX_CHUNK_SECONDS}
            disabled={isSubmitting}
            onChange={(e) => setChunkDuration(parseInt(e.target.value, 10))}
          />
          <span className={styles.inputSuffix}>s</span>
        </div>
        {!isValid && (
          <p className={styles.validationError}>
            Enter a value between {MIN_CHUNK_SECONDS} and {MAX_CHUNK_SECONDS} seconds.
          </p>
        )}

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
            disabled={isSubmitting || !isValid}
          >
            {isSubmitting ? 'Submitting…' : 'Retry redub'}
          </button>
        </div>
      </div>
    </div>
  );
};
