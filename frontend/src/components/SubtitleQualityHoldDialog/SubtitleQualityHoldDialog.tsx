import type { SubtitleQualityIssue } from '../../types';
import styles from './SubtitleQualityHoldDialog.module.css';

interface SubtitleQualityHoldDialogProps {
  videoFilename: string;
  issues: SubtitleQualityIssue[];
  isSubmitting?: boolean;
  onCancel: () => void;
  onRetry: () => void;
  onEdit: () => void;
  onIgnore: () => void;
}

export const SubtitleQualityHoldDialog = ({
  videoFilename,
  issues,
  isSubmitting = false,
  onCancel,
  onRetry,
  onEdit,
  onIgnore,
}: SubtitleQualityHoldDialogProps) => (
  <div
    className={styles.overlay}
    role="dialog"
    aria-modal="true"
    aria-labelledby="subtitle-quality-hold-title"
  >
    <div className={styles.dialog}>
      <h2 id="subtitle-quality-hold-title" className={styles.title}>
        Subtitle review required
      </h2>
      <p className={styles.body}>
        Generated subtitles for <strong>{videoFilename}</strong> triggered quality
        warnings. Dubbing is paused before voice generation.
      </p>

      <div className={styles.warningBox}>
        <strong>
          {issues.length} warning{issues.length === 1 ? '' : 's'} detected
        </strong>
        <ul>
          {issues.slice(0, 5).map((issue, index) => (
            <li key={`${issue.rule_id}-${issue.segment_index ?? 'file'}-${index}`}>
              {issue.label}: {issue.message}
            </li>
          ))}
        </ul>
      </div>

      <div className={styles.options}>
        <button type="button" onClick={onRetry} disabled={isSubmitting}>
          <strong>Retry transcription</strong>
          <span>Choose a different audio segment size and generate new subtitles.</span>
        </button>
        <button type="button" onClick={onEdit} disabled={isSubmitting}>
          <strong>Edit subtitles</strong>
          <span>Open the subtitle viewer, fix flagged cues, then continue.</span>
        </button>
        <button
          type="button"
          className={styles.ignoreButton}
          onClick={onIgnore}
          disabled={isSubmitting}
        >
          <strong>{isSubmitting ? 'Continuing…' : 'Continue anyway'}</strong>
          <span>Ignore these warnings and continue voice generation.</span>
        </button>
      </div>

      <button
        type="button"
        className={styles.cancelButton}
        onClick={onCancel}
        disabled={isSubmitting}
      >
        Close
      </button>
    </div>
  </div>
);
