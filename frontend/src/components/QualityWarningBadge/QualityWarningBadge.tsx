import type { SubtitleQualityIssue } from '../../types';
import styles from './QualityWarningBadge.module.css';

interface QualityWarningBadgeProps {
  issues: SubtitleQualityIssue[];
  compact?: boolean;
}

export const QualityWarningBadge = ({
  issues,
  compact = false,
}: QualityWarningBadgeProps) => {
  if (issues.length === 0) return null;

  const ruleIds = [...new Set(issues.map((issue) => issue.rule_id))];
  const ruleCount = ruleIds.length;
  const fileLevelCount = issues.filter((issue) => issue.segment_index == null).length;

  return (
    <span
      className={styles.warningIndicator}
      tabIndex={0}
      aria-label={`${ruleCount} quality rule${ruleCount === 1 ? '' : 's'} breached, ${issues.length} issue${issues.length === 1 ? '' : 's'}`}
    >
      <span className={styles.warningIcon} aria-hidden="true">!</span>
      <span className={styles.warningCount}>
        {compact
          ? String(ruleCount)
          : `${ruleCount} rule${ruleCount === 1 ? '' : 's'}`}
      </span>
      <span className={styles.warningTooltip} role="tooltip">
        <strong>Quality rule breaches</strong>
        <ul className={styles.warningList}>
          {issues.map((issue, index) => (
            <li key={`${issue.rule_id}-${issue.segment_index ?? 'file'}-${index}`}>
              <span className={styles.ruleLabel}>{issue.label}</span>
              {issue.segment_index != null && (
                <span className={styles.cueIndex}> cue {issue.segment_index + 1}</span>
              )}
              {': '}
              {issue.message}
            </li>
          ))}
        </ul>
        {fileLevelCount > 0 && (
          <p className={styles.fileLevelNote}>
            Includes {fileLevelCount} file-level breach{fileLevelCount === 1 ? '' : 'es'}.
          </p>
        )}
      </span>
    </span>
  );
};
