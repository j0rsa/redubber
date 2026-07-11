import styles from './TargetLanguage.module.css';

// ─── Language options ─────────────────────────────────────────────────────────

interface LanguageOption {
  code: string;
  label: string;
}

const LANGUAGE_OPTIONS: LanguageOption[] = [
  { code: 'eng', label: 'English' },
  { code: 'spa', label: 'Spanish' },
  { code: 'fra', label: 'French' },
  { code: 'deu', label: 'German' },
  { code: 'ita', label: 'Italian' },
  { code: 'por', label: 'Portuguese' },
  { code: 'rus', label: 'Russian' },
  { code: 'zho', label: 'Chinese' },
  { code: 'jpn', label: 'Japanese' },
  { code: 'kor', label: 'Korean' },
  { code: 'ara', label: 'Arabic' },
  { code: 'hin', label: 'Hindi' },
];

// ─── Props ────────────────────────────────────────────────────────────────────

export interface TargetLanguageProps {
  targetLanguage: string;
  onUpdate: (lang: string) => void;
  isSaving?: boolean;
}

// ─── Component ────────────────────────────────────────────────────────────────

/**
 * Compact target-language selector for a project.
 *
 * Pure presentational component — calls `onUpdate` immediately on change
 * (auto-save, no separate save button).
 */
export const TargetLanguage = ({
  targetLanguage,
  onUpdate,
  isSaving = false,
}: TargetLanguageProps) => {
  return (
    <div className={styles.wrapper}>
      <label className={styles.label} htmlFor="target-language-select">
        Target Language
      </label>
      <select
        id="target-language-select"
        className={styles.select}
        value={targetLanguage}
        disabled={isSaving}
        onChange={(e) => onUpdate(e.target.value)}
        aria-label="Select target dubbing language"
      >
        {LANGUAGE_OPTIONS.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.label} ({lang.code})
          </option>
        ))}
      </select>
      {isSaving && <span className={styles.savingIndicator} aria-live="polite">Saving…</span>}
    </div>
  );
};
