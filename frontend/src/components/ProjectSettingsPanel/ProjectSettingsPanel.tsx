import { useState } from 'react';
import type { Project } from '../../types';
import { getVoiceIcon, getVoiceDisplayName } from '../../constants/voices';
import styles from './ProjectSettingsPanel.module.css';

// ─── Language list (shared with TargetLanguage component) ────────────────────

const LANGUAGE_OPTIONS = [
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

const langLabel = (code: string) =>
  LANGUAGE_OPTIONS.find((l) => l.code === code)?.label ?? code.toUpperCase();

// ─── Props ───────────────────────────────────────────────────────────────────

interface ProjectSettingsPanelProps {
  project: Project;
  onOpenVoiceRefinement: () => void;
  onUpdateSourceLanguage: (lang: string) => Promise<void>;
  onUpdateTargetLanguage: (lang: string) => Promise<void>;
  isSavingSource?: boolean;
  isSavingTarget?: boolean;
}

// ─── Component ───────────────────────────────────────────────────────────────

export const ProjectSettingsPanel = ({
  project,
  onOpenVoiceRefinement,
  onUpdateSourceLanguage,
  onUpdateTargetLanguage,
  isSavingSource = false,
  isSavingTarget = false,
}: ProjectSettingsPanelProps) => {
  const [isOpen, setIsOpen] = useState(false);

  const voiceIcon = getVoiceIcon(project.voice);
  const voiceName = getVoiceDisplayName(project.voice);
  const hasInstructions = !!project.voice_instructions?.trim();

  const srcCode = project.source_language_override || '';
  const tgtCode = project.target_language ?? 'eng';

  // Collapsed summary: voice · src → tgt
  const srcLabel = srcCode ? langLabel(srcCode) : 'Auto';
  const tgtLabel = langLabel(tgtCode);

  return (
    <div className={styles.wrapper}>
      {/* ── Trigger row ── */}
      <button
        className={styles.trigger}
        onClick={() => setIsOpen((v) => !v)}
        type="button"
        aria-expanded={isOpen}
      >
        <span className={styles.triggerIcon}>⚙</span>
        <span className={styles.triggerLabel}>Project Settings</span>

        {/* Summary badges when collapsed: voice then src → tgt */}
        {!isOpen && (
          <span className={styles.summary}>
            <span className={styles.summaryBadge}>
              {voiceIcon} {voiceName}{hasInstructions ? ' ✦' : ''}
            </span>
            <span className={styles.summaryBadge}>
              {srcLabel} → {tgtLabel}
            </span>
          </span>
        )}

        <span className={`${styles.chevron} ${isOpen ? styles.chevronOpen : ''}`}>›</span>
      </button>

      {/* ── Expandable panel ── */}
      {isOpen && (
        <div className={styles.panel}>
          <div className={styles.grid}>

            {/* Source language */}
            <div className={styles.field}>
              <label className={styles.fieldLabel} htmlFor="src-lang-select">
                Source Language
              </label>
              <div className={styles.selectRow}>
                <select
                  id="src-lang-select"
                  className={styles.select}
                  value={srcCode}
                  disabled={isSavingSource}
                  onChange={(e) => onUpdateSourceLanguage(e.target.value)}
                >
                  <option value="">Auto-detect</option>
                  {LANGUAGE_OPTIONS.map((l) => (
                    <option key={l.code} value={l.code}>
                      {l.label} ({l.code})
                    </option>
                  ))}
                </select>
                {isSavingSource && <span className={styles.saving}>Saving…</span>}
              </div>
            </div>

            {/* Target language */}
            <div className={styles.field}>
              <label className={styles.fieldLabel} htmlFor="tgt-lang-select">
                Target Language
              </label>
              <div className={styles.selectRow}>
                <select
                  id="tgt-lang-select"
                  className={styles.select}
                  value={tgtCode}
                  disabled={isSavingTarget}
                  onChange={(e) => onUpdateTargetLanguage(e.target.value)}
                >
                  {LANGUAGE_OPTIONS.map((l) => (
                    <option key={l.code} value={l.code}>
                      {l.label} ({l.code})
                    </option>
                  ))}
                </select>
                {isSavingTarget && <span className={styles.saving}>Saving…</span>}
              </div>
            </div>

            {/* Voice */}
            <div className={styles.field}>
              <span className={styles.fieldLabel}>Voice</span>
              <div className={styles.voiceRow}>
                <span className={styles.voiceValue}>
                  {voiceIcon} {voiceName}
                </span>
                <button
                  className={styles.refineButton}
                  onClick={onOpenVoiceRefinement}
                  type="button"
                >
                  ✨ Customize
                </button>
              </div>
            </div>

            {/* Voice instructions */}
            <div className={`${styles.field} ${styles.fieldFull}`}>
              <span className={styles.fieldLabel}>Voice Instructions</span>
              {hasInstructions ? (
                <p className={styles.instructions}>{project.voice_instructions}</p>
              ) : (
                <span className={styles.fieldMuted}>
                  None — click Customize to generate with AI
                </span>
              )}
            </div>

          </div>
        </div>
      )}
    </div>
  );
};
