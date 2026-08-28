import { useEffect, useMemo, useRef, useState } from 'react';
import type { SubtitleReviewData, SubtitleReviewSegment } from './types';
import styles from './SubtitleReview.module.css';

function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) {
    return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }
  return `${m}:${s.toString().padStart(2, '0')}`;
}

interface SubtitleReviewProps {
  isOpen: boolean;
  onClose: () => void;
  filename?: string;
  data: SubtitleReviewData | null;
  loading: boolean;
  error: string | null;
  selectedSrtPath?: string | null;
  onSrtPathChange?: (path: string) => void;
  onSaveCue?: (index: number, text: string) => Promise<void>;
  savingCueIndex?: number | null;
  onDeleteCue?: (index: number) => Promise<void>;
  deletingCueIndex?: number | null;
  onContinueRedub?: () => Promise<void>;
  continuingRedub?: boolean;
}

export const SubtitleReview = ({
  isOpen,
  onClose,
  filename,
  data,
  loading,
  error,
  selectedSrtPath,
  onSrtPathChange,
  onSaveCue,
  savingCueIndex = null,
  onDeleteCue,
  deletingCueIndex = null,
  onContinueRedub,
  continuingRedub = false,
}: SubtitleReviewProps) => {
  const [minDuration, setMinDuration] = useState(0);
  const [maxDuration, setMaxDuration] = useState(0);
  const [playing, setPlaying] = useState<{ index: number; kind: 'orig' | 'dub' } | null>(null);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [draftText, setDraftText] = useState('');
  const [saveError, setSaveError] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<{ index: number; message: string } | null>(null);
  const origAudio = useRef<HTMLAudioElement | null>(null);
  const ttsAudio = useRef<HTMLAudioElement | null>(null);
  const stopAtRef = useRef<number | null>(null);
  const lastChunkUrl = useRef('');
  const draftRef = useRef('');
  const committingRef = useRef(false);

  useEffect(() => {
    if (!isOpen) {
      origAudio.current?.pause();
      ttsAudio.current?.pause();
      setPlaying(null);
      setEditingIndex(null);
      setSaveError(null);
      setDeleteError(null);
    }
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (editingIndex !== null) return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isOpen, onClose, editingIndex]);

  useEffect(() => {
    if (!isOpen) return;
    const audio = origAudio.current;
    if (!audio) return;
    const onTime = () => {
      if (stopAtRef.current != null && audio.currentTime >= stopAtRef.current) {
        audio.pause();
        stopAtRef.current = null;
        setPlaying(null);
      }
    };
    const onEnded = () => {
      stopAtRef.current = null;
      setPlaying((current) => (current?.kind === 'orig' ? null : current));
    };
    audio.addEventListener('timeupdate', onTime);
    audio.addEventListener('ended', onEnded);
    return () => {
      audio.removeEventListener('timeupdate', onTime);
      audio.removeEventListener('ended', onEnded);
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const stopAll = () => {
    origAudio.current?.pause();
    ttsAudio.current?.pause();
    stopAtRef.current = null;
  };

  const playOriginal = async (segment: SubtitleReviewSegment) => {
    if (!segment.original || !origAudio.current) return;
    if (playing?.index === segment.index && playing.kind === 'orig') {
      origAudio.current.pause();
      setPlaying(null);
      return;
    }
    stopAll();
    const audio = origAudio.current;
    const { chunk_url, seek_start, seek_end } = segment.original;
    if (lastChunkUrl.current !== chunk_url) {
      audio.src = chunk_url;
      lastChunkUrl.current = chunk_url;
      await new Promise<void>((resolve) => {
        const ready = () => {
          audio.removeEventListener('loadedmetadata', ready);
          resolve();
        };
        audio.addEventListener('loadedmetadata', ready);
        audio.load();
      });
    }
    audio.currentTime = seek_start;
    stopAtRef.current = seek_end;
    setPlaying({ index: segment.index, kind: 'orig' });
    try {
      await audio.play();
    } catch {
      setPlaying(null);
    }
  };

  const playTts = async (segment: SubtitleReviewSegment) => {
    if (!segment.tts_url || !ttsAudio.current) return;
    if (playing?.index === segment.index && playing.kind === 'dub') {
      ttsAudio.current.pause();
      setPlaying(null);
      return;
    }
    stopAll();
    const audio = ttsAudio.current;
    audio.src = segment.tts_url;
    setPlaying({ index: segment.index, kind: 'dub' });
    try {
      await audio.play();
    } catch {
      setPlaying(null);
    }
  };

  const beginEdit = (segment: SubtitleReviewSegment) => {
    if (!onSaveCue) return;
    setEditingIndex(segment.index);
    setDraftText(segment.text);
    draftRef.current = segment.text;
    setSaveError(null);
  };

  const cancelEdit = () => {
    setEditingIndex(null);
    setDraftText('');
    setSaveError(null);
  };

  const commitEdit = async (segment: SubtitleReviewSegment) => {
    if (!onSaveCue || editingIndex !== segment.index || committingRef.current) return;
    const next = draftRef.current.trim();
    if (!next) {
      setSaveError('Cue text cannot be empty');
      return;
    }
    if (next === segment.text) {
      cancelEdit();
      return;
    }
    committingRef.current = true;
    try {
      await onSaveCue(segment.index, next);
      cancelEdit();
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setSaveError(detail || (err instanceof Error ? err.message : 'Failed to save cue'));
    } finally {
      committingRef.current = false;
    }
  };

  const deleteCue = async (segment: SubtitleReviewSegment) => {
    if (!onDeleteCue || deletingCueIndex !== null) return;
    if (!window.confirm(`Delete subtitle segment ${segment.index + 1}?`)) return;

    setDeleteError(null);
    stopAll();
    setPlaying(null);
    try {
      await onDeleteCue(segment.index);
      cancelEdit();
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setDeleteError({
        index: segment.index,
        message: detail || (err instanceof Error ? err.message : 'Failed to delete cue'),
      });
    }
  };

  const filtered = (data?.segments ?? []).filter((segment) => {
    if (minDuration > 0 && segment.duration < minDuration) return false;
    if (maxDuration > 0 && segment.duration > maxDuration) return false;
    return true;
  });

  const availableFiles = data?.available_files ?? [];
  const showFileSelector = availableFiles.length > 1;
  const activeSrtPath = selectedSrtPath ?? data?.srt_path ?? '';
  const qualityRules = data?.quality_rules ?? [];
  const qualityBreaches = data?.quality_breaches ?? [];
  const ruleLabels = Object.fromEntries(qualityRules.map((rule) => [rule.id, rule.label]));
  const cueBreaches = (index: number) =>
    qualityBreaches.filter((breach) => breach.segment_index === index);
  const breachedCueRuleCount = new Set(
    qualityBreaches
      .filter((breach) => breach.segment_index != null)
      .map((breach) => breach.rule_id),
  ).size;
  const fileLevelBreaches = qualityBreaches.filter((breach) => breach.segment_index == null);

  const firstBreachedCueIndex = useMemo(() => {
    const indices = qualityBreaches
      .filter((b) => b.segment_index != null)
      .map((b) => b.segment_index as number);
    return indices.length ? Math.min(...indices) : null;
  }, [qualityBreaches]);

  const scrollToFirstBreach = () => {
    if (firstBreachedCueIndex == null) return;
    document.getElementById(`cue-${firstBreachedCueIndex}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  return (
    <div className={styles.overlay} role="dialog" aria-modal="true" aria-label="Subtitle review">
      <div className={styles.panel}>
        <header className={styles.header}>
          <div className={styles.headerText}>
            <h2 className={styles.title}>Subtitles</h2>
            <p className={styles.filename}>{filename || data?.filename}</p>
          </div>
          {showFileSelector && (
            <label className={styles.fileSelector}>
              <span className={styles.fileSelectorLabel}>file</span>
              <select
                className={styles.fileSelect}
                value={activeSrtPath}
                onChange={(e) => onSrtPathChange?.(e.target.value)}
                aria-label="Select subtitle file"
              >
                {availableFiles.map((file) => {
                  const sourceTag = ({ generated: 'generated', sidecar: 'external', working_dir: 'edited' } as Record<string, string>)[file.source] ?? file.source;
                  const hasDuplicate = availableFiles.some((f) => f.path !== file.path && f.label === file.label);
                  return (
                    <option key={file.path} value={file.path}>
                      {hasDuplicate ? `${file.label} — ${sourceTag}` : `${file.label} (${sourceTag})`}
                    </option>
                  );
                })}
              </select>
            </label>
          )}
          <div className={styles.filters}>
            {qualityBreaches.length > 0 && (
              <button
                type="button"
                className={styles.warningIndicator}
                onClick={scrollToFirstBreach}
                title="Click to scroll to first warning"
                aria-label={`${breachedCueRuleCount} rule(s) breached — click to jump to first`}
              >
                <span className={styles.warningIcon} aria-hidden="true">!</span>
                <span className={styles.warningCount}>
                  {breachedCueRuleCount} rule{breachedCueRuleCount === 1 ? '' : 's'}
                </span>
                <span className={styles.warningTooltip} role="tooltip">
                  <strong>Quality rule breaches</strong>
                  <ul className={styles.warningList}>
                    {qualityRules
                      .filter((rule) =>
                        qualityBreaches.some((breach) => breach.rule_id === rule.id),
                      )
                      .map((rule) => {
                        const ruleBreaches = qualityBreaches.filter(
                          (breach) => breach.rule_id === rule.id,
                        );
                        return (
                          <li key={rule.id}>
                            <span className={styles.ruleLabel}>{rule.label}</span>
                            {' '}
                            ({ruleBreaches.length} hit{ruleBreaches.length === 1 ? '' : 's'})
                          </li>
                        );
                      })}
                  </ul>
                  {fileLevelBreaches.length > 0 && (
                    <p className={styles.fileLevelNote}>
                      Plus {fileLevelBreaches.length} file-level breach
                      {fileLevelBreaches.length === 1 ? '' : 'es'}.
                    </p>
                  )}
                </span>
              </button>
            )}
            <label className={styles.filterLabel}>
              min
              <input
                className={styles.filterInput}
                type="number"
                min={0}
                step={0.5}
                value={minDuration || ''}
                placeholder="0"
                onChange={(e) => setMinDuration(e.target.value === '' ? 0 : Number(e.target.value))}
                aria-label="Minimum duration in seconds"
              />
            </label>
            <label className={styles.filterLabel}>
              max
              <input
                className={styles.filterInput}
                type="number"
                min={0}
                step={0.5}
                value={maxDuration || ''}
                placeholder="any"
                onChange={(e) => setMaxDuration(e.target.value === '' ? 0 : Number(e.target.value))}
                aria-label="Maximum duration in seconds"
              />
            </label>
            <span className={styles.count}>
              {data ? `${filtered.length}/${data.total}` : ''}
            </span>
          </div>
          {onContinueRedub && (
            <button
              type="button"
              className={styles.continueButton}
              onClick={() => void onContinueRedub()}
              disabled={continuingRedub}
            >
              {continuingRedub ? 'Continuing…' : 'Recheck & continue'}
            </button>
          )}
          <button className={styles.close} type="button" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className={styles.script}>
          {loading && <p className={styles.status}>Loading…</p>}
          {error && <p className={styles.statusError}>{error}</p>}
          {!loading && !error && filtered.length === 0 && (
            <p className={styles.status}>No cues match the duration filter.</p>
          )}
          {filtered.map((segment) => {
            const isOrig = playing?.index === segment.index && playing.kind === 'orig';
            const isDub = playing?.index === segment.index && playing.kind === 'dub';
            const hasTts = Boolean(segment.tts_url);
            const rowBreaches = cueBreaches(segment.index);
            const ruleCount = segment.breached_rule_count ?? new Set(rowBreaches.map((b) => b.rule_id)).size;
            return (
              <div
                key={segment.index}
                id={`cue-${segment.index}`}
                className={`${styles.cue} ${isOrig || isDub ? styles.cueActive : ''}`}
              >
                <div className={styles.cueMain}>
                  {ruleCount > 0 ? (
                    <span
                      className={styles.cueWarningBar}
                      tabIndex={0}
                      aria-label={`${ruleCount} rule(s) breached in this cue`}
                    >
                      <span className={styles.cueRuleCount}>
                        {ruleCount}
                      </span>
                      <span className={styles.cueWarningTooltip} role="tooltip">
                        <strong>
                          {ruleCount} rule{ruleCount === 1 ? '' : 's'} breached
                        </strong>
                        <ul className={styles.warningList}>
                          {rowBreaches.map((breach, breachIndex) => (
                            <li key={`${breach.rule_id}-${breachIndex}`}>
                              <span className={styles.ruleLabel}>
                                {ruleLabels[breach.rule_id] ?? breach.rule_id}
                              </span>
                              {': '}
                              {breach.message}
                            </li>
                          ))}
                        </ul>
                      </span>
                    </span>
                  ) : (
                    <span className={styles.cueWarningSpacer} aria-hidden="true" />
                  )}
                  <div className={styles.cueText}>
                    <span className={styles.cueTime}>{formatTime(segment.start)}</span>
                    {editingIndex === segment.index ? (
                      <textarea
                        className={styles.cueEditor}
                        value={draftText}
                        autoFocus
                        disabled={savingCueIndex === segment.index}
                        aria-label={`Edit cue ${segment.index + 1}`}
                        onChange={(event) => {
                          setDraftText(event.target.value);
                          draftRef.current = event.target.value;
                        }}
                        onBlur={() => void commitEdit(segment)}
                        onKeyDown={(event) => {
                          if (event.key === 'Escape') {
                            event.preventDefault();
                            event.stopPropagation();
                            cancelEdit();
                          }
                          if (event.key === 'Enter' && !event.shiftKey) {
                            event.preventDefault();
                            void commitEdit(segment);
                          }
                        }}
                      />
                    ) : (
                      <button
                        type="button"
                        className={onSaveCue ? styles.cueTextButton : styles.cueTextStatic}
                        onClick={() => beginEdit(segment)}
                        disabled={!onSaveCue}
                      >
                        {segment.text}
                      </button>
                    )}
                  </div>
                  {editingIndex === segment.index && saveError && (
                    <p className={styles.cueSaveError} role="alert">{saveError}</p>
                  )}
                  {deleteError?.index === segment.index && (
                    <p className={styles.cueSaveError} role="alert">{deleteError.message}</p>
                  )}
                </div>
                {(hasTts || onDeleteCue) && (
                  <div className={styles.cueActions}>
                    {hasTts && (
                      <>
                        <button
                          type="button"
                          className={`${styles.play} ${isOrig ? styles.playActive : ''}`}
                          disabled={!segment.original}
                          title={segment.original ? 'Play original audio chunk' : 'Original chunk not available'}
                          onClick={() => void playOriginal(segment)}
                        >
                          {isOrig ? '■ orig' : '▶ orig'}
                        </button>
                        <button
                          type="button"
                          className={`${styles.play} ${isDub ? styles.playActive : ''}`}
                          title="Play dubbed TTS segment"
                          onClick={() => void playTts(segment)}
                        >
                          {isDub ? '■ dub' : '▶ dub'}
                        </button>
                      </>
                    )}
                    {onDeleteCue && (
                      <button
                        type="button"
                        className={styles.deleteCue}
                        disabled={deletingCueIndex !== null || savingCueIndex !== null}
                        aria-label={`Delete subtitle segment ${segment.index + 1}`}
                        onClick={() => void deleteCue(segment)}
                      >
                        {deletingCueIndex === segment.index ? 'Deleting…' : 'Delete'}
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <audio ref={origAudio} preload="auto" />
        <audio
          ref={ttsAudio}
          preload="auto"
          onEnded={() => setPlaying((current) => (current?.kind === 'dub' ? null : current))}
        />
      </div>
    </div>
  );
};
