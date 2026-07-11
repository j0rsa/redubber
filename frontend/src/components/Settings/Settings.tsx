import { useState, useId } from 'react';
import styles from './Settings.module.css';
import type { SettingsData, SttModel, TtsModel, VoiceAnalysisModel, VoiceAnalysisAudioModel, DefaultVoice } from '../../types/settings';
import { AVAILABLE_VOICES } from '../../constants/voices';

// ─── Props ────────────────────────────────────────────────────────────────────

export interface SettingsProps {
  settings: SettingsData;
  isSaving: boolean;
  error: string | null;
  successMessage: string | null;
  onSave: (update: Partial<SettingsData>) => void;
}

// ─── Option metadata ──────────────────────────────────────────────────────────

const STT_MODEL_OPTIONS: { value: SttModel; label: string }[] = [
  { value: 'gpt-4o-transcribe',      label: 'gpt-4o-transcribe — Best accuracy' },
  { value: 'gpt-4o-mini-transcribe', label: 'gpt-4o-mini-transcribe — Fast, affordable' },
  { value: 'whisper-1',              label: 'whisper-1 — Legacy, broad format support' },
];

const TTS_MODEL_OPTIONS: { value: TtsModel; label: string }[] = [
  { value: 'gpt-4o-mini-tts',  label: 'gpt-4o-mini-tts — Expressive, supports instructions' },
  { value: 'tts-1-hd',         label: 'tts-1-hd — High quality' },
  { value: 'tts-1',            label: 'tts-1 — Standard, lowest latency' },
];

const VOICE_ANALYSIS_MODEL_OPTIONS: { value: VoiceAnalysisModel; label: string }[] = [
  { value: 'o4-mini',   label: 'o4-mini — Best reasoning, fast' },
  { value: 'gpt-4o',    label: 'gpt-4o — Strong, balanced' },
  { value: 'o3',        label: 'o3 — Most capable reasoning' },
  { value: 'gpt-4o-mini', label: 'gpt-4o-mini — Fastest, cheapest' },
];

// ─── Component ────────────────────────────────────────────────────────────────

/**
 * Pure presentational Settings screen.
 *
 * Renders all application settings grouped into "API Configuration" and
 * "Workspace" sections. Holds only local UI state (API key visibility);
 * all data state is controlled externally via props.
 */
export const Settings = ({
  settings,
  isSaving,
  error,
  successMessage,
  onSave,
}: SettingsProps) => {
  const [showKey, setShowKey] = useState(false);
  const [localSettings, setLocalSettings] = useState<SettingsData>(settings);

  // Keep local copy in sync when parent settings change (e.g. after initial fetch)
  // We use the JSON key as a change signal to avoid infinite loops.
  const parentJson = JSON.stringify(settings);
  const [lastParentJson, setLastParentJson] = useState(parentJson);
  if (parentJson !== lastParentJson) {
    setLastParentJson(parentJson);
    setLocalSettings(settings);
  }

  const apiKeyId = useId();
  const baseUrlId = useId();
  const sttModelId = useId();
  const ttsModelId = useId();
  const analysisModelId = useId();
  const defaultVoiceId = useId();
  const projectsRootId = useId();
  const workDirId = useId();
  const autoProcessId = useId();
  const ttsSpeedId = useId();
  const ttsConcurrencyId = useId();
  const openaiTimeoutId = useId();
  const openaiRetriesId = useId();
  const audioChunkId = useId();

  const handleSave = (): void => {
    onSave(localSettings);
  };

  const isMaskedKey =
    localSettings.openai_api_key.startsWith('sk-...') ||
    (localSettings.openai_api_key.length > 0 &&
      localSettings.openai_api_key !== settings.openai_api_key);

  // Determine placeholder: masked display for a saved key
  const isSavedKey =
    settings.openai_api_key.length > 0 &&
    (settings.openai_api_key.startsWith('sk-...') || !showKey);

  return (
    <div className={styles.container}>
      <h1 className={styles.pageTitle}>Settings</h1>

      {error && (
        <div className={styles.errorBanner} role="alert">
          <WarningIcon />
          {error}
        </div>
      )}

      {successMessage && (
        <div className={styles.successBanner} role="status">
          <CheckIcon />
          {successMessage}
        </div>
      )}

      {/* ── API Configuration ─────────────────────────────────────────────── */}
      <section className={styles.section}>
        <h2 className={styles.sectionHeader}>API Configuration</h2>

        {/* OpenAI API Key */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={apiKeyId}>
              OpenAI API Key
            </label>
            <span className={styles.hint}>Used for transcription, translation, and TTS</span>
          </div>
          <div className={styles.controlCol}>
            <div className={styles.inputWrapper}>
              <input
                id={apiKeyId}
                type={showKey ? 'text' : 'password'}
                className={[
                  styles.input,
                  styles.inputWithToggle,
                  isSavedKey && !showKey ? styles.apiKeyMasked : '',
                ]
                  .filter(Boolean)
                  .join(' ')}
                value={localSettings.openai_api_key}
                onChange={(e) =>
                  setLocalSettings((s) => ({ ...s, openai_api_key: e.target.value }))
                }
                placeholder={
                  isMaskedKey && !showKey
                    ? '●●●●●●●●●●●●'
                    : 'sk-...'
                }
                autoComplete="off"
                spellCheck={false}
              />
              <button
                type="button"
                className={styles.eyeButton}
                onClick={() => setShowKey((v) => !v)}
                aria-label={showKey ? 'Hide API key' : 'Show API key'}
              >
                {showKey ? <EyeOffIcon /> : <EyeIcon />}
              </button>
            </div>
          </div>
        </div>

        {/* API Base URL */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={baseUrlId}>
              API Base URL
            </label>
            <span className={styles.hint}>Leave empty to use the default OpenAI endpoint</span>
          </div>
          <div className={styles.controlCol}>
            <input
              id={baseUrlId}
              type="url"
              className={styles.input}
              value={localSettings.openai_base_url ?? ''}
              onChange={(e) =>
                setLocalSettings((s) => ({ ...s, openai_base_url: e.target.value }))
              }
              placeholder="https://api.openai.com/v1"
              autoComplete="off"
              spellCheck={false}
            />
          </div>
        </div>

        {/* STT Model */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={sttModelId}>
              Transcription Model
            </label>
            <span className={styles.hint}>Model used to transcribe audio to text</span>
          </div>
          <div className={styles.controlCol}>
            <select
              id={sttModelId}
              className={styles.select}
              value={localSettings.stt_model}
              onChange={(e) =>
                setLocalSettings((s) => ({ ...s, stt_model: e.target.value as SttModel }))
              }
            >
              {STT_MODEL_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* TTS Model */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={ttsModelId}>
              TTS Model
            </label>
            <span className={styles.hint}>Model used to synthesise speech</span>
          </div>
          <div className={styles.controlCol}>
            <select
              id={ttsModelId}
              className={styles.select}
              value={localSettings.tts_model}
              onChange={(e) =>
                setLocalSettings((s) => ({ ...s, tts_model: e.target.value as TtsModel }))
              }
            >
              {TTS_MODEL_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Voice Analysis Model */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={analysisModelId}>
              Voice Analysis Model
            </label>
            <span className={styles.hint}>Text model for voice characteristic detection</span>
          </div>
          <div className={styles.controlCol}>
            <select
              id={analysisModelId}
              className={styles.select}
              value={localSettings.voice_analysis_model}
              onChange={(e) =>
                setLocalSettings((s) => ({
                  ...s,
                  voice_analysis_model: e.target.value as VoiceAnalysisModel,
                }))
              }
            >
              {VOICE_ANALYSIS_MODEL_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Voice Analysis Audio Model */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label}>
              Voice Analysis Audio Model
            </label>
            <span className={styles.hint}>Multimodal model for gender/accent detection from audio</span>
          </div>
          <div className={styles.controlCol}>
            <select
              className={styles.select}
              value={localSettings.voice_analysis_audio_model}
              onChange={(e) =>
                setLocalSettings((s) => ({
                  ...s,
                  voice_analysis_audio_model: e.target.value as VoiceAnalysisAudioModel,
                }))
              }
            >
              <option value="gpt-audio-1">gpt-audio-1 — Best accuracy</option>
              <option value="gpt-audio-mini">gpt-audio-mini — Faster, cheaper</option>
            </select>
          </div>
        </div>
      </section>

      {/* ── Processing ───────────────────────────────────────────────────── */}
      <section className={styles.section}>
        <h2 className={styles.sectionHeader}>Processing</h2>

        {/* TTS Speed */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={ttsSpeedId}>
              TTS Speed
            </label>
            <span className={styles.hint}>1.25 helps dubs fit timing, 1.0 = natural pace</span>
          </div>
          <div className={styles.controlCol}>
            <input
              id={ttsSpeedId}
              type="number"
              className={styles.input}
              value={localSettings.tts_speed}
              min={0.5}
              max={2.0}
              step={0.05}
              onChange={(e) =>
                setLocalSettings((s) => ({ ...s, tts_speed: parseFloat(e.target.value) }))
              }
            />
          </div>
        </div>

        {/* TTS Concurrency */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={ttsConcurrencyId}>
              TTS Concurrency
            </label>
            <span className={styles.hint}>Parallel TTS requests. Higher = faster but more API load</span>
          </div>
          <div className={styles.controlCol}>
            <input
              id={ttsConcurrencyId}
              type="number"
              className={styles.input}
              value={localSettings.tts_concurrency}
              min={1}
              max={100}
              onChange={(e) =>
                setLocalSettings((s) => ({ ...s, tts_concurrency: parseInt(e.target.value, 10) }))
              }
            />
          </div>
        </div>

        {/* OpenAI Timeout */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={openaiTimeoutId}>
              OpenAI Timeout
            </label>
            <span className={styles.hint}>Timeout per API request</span>
          </div>
          <div className={styles.controlCol}>
            <div className={styles.inputWithSuffix}>
              <input
                id={openaiTimeoutId}
                type="number"
                className={styles.input}
                value={localSettings.openai_timeout}
                min={5}
                max={600}
                onChange={(e) =>
                  setLocalSettings((s) => ({ ...s, openai_timeout: parseInt(e.target.value, 10) }))
                }
              />
              <span className={styles.inputSuffix}>s</span>
            </div>
          </div>
        </div>

        {/* OpenAI Retries */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={openaiRetriesId}>
              OpenAI Retries
            </label>
            <span className={styles.hint}>Retries on failed requests</span>
          </div>
          <div className={styles.controlCol}>
            <input
              id={openaiRetriesId}
              type="number"
              className={styles.input}
              value={localSettings.openai_retries}
              min={0}
              max={10}
              onChange={(e) =>
                setLocalSettings((s) => ({ ...s, openai_retries: parseInt(e.target.value, 10) }))
              }
            />
          </div>
        </div>

        {/* Audio Chunk Duration */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={audioChunkId}>
              Audio Chunk
            </label>
            <span className={styles.hint}>Audio chunk size for Whisper (max ~25MB)</span>
          </div>
          <div className={styles.controlCol}>
            <div className={styles.inputWithSuffix}>
              <input
                id={audioChunkId}
                type="number"
                className={styles.input}
                value={localSettings.audio_chunk_duration}
                min={60}
                max={3600}
                onChange={(e) =>
                  setLocalSettings((s) => ({ ...s, audio_chunk_duration: parseInt(e.target.value, 10) }))
                }
              />
              <span className={styles.inputSuffix}>s</span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Workspace ────────────────────────────────────────────────────── */}
      <section className={styles.section}>
        <h2 className={styles.sectionHeader}>Workspace</h2>

        {/* Default Voice */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={defaultVoiceId}>
              Default Voice
            </label>
            <span className={styles.hint}>Voice used when no preference is set</span>
          </div>
          <div className={styles.controlCol}>
            <select
              id={defaultVoiceId}
              className={styles.select}
              value={localSettings.default_voice}
              onChange={(e) =>
                setLocalSettings((s) => ({ ...s, default_voice: e.target.value as DefaultVoice }))
              }
            >
              {AVAILABLE_VOICES.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.icon}  {v.name} — {v.description}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Projects Root Path */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={projectsRootId}>
              Projects Root
            </label>
            <span className={styles.hint}>Starting folder for the file browser when creating a project. Also set via REDUBBER_PROJECTS_ROOT.</span>
          </div>
          <div className={styles.controlCol}>
            <input
              id={projectsRootId}
              type="text"
              className={styles.input}
              value={localSettings.projects_root_path}
              onChange={(e) =>
                setLocalSettings((s) => ({ ...s, projects_root_path: e.target.value }))
              }
              placeholder="/Users/you/Videos"
              spellCheck={false}
            />
          </div>
        </div>

        {/* Working Directory */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={workDirId}>
              Working Directory
            </label>
            <span className={styles.hint}>Where project folders are created. Also set via REDUBBER_WORKING_DIR.</span>
          </div>
          <div className={styles.controlCol}>
            <input
              id={workDirId}
              type="text"
              className={styles.input}
              value={localSettings.working_directory}
              onChange={(e) =>
                setLocalSettings((s) => ({ ...s, working_directory: e.target.value }))
              }
              placeholder="/Users/you/Videos"
              spellCheck={false}
            />
          </div>
        </div>

        {/* Auto-process */}
        <div className={styles.row}>
          <div className={styles.labelCol}>
            <label className={styles.label} htmlFor={autoProcessId}>
              Auto-process
            </label>
          </div>
          <div className={styles.controlCol}>
            <div className={styles.toggleWrapper}>
              <div className={styles.toggleRow}>
                <label className={styles.toggle}>
                  <input
                    id={autoProcessId}
                    type="checkbox"
                    checked={localSettings.auto_process}
                    onChange={(e) =>
                      setLocalSettings((s) => ({ ...s, auto_process: e.target.checked }))
                    }
                  />
                  <span className={styles.toggleTrack} />
                  <span className={styles.toggleThumb} />
                </label>
                <span className={styles.toggleLabel}>
                  Run all redub steps and replace the original file
                </span>
              </div>

              {localSettings.auto_process && (
                <div className={styles.autoProcessWarning} role="alert">
                  <span className={styles.warningIcon}>
                    <WarningIcon />
                  </span>
                  This will permanently overwrite original files
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <div className={styles.footer}>
        <button
          type="button"
          className={styles.saveButton}
          onClick={handleSave}
          disabled={isSaving}
        >
          {isSaving && <span className={styles.buttonSpinner} aria-hidden="true" />}
          {isSaving ? 'Saving…' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
};

// ─── Inline SVG icons (no external dep) ──────────────────────────────────────

const EyeIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M1 12S5 4 12 4s11 8 11 8-4 8-11 8S1 12 1 12z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

const EyeOffIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
    <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
    <line x1="1" y1="1" x2="23" y2="23" />
  </svg>
);

const CheckIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2.5"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

const WarningIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);
