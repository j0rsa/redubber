import type { VoicePreview } from './types';
import { AVAILABLE_VOICES } from '../../constants/voices';
import styles from './VoiceRefinement.module.css';

interface ManualVoiceTabProps {
  voiceInstructions: string;
  selectedVoice: string | null;
  previews: VoicePreview[];
  generatingPreviews: boolean;
  onUpdateInstructions: (instructions: string) => void;
  onSelectVoice: (voice: string) => void;
  onGeneratePreviews: () => Promise<void>;
}

export const ManualVoiceTab = ({
  voiceInstructions,
  selectedVoice,
  previews,
  generatingPreviews,
  onUpdateInstructions,
  onSelectVoice,
  onGeneratePreviews,
}: ManualVoiceTabProps) => {
  const hasPreviews = previews.length > 0;

  return (
    <div className={styles.manualTab}>

      {/* Voice selector */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Select Voice</h3>
        <p className={styles.sectionDescription}>
          Pick the voice that best fits your content.
        </p>
        <div className={styles.manualVoiceGrid}>
          {AVAILABLE_VOICES.map((v) => {
            const isSelected = selectedVoice === v.id;
            return (
              <button
                key={v.id}
                className={`${styles.manualVoiceCard} ${isSelected ? styles.manualVoiceCardSelected : ''}`}
                onClick={() => onSelectVoice(v.id)}
              >
                <span className={styles.manualVoiceIcon}>{v.icon}</span>
                <span className={styles.manualVoiceName}>{v.name}</span>
                <span className={styles.manualVoiceDesc}>{v.description}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Instructions */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Voice Instructions</h3>
        <p className={styles.sectionDescription}>
          Describe the speaking style, tone, pace, and accent. This is sent directly to the TTS engine.
        </p>
        <textarea
          className={styles.manualInstructions}
          value={voiceInstructions}
          onChange={(e) => onUpdateInstructions(e.target.value)}
          placeholder={
            'e.g., Speak with a warm, professional tone at a moderate pace. ' +
            'Clear enunciation with a slight Japanese accent — softer consonants, ' +
            'flatter intonation compared to native English.'
          }
          rows={6}
        />
        <div className={styles.instructionsFooter}>
          <span
            className={`${styles.characterCount} ${voiceInstructions.length > 500 ? styles.characterCountWarning : ''}`}
          >
            {voiceInstructions.length} characters
            {voiceInstructions.length > 500 && ' (may be truncated)'}
          </span>
        </div>
      </div>

      {/* Optional preview */}
      {voiceInstructions && selectedVoice && (
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>Preview (optional)</h3>
          <p className={styles.sectionDescription}>
            Generate audio samples to hear how each voice sounds with your instructions before saving.
          </p>
          <button
            className={styles.generateButton}
            onClick={onGeneratePreviews}
            disabled={generatingPreviews}
          >
            {generatingPreviews ? (
              <>
                <span className={styles.buttonSpinner} />
                Generating previews…
              </>
            ) : hasPreviews ? (
              '↻ Regenerate Previews'
            ) : (
              '▶ Generate Previews'
            )}
          </button>
        </div>
      )}

    </div>
  );
};
