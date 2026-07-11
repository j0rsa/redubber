import { useState, useRef, useEffect } from 'react';
import { AVAILABLE_VOICES, type VoicePreview } from './types';
import styles from './VoiceRefinement.module.css';

interface VoicePreviewGridProps {
  previews: VoicePreview[];
  selectedVoice: string | null;
  onSelectVoice: (voice: string) => void;
  onGeneratePreviews: () => Promise<void>;
  isGenerating?: boolean;
  hasInstructions?: boolean;
  speakerGender?: 'male' | 'female' | 'unknown';
}

export const VoicePreviewGrid = ({
  previews,
  selectedVoice,
  onSelectVoice,
  onGeneratePreviews,
  isGenerating = false,
  hasInstructions = false,
  speakerGender,
}: VoicePreviewGridProps) => {
  const [playingVoice, setPlayingVoice] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState<Record<string, number>>({});
  const audioRefs = useRef<Map<string, HTMLAudioElement>>(new Map());

  const getPreview = (voiceId: string): VoicePreview | undefined => {
    return previews.find((p) => p.voice === voiceId);
  };

  const handlePlay = (voiceId: string, audioUrl: string) => {
    // Stop any currently playing audio
    if (playingVoice) {
      const currentAudio = audioRefs.current.get(playingVoice);
      if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
      }
    }

    // Get or create audio element
    let audio = audioRefs.current.get(voiceId);
    if (!audio) {
      audio = new Audio(audioUrl);
      audioRefs.current.set(voiceId, audio);

      audio.addEventListener('ended', () => {
        setPlayingVoice(null);
        setCurrentTime((prev) => ({ ...prev, [voiceId]: 0 }));
      });

      audio.addEventListener('timeupdate', () => {
        setCurrentTime((prev) => ({ ...prev, [voiceId]: audio!.currentTime }));
      });

      audio.addEventListener('error', () => {
        setPlayingVoice(null);
        console.error('Error playing audio:', audioUrl);
      });
    }

    if (playingVoice === voiceId) {
      audio.pause();
      audio.currentTime = 0;
      setPlayingVoice(null);
    } else {
      audio.play().catch((err) => {
        console.error('Error playing audio:', err);
        setPlayingVoice(null);
      });
      setPlayingVoice(voiceId);
    }
  };

  const formatDuration = (ms: number): string => {
    const seconds = ms / 1000;
    return `${seconds.toFixed(1)}s`;
  };

  const getProgress = (voiceId: string, durationMs: number): number => {
    const current = currentTime[voiceId] || 0;
    const duration = durationMs / 1000;
    return duration > 0 ? (current / duration) * 100 : 0;
  };

  // Cleanup audio elements on unmount
  useEffect(() => {
    return () => {
      audioRefs.current.forEach((audio) => {
        audio.pause();
        audio.currentTime = 0;
      });
      audioRefs.current.clear();
    };
  }, []);

  if (!hasInstructions) {
    return (
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Step 3: Preview Voices</h3>
        <div className={styles.disabledState}>
          <p className={styles.disabledText}>
            Please analyze voice characteristics first
          </p>
        </div>
      </div>
    );
  }

  if (previews.length === 0) {
    return (
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Step 3: Preview Voices</h3>
        <p className={styles.sectionDescription}>
          Generate TTS previews for all available voices
        </p>
        <button
          className={styles.generateButton}
          onClick={onGeneratePreviews}
          disabled={isGenerating}
        >
          {isGenerating ? (
            <>
              <span className={styles.buttonSpinner} />
              Generating Previews...
            </>
          ) : (
            <>🎵 Generate Voice Previews</>
          )}
        </button>
      </div>
    );
  }

  return (
    <div className={styles.section}>
      <h3 className={styles.sectionTitle}>Step 3: Preview Voices</h3>
      <p className={styles.sectionDescription}>
        Listen and compare all voices to find the best match
      </p>

      <div className={styles.voiceGrid}>
        {AVAILABLE_VOICES.map((voice) => {
          const preview = getPreview(voice.id);
          const isPlaying = playingVoice === voice.id;
          const isSelected = selectedVoice === voice.id;
          const progress = preview ? getProgress(voice.id, preview.duration_ms) : 0;
          const isGenderMatch = speakerGender && speakerGender !== 'unknown' && voice.gender === speakerGender;

          return (
            <div
              key={voice.id}
              className={`${styles.voiceCard} ${
                isSelected ? styles.voiceCardSelected : ''
              } ${isGenderMatch && !isSelected ? styles.voiceCardGenderMatch : ''}`}
            >
              <div className={styles.voiceHeader}>
                <span className={styles.voiceIcon}>{voice.icon}</span>
                <div className={styles.voiceInfo}>
                  <h4 className={styles.voiceName}>
                    {voice.name}
                    {isGenderMatch && (
                      <span className={styles.genderMatchBadge} title={`Matches speaker gender (${speakerGender})`}>
                        {speakerGender === 'female' ? '♀' : '♂'}
                      </span>
                    )}
                  </h4>
                  <p className={styles.voiceDescription}>{voice.description}</p>
                </div>
                {preview?.cached && (
                  <span className={styles.cachedBadge}>Cached</span>
                )}
              </div>

              {preview && (
                <>
                  <div className={styles.audioPlayer}>
                    <button
                      className={styles.playButtonLarge}
                      onClick={() => handlePlay(voice.id, preview.audio_url)}
                    >
                      {isPlaying ? '⏸' : '▶'}
                    </button>

                    <div className={styles.waveformContainer}>
                      <div className={styles.waveform}>
                        <div
                          className={styles.waveformProgress}
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                      <span className={styles.duration}>
                        {formatDuration(preview.duration_ms)}
                      </span>
                    </div>
                  </div>

                  <label className={styles.radioLabel}>
                    <input
                      type="radio"
                      name="voice-select"
                      checked={isSelected}
                      onChange={() => onSelectVoice(voice.id)}
                      className={styles.radioInput}
                    />
                    <span className={styles.radioText}>
                      {isSelected ? 'Selected' : 'Select this voice'}
                    </span>
                  </label>
                </>
              )}

              {!preview && (
                <div className={styles.noPreview}>
                  <p>No preview available</p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
