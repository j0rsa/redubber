import { useEffect } from 'react';
import { useAudioPlayer } from '../../hooks/useAudioPlayer';
import { WaveformVisualizer } from './WaveformVisualizer';
import { formatTime } from './utils';
import styles from './AudioPlayerWithWaveform.module.css';

export interface AudioPlayerWithWaveformProps {
  audioUrl?: string;
  autoPlay?: boolean;
  onEnded?: () => void;
  className?: string;
  label?: string;
}

/**
 * Enhanced audio player with animated waveform visualization
 */
export const AudioPlayerWithWaveform = ({
  audioUrl,
  autoPlay = false,
  onEnded,
  className,
  label,
}: AudioPlayerWithWaveformProps) => {
  const {
    isPlaying,
    isLoading,
    error,
    duration,
    currentTime,
    progress,
    togglePlayPause,
    seekToProgress,
    audioRef,
  } = useAudioPlayer(audioUrl);

  // Auto-play when audio loads
  useEffect(() => {
    if (autoPlay && audioRef.current && !isLoading && audioUrl) {
      audioRef.current.play().catch((err) => {
        console.error('Auto-play failed:', err);
      });
    }
  }, [autoPlay, isLoading, audioUrl, audioRef]);

  // Call onEnded callback
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !onEnded) return;

    const handleEnded = () => onEnded();
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('ended', handleEnded);
    };
  }, [onEnded, audioRef]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
        return;
      }

      if (e.code === 'Space' && audioUrl) {
        e.preventDefault();
        togglePlayPause();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [togglePlayPause, audioUrl]);

  const handleWaveformClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const clickedProgress = (x / rect.width) * 100;
    seekToProgress(clickedProgress);
  };

  if (!audioUrl) {
    return (
      <div className={`${styles.container} ${className || ''}`}>
        <div className={styles.noAudio}>No audio available</div>
      </div>
    );
  }

  return (
    <div className={`${styles.container} ${className || ''}`}>
      {label && <div className={styles.label}>{label}</div>}

      <audio ref={audioRef} src={audioUrl} preload="metadata" />

      <div className={styles.player}>
        {/* Waveform Visualization */}
        <div
          className={styles.waveformWrapper}
          onClick={handleWaveformClick}
          role="slider"
          aria-label="Seek"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={progress}
          tabIndex={0}
        >
          {isLoading ? (
            <div className={styles.loadingWaveform}>
              <div className={styles.spinner} />
              <span>Loading audio...</span>
            </div>
          ) : error ? (
            <div className={styles.errorWaveform}>
              <span className={styles.errorIcon}>⚠️</span>
              <span>{error}</span>
            </div>
          ) : (
            <WaveformVisualizer isPlaying={isPlaying} progress={progress} />
          )}
        </div>

        {/* Controls */}
        <div className={styles.controls}>
          <button
            className={styles.playButton}
            onClick={togglePlayPause}
            disabled={isLoading || !!error}
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              <svg viewBox="0 0 24 24" className={styles.icon}>
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" className={styles.icon}>
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>

          <div className={styles.timeDisplay}>
            <span className={styles.time}>{formatTime(currentTime)}</span>
            <span className={styles.timeSeparator}>/</span>
            <span className={styles.time}>{formatTime(duration)}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
