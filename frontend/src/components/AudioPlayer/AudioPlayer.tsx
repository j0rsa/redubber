import { useEffect, useImperativeHandle, forwardRef } from 'react';
import { useAudioPlayer } from '../../hooks/useAudioPlayer';
import { formatTime } from './utils';
import styles from './AudioPlayer.module.css';

export interface AudioPlayerProps {
  audioUrl?: string;
  autoPlay?: boolean;
  onEnded?: () => void;
  className?: string;
  label?: string;
  /** Called when the user clicks play. Parent can use this to pause other players. */
  onPlayRequest?: () => void;
}

export interface AudioPlayerHandle {
  pause: () => void;
}

export const AudioPlayer = forwardRef<AudioPlayerHandle, AudioPlayerProps>(({
  audioUrl,
  autoPlay = false,
  onEnded,
  className,
  label,
  onPlayRequest,
}, ref) => {
  const {
    isPlaying,
    isLoading,
    error,
    duration,
    currentTime,
    progress,
    togglePlayPause,
    seekToProgress,
    pause,
    audioRef,
  } = useAudioPlayer(audioUrl);

  useImperativeHandle(ref, () => ({ pause }), [pause]);

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
      // Only handle if the player is focused or no input is focused
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

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
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
        {/* Play/Pause Button */}
        <button
          className={styles.playButton}
          onClick={() => {
            if (!isPlaying) onPlayRequest?.();
            togglePlayPause();
          }}
          disabled={isLoading || !!error}
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isLoading ? (
            <div className={styles.spinner} />
          ) : error ? (
            <span className={styles.errorIcon}>⚠️</span>
          ) : isPlaying ? (
            <svg viewBox="0 0 24 24" className={styles.icon}>
              <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" className={styles.icon}>
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        {/* Progress Bar */}
        <div className={styles.progressSection}>
          <span className={styles.time}>{formatTime(currentTime)}</span>

          <div
            className={styles.progressBar}
            onClick={handleProgressClick}
            role="slider"
            aria-label="Seek"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progress}
            tabIndex={0}
          >
            <div
              className={styles.progressFill}
              style={{ width: `${progress}%` }}
            />
            <div
              className={styles.progressHandle}
              style={{ left: `${progress}%` }}
            />
          </div>

          <span className={styles.time}>{formatTime(duration)}</span>
        </div>
      </div>

      {/* Error Message */}
      {error && <div className={styles.error}>{error}</div>}
    </div>
  );
});

AudioPlayer.displayName = 'AudioPlayer';
