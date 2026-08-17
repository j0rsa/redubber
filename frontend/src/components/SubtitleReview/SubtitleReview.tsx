import { useEffect, useRef, useState } from 'react';
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
}

export const SubtitleReview = ({
  isOpen,
  onClose,
  filename,
  data,
  loading,
  error,
}: SubtitleReviewProps) => {
  const [minDuration, setMinDuration] = useState(0);
  const [maxDuration, setMaxDuration] = useState(0);
  const [playing, setPlaying] = useState<{ index: number; kind: 'orig' | 'dub' } | null>(null);
  const origAudio = useRef<HTMLAudioElement | null>(null);
  const ttsAudio = useRef<HTMLAudioElement | null>(null);
  const stopAtRef = useRef<number | null>(null);
  const lastChunkUrl = useRef('');

  useEffect(() => {
    if (!isOpen) {
      origAudio.current?.pause();
      ttsAudio.current?.pause();
      setPlaying(null);
    }
  }, [isOpen]);

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

  const filtered = (data?.segments ?? []).filter((segment) => {
    if (minDuration > 0 && segment.duration < minDuration) return false;
    if (maxDuration > 0 && segment.duration > maxDuration) return false;
    return true;
  });

  return (
    <div className={styles.overlay} role="dialog" aria-modal="true" aria-label="Subtitle review">
      <div className={styles.panel}>
        <header className={styles.header}>
          <div className={styles.headerText}>
            <h2 className={styles.title}>Subtitles</h2>
            <p className={styles.filename}>{filename || data?.filename}</p>
          </div>
          <div className={styles.filters}>
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
            return (
              <div
                key={segment.index}
                className={`${styles.cue} ${isOrig || isDub ? styles.cueActive : ''}`}
              >
                <p className={styles.cueText}>
                  <span className={styles.cueTime}>{formatTime(segment.start)}</span>
                  {segment.text}
                </p>
                {hasTts && (
                  <div className={styles.cueActions}>
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
