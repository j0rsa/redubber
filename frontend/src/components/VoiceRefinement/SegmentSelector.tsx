import { useState, useEffect, useRef } from 'react';
import type { TranscriptionSegment, SegmentFilter } from './types';
import styles from './VoiceRefinement.module.css';

const DEFAULT_FILTER: SegmentFilter = {
  search: '',
  minDuration: 0,
  maxDuration: 0,
};

function isDefaultFilter(filter: SegmentFilter): boolean {
  return (
    filter.search === DEFAULT_FILTER.search &&
    filter.minDuration === DEFAULT_FILTER.minDuration &&
    filter.maxDuration === DEFAULT_FILTER.maxDuration
  );
}

interface SegmentSelectorProps {
  segments: TranscriptionSegment[];
  selectedSegment: TranscriptionSegment | null;
  onSelectSegment: (segment: TranscriptionSegment) => void;
  isLoading?: boolean;
  /** Total segments before sampling/filtering, from API metadata */
  totalCandidates?: number;
  /** Whether the API has more pages of results */
  hasMore?: boolean;
  /** Controlled filter state — parent / hook owns it */
  filter: SegmentFilter;
  onFilterChange: (filter: SegmentFilter) => void;
  onLoadMore: () => void;
  /** Trigger STT-only pipeline run to generate segments */
  onTranscribe?: () => Promise<void>;
  transcribing?: boolean;
}

export const SegmentSelector = ({
  segments,
  selectedSegment,
  onSelectSegment,
  isLoading = false,
  totalCandidates,
  hasMore = false,
  filter,
  onFilterChange,
  onLoadMore,
  onTranscribe,
  transcribing = false,
}: SegmentSelectorProps) => {
  const [playingSegment, setPlayingSegment] = useState<string | null>(null);
  const [audioElements] = useState<Map<string, HTMLAudioElement>>(new Map());

  // Debounced search — we hold a local draft so the input stays responsive
  const [searchDraft, setSearchDraft] = useState(filter.search);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep local draft in sync when the filter is reset from outside
  useEffect(() => {
    setSearchDraft(filter.search);
  }, [filter.search]);

  const handleSearchChange = (value: string) => {
    setSearchDraft(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      onFilterChange({ ...filter, search: value });
    }, 300);
  };

  const handleMinDurationChange = (value: string) => {
    const parsed = value === '' ? 0 : Number(value);
    if (!Number.isNaN(parsed)) {
      onFilterChange({ ...filter, minDuration: parsed });
    }
  };

  const handleMaxDurationChange = (value: string) => {
    const parsed = value === '' ? 0 : Number(value);
    if (!Number.isNaN(parsed)) {
      onFilterChange({ ...filter, maxDuration: parsed });
    }
  };

  const handleResetFilters = () => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    setSearchDraft('');
    onFilterChange({ ...DEFAULT_FILTER });
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handlePlay = (segment: TranscriptionSegment, e: React.MouseEvent) => {
    e.stopPropagation();

    if (playingSegment) {
      const currentAudio = audioElements.get(playingSegment);
      if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
      }
    }

    let audio = audioElements.get(segment.id);
    if (!audio) {
      audio = new Audio(segment.audio_url);
      audioElements.set(segment.id, audio);

      audio.addEventListener('ended', () => {
        setPlayingSegment(null);
      });

      audio.addEventListener('error', () => {
        setPlayingSegment(null);
        console.error('Error playing audio:', segment.audio_url);
      });
    }

    if (playingSegment === segment.id) {
      audio.pause();
      audio.currentTime = 0;
      setPlayingSegment(null);
    } else {
      audio.play().catch((err) => {
        console.error('Error playing audio:', err);
        setPlayingSegment(null);
      });
      setPlayingSegment(segment.id);
    }
  };

  const showResetButton = !isDefaultFilter(filter);

  /** Result count line text */
  const renderResultCount = () => {
    if (isLoading && segments.length === 0) {
      return <span className={styles.segmentCountSkeleton} aria-busy="true" />;
    }

    const count = segments.length;

    if (filter.search.trim()) {
      return (
        <span className={styles.segmentCount}>
          {count === 0
            ? `No results for "${filter.search}"`
            : `${count} result${count !== 1 ? 's' : ''} for "${filter.search}"`}
        </span>
      );
    }

    if (totalCandidates !== undefined && totalCandidates > 0) {
      return (
        <span className={styles.segmentCount}>
          Showing {count} of {totalCandidates} segments (evenly sampled)
        </span>
      );
    }

    return null;
  };

  return (
    <div className={styles.section}>
      <h3 className={styles.sectionTitle}>Step 1: Choose Segment</h3>

      {/* Filter bar */}
      <div className={styles.filterBar}>
        <input
          className={styles.filterSearch}
          type="search"
          placeholder="Search transcription text…"
          value={searchDraft}
          onChange={(e) => handleSearchChange(e.target.value)}
          aria-label="Search transcription text"
        />

        <div className={styles.filterDuration}>
          <span className={styles.filterDurationLabel}>Duration:</span>
          <input
            className={styles.filterDurationInput}
            type="number"
            min={0}
            step={1}
            value={filter.minDuration === 0 ? '' : filter.minDuration}
            placeholder="0"
            onChange={(e) => handleMinDurationChange(e.target.value)}
            aria-label="Minimum duration in seconds"
          />
          <span className={styles.filterDurationSep}>s –</span>
          <input
            className={styles.filterDurationInput}
            type="number"
            min={0}
            step={1}
            value={filter.maxDuration === 0 ? '' : filter.maxDuration}
            placeholder="any"
            onChange={(e) => handleMaxDurationChange(e.target.value)}
            aria-label="Maximum duration in seconds"
          />
          <span className={styles.filterDurationLabel}>s</span>
        </div>

        {showResetButton && (
          <button
            className={styles.filterResetButton}
            onClick={handleResetFilters}
            type="button"
            aria-label="Reset filters"
          >
            Reset
          </button>
        )}
      </div>

      {/* Result count */}
      <div className={styles.filterResultCount}>{renderResultCount()}</div>

      {/* Loading / transcribing */}
      {(isLoading || transcribing) && segments.length === 0 && (
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <p>{transcribing ? 'Transcribing audio… this may take a few minutes' : 'Loading segments...'}</p>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !transcribing && segments.length === 0 && (
        <div className={styles.emptyState}>
          {filter.search.trim() || filter.minDuration > 0 || filter.maxDuration > 0 ? (
            <>
              <p className={styles.emptyText}>No segments match your filters.</p>
              <p className={styles.emptyHint}>Try adjusting the search or duration range.</p>
            </>
          ) : (
            <>
              <p className={styles.emptyText}>No transcription segments yet.</p>
              {onTranscribe ? (
                <>
                  <p className={styles.emptyHint}>
                    Transcribe the first video to get segments for voice selection — audio extraction + Whisper only, no TTS cost.
                  </p>
                  <button
                    className={styles.transcribeButton}
                    onClick={() => void onTranscribe()}
                    type="button"
                  >
                    Transcribe First Video
                  </button>
                </>
              ) : (
                <p className={styles.emptyHint}>Run the redubbing pipeline to generate segments.</p>
              )}
            </>
          )}
        </div>
      )}

      {/* Segment list */}
      {segments.length > 0 && (
        <div className={styles.segmentList}>
          {segments.map((segment) => (
            <div
              key={segment.id}
              className={`${styles.segmentCard} ${
                selectedSegment?.id === segment.id ? styles.segmentCardSelected : ''
              }`}
              onClick={() => onSelectSegment(segment)}
            >
              <div className={styles.segmentHeader}>
                <span className={styles.segmentFilename}>
                  📹 {segment.video_filename}
                </span>
                <span className={styles.segmentDuration}>
                  {formatTime(segment.start_time)} - {formatTime(segment.end_time)}{' '}
                  ({segment.duration.toFixed(1)}s)
                </span>
              </div>

              <p className={styles.segmentText}>{segment.original_text}</p>

              <div className={styles.segmentActions}>
                <button
                  className={styles.playButton}
                  onClick={(e) => handlePlay(segment, e)}
                  disabled={!segment.audio_url}
                >
                  {playingSegment === segment.id ? '⏸ Pause' : '▶ Play'}
                </button>
                <button
                  className={`${styles.selectButton} ${
                    selectedSegment?.id === segment.id ? styles.selectButtonActive : ''
                  }`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectSegment(segment);
                  }}
                >
                  {selectedSegment?.id === segment.id ? '✓ Selected' : 'Select'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Load more */}
      {hasMore && (
        <div className={styles.loadMoreContainer}>
          <button
            className={styles.loadMoreButton}
            onClick={onLoadMore}
            disabled={isLoading}
            type="button"
          >
            {isLoading ? (
              <>
                <span className={styles.loadMoreSpinner} />
                Loading…
              </>
            ) : (
              'Load more segments'
            )}
          </button>
        </div>
      )}
    </div>
  );
};
