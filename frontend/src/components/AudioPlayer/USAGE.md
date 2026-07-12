# AudioPlayer Usage Examples

Quick reference for using AudioPlayer components in the Redubber voice refinement system.

## Quick Start

```tsx
import { AudioPlayer, AudioPlayerWithWaveform } from '@/components/AudioPlayer';

// Basic player
<AudioPlayer
  audioUrl="/api/audio/segment/123"
  label="Original Audio"
/>

// Player with waveform
<AudioPlayerWithWaveform
  audioUrl="/api/audio/tts/123?voice=nova"
  label="TTS Preview: Nova"
  autoPlay
/>
```

## Voice Refinement Workflow

### Original vs TTS Comparison

```tsx
import { AudioPlayerWithWaveform } from '@/components/AudioPlayer';

function SegmentComparison({ segmentId }: { segmentId: string }) {
  return (
    <div className={styles.comparison}>
      <div className={styles.section}>
        <h3>Original Audio</h3>
        <AudioPlayerWithWaveform
          audioUrl={`/api/audio/segment/${segmentId}/original`}
          label="Transcribed from video"
        />
      </div>

      <div className={styles.section}>
        <h3>TTS Preview</h3>
        <AudioPlayerWithWaveform
          audioUrl={`/api/audio/segment/${segmentId}/tts`}
          label="Generated voice"
        />
      </div>
    </div>
  );
}
```

### Voice Selection Grid

```tsx
import { AudioPlayerWithWaveform } from '@/components/AudioPlayer';
import { useState } from 'react';

const VOICES = [
  { id: 'alloy', name: 'Alloy', description: 'Neutral, balanced' },
  { id: 'echo', name: 'Echo', description: 'Male, clear' },
  { id: 'fable', name: 'Fable', description: 'British, expressive' },
  { id: 'nova', name: 'Nova', description: 'Warm, engaging' },
];

function VoiceSelector({ segmentId }: { segmentId: string }) {
  const [selectedVoice, setSelectedVoice] = useState<string | null>(null);

  return (
    <div className={styles.grid}>
      {VOICES.map((voice) => (
        <div
          key={voice.id}
          className={selectedVoice === voice.id ? styles.selected : ''}
        >
          <AudioPlayerWithWaveform
            audioUrl={`/api/audio/segment/${segmentId}/tts?voice=${voice.id}`}
            label={`${voice.name} - ${voice.description}`}
            onEnded={() => setSelectedVoice(voice.id)}
          />
          {selectedVoice === voice.id && (
            <button onClick={() => handleSelectVoice(voice.id)}>
              Use This Voice
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
```

### Segment List with Audio Preview

```tsx
import { AudioPlayer } from '@/components/AudioPlayer';
import type { Segment } from '@/types';

function SegmentList({ segments }: { segments: Segment[] }) {
  return (
    <div className={styles.list}>
      {segments.map((segment) => (
        <div key={segment.id} className={styles.segmentCard}>
          <div className={styles.segmentInfo}>
            <span className={styles.timing}>
              {formatTime(segment.start)} - {formatTime(segment.end)}
            </span>
            <p className={styles.text}>{segment.text}</p>
          </div>

          <AudioPlayer
            audioUrl={`/api/audio/segment/${segment.id}`}
            className={styles.compactPlayer}
          />
        </div>
      ))}
    </div>
  );
}
```

## Advanced Usage

### Custom Hook Integration

```tsx
import { useAudioPlayer } from '@/hooks/useAudioPlayer';
import { useEffect } from 'react';

function CustomAudioControl({ audioUrl }: { audioUrl: string }) {
  const {
    isPlaying,
    progress,
    currentTime,
    duration,
    togglePlayPause,
    seekToProgress,
    audioRef,
  } = useAudioPlayer(audioUrl);

  // Custom behavior
  useEffect(() => {
    if (progress > 50 && progress < 51) {
      console.log('Reached midpoint!');
    }
  }, [progress]);

  return (
    <div>
      <audio ref={audioRef} src={audioUrl} />

      <button onClick={togglePlayPause}>
        {isPlaying ? 'Pause' : 'Play'}
      </button>

      <div className={styles.customProgress}>
        <div className={styles.timeline}>
          {Array.from({ length: 10 }).map((_, i) => (
            <button
              key={i}
              onClick={() => seekToProgress(i * 10)}
              className={progress >= i * 10 ? styles.passed : ''}
            >
              {i * 10}%
            </button>
          ))}
        </div>
      </div>

      <div className={styles.timeInfo}>
        {formatTime(currentTime)} / {formatTime(duration)}
      </div>
    </div>
  );
}
```

### Synchronized Playback

```tsx
import { useAudioPlayer } from '@/hooks/useAudioPlayer';
import { useEffect } from 'react';

function SynchronizedPlayers({ urls }: { urls: string[] }) {
  const players = urls.map((url) => useAudioPlayer(url));

  const playAll = () => {
    players.forEach((player) => player.play());
  };

  const pauseAll = () => {
    players.forEach((player) => player.pause());
  };

  const seekAll = (progress: number) => {
    players.forEach((player) => player.seekToProgress(progress));
  };

  return (
    <div>
      <div className={styles.controls}>
        <button onClick={playAll}>Play All</button>
        <button onClick={pauseAll}>Pause All</button>
      </div>

      {urls.map((url, index) => (
        <div key={url}>
          <audio ref={players[index].audioRef} src={url} />
          <div>Player {index + 1}: {players[index].progress.toFixed(0)}%</div>
        </div>
      ))}
    </div>
  );
}
```

### Waveform Only

```tsx
import { WaveformVisualizer } from '@/components/AudioPlayer';
import { useAudioPlayer } from '@/hooks/useAudioPlayer';

function MinimalPlayer({ audioUrl }: { audioUrl: string }) {
  const { isPlaying, progress, togglePlayPause, audioRef } = useAudioPlayer(audioUrl);

  return (
    <div className={styles.minimal}>
      <audio ref={audioRef} src={audioUrl} />

      <div onClick={togglePlayPause} className={styles.waveformContainer}>
        <WaveformVisualizer isPlaying={isPlaying} progress={progress} />
      </div>

      <button onClick={togglePlayPause}>
        {isPlaying ? '⏸' : '▶️'}
      </button>
    </div>
  );
}
```

## Styling Examples

### Custom Theme

```css
/* Override CSS variables for custom theme */
.audioPlayerContainer {
  --color-primary: #ff6b6b;
  --color-primary-hover: #ff5252;
  --radius-lg: 16px;
}
```

### Compact Layout

```tsx
<AudioPlayer
  audioUrl={url}
  className={styles.compact}
/>

/* styles.module.css */
.compact {
  max-width: 400px;
}

.compact :global(.player) {
  padding: var(--spacing-sm);
  gap: var(--spacing-sm);
}
```

### Large Format

```tsx
<AudioPlayerWithWaveform
  audioUrl={url}
  label="High-Quality Preview"
  className={styles.large}
/>

/* styles.module.css */
.large {
  max-width: 900px;
}

.large :global(.waveformWrapper) {
  height: 120px;
}

.large :global(.playButton) {
  width: 64px;
  height: 64px;
}
```

## Performance Tips

1. **Lazy Loading**: Load audio URLs only when needed
   ```tsx
   const [shouldLoad, setShouldLoad] = useState(false);

   <AudioPlayer
     audioUrl={shouldLoad ? url : undefined}
   />
   ```

2. **Preloading**: Use `preload` attribute for faster playback
   ```tsx
   // Hook automatically sets preload="metadata"
   // Override via audioRef if needed
   useEffect(() => {
     if (audioRef.current) {
       audioRef.current.preload = 'auto';
     }
   }, [audioRef]);
   ```

3. **Memory Management**: Clean up when component unmounts
   ```tsx
   // Handled automatically by useAudioPlayer hook
   // No manual cleanup needed
   ```

## Accessibility

All components include:
- ARIA labels for screen readers
- Keyboard shortcuts (Space to play/pause)
- Focus indicators
- Semantic HTML structure

```tsx
// Custom ARIA labels
<AudioPlayer
  audioUrl={url}
  aria-label={`Audio preview for segment ${segmentId}`}
/>
```

## Mobile Optimization

Components automatically adapt to mobile:
- Larger touch targets (56px on mobile)
- Simplified layouts
- Touch-friendly progress bars
- Optimized animations

Test on mobile devices or use Chrome DevTools mobile emulation.
