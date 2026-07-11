# AudioPlayer Components

Reusable audio player components for the Redubber voice refinement system.

## Components

### AudioPlayer

A clean, modern audio player with standard controls.

**Features:**
- Play/pause button with loading state
- Seekable progress bar
- Time display (current / total)
- Error handling
- Keyboard shortcuts (Space = play/pause)
- Mobile-friendly (48px+ touch targets)
- Accessible (ARIA labels)

**Usage:**
```tsx
import { AudioPlayer } from '@/components/AudioPlayer';

<AudioPlayer
  audioUrl="/audio/sample.mp3"
  label="Original Audio"
  onEnded={() => console.log('Audio finished')}
/>
```

**Props:**
- `audioUrl?: string` - URL of the audio file to play
- `autoPlay?: boolean` - Auto-play when loaded (default: false)
- `onEnded?: () => void` - Callback when audio finishes
- `className?: string` - Additional CSS classes
- `label?: string` - Optional label above the player

---

### AudioPlayerWithWaveform

Enhanced audio player with animated waveform visualization.

**Features:**
- All features from AudioPlayer
- Animated waveform bars that respond to playback
- Click-to-seek on waveform
- Visual progress indicator on waveform
- Larger layout optimized for visual feedback

**Usage:**
```tsx
import { AudioPlayerWithWaveform } from '@/components/AudioPlayer';

<AudioPlayerWithWaveform
  audioUrl="/audio/tts-preview.mp3"
  label="TTS Preview: Nova"
  autoPlay
/>
```

**Props:**
- Same as AudioPlayer

---

### WaveformVisualizer

Standalone animated waveform visualization component.

**Usage:**
```tsx
import { WaveformVisualizer } from '@/components/AudioPlayer';

<WaveformVisualizer
  isPlaying={true}
  progress={45}
/>
```

**Props:**
- `isPlaying: boolean` - Whether audio is currently playing
- `progress: number` - Progress percentage (0-100)
- `className?: string` - Additional CSS classes

---

## Custom Hook: useAudioPlayer

A React hook for managing audio playback state.

**Usage:**
```tsx
import { useAudioPlayer } from '@/hooks/useAudioPlayer';

function MyPlayer() {
  const {
    isPlaying,
    isLoading,
    error,
    duration,
    currentTime,
    progress,
    play,
    pause,
    togglePlayPause,
    seek,
    seekToProgress,
    setVolume,
    audioRef,
  } = useAudioPlayer(audioUrl);

  return (
    <div>
      <audio ref={audioRef} src={audioUrl} />
      <button onClick={togglePlayPause}>
        {isPlaying ? 'Pause' : 'Play'}
      </button>
      <div>Progress: {progress.toFixed(0)}%</div>
    </div>
  );
}
```

**Return Value:**
- `isPlaying: boolean` - Whether audio is playing
- `isLoading: boolean` - Whether audio is loading
- `error: string | null` - Error message if load failed
- `duration: number` - Total duration in seconds
- `currentTime: number` - Current playback time in seconds
- `volume: number` - Current volume (0-1)
- `progress: number` - Progress percentage (0-100)
- `play: () => void` - Start playback
- `pause: () => void` - Pause playback
- `togglePlayPause: () => void` - Toggle play/pause
- `seek: (time: number) => void` - Seek to time in seconds
- `seekToProgress: (progress: number) => void` - Seek to progress % (0-100)
- `setVolume: (volume: number) => void` - Set volume (0-1)
- `audioRef: RefObject<HTMLAudioElement>` - Ref to audio element

**Features:**
- Automatic state management
- Event listeners for audio events
- Auto-cleanup on unmount
- State reset on URL change
- Error handling

---

## Utilities

### formatTime

Format time in seconds to human-readable format.

**Usage:**
```tsx
import { formatTime } from '@/components/AudioPlayer';

formatTime(0);      // "0:00"
formatTime(65);     // "1:05"
formatTime(3665);   // "1:01:05"
```

---

## Styling

All components use CSS modules with CSS variables from `variables.css`:

- `--color-primary` / `--color-primary-hover` - Primary button colors
- `--color-bg-elevated` / `--color-bg-secondary` - Background colors
- `--color-text-primary` / `--color-text-secondary` - Text colors
- `--color-border` / `--color-border-light` - Border colors
- `--color-error` / `--color-error-light` - Error state colors
- `--spacing-*` - Spacing scale
- `--radius-*` - Border radius scale
- `--transition-*` - Transition timing
- `--shadow-*` - Box shadows

Components automatically adapt to light/dark mode via CSS variables.

---

## Accessibility

All components follow accessibility best practices:

- Proper ARIA labels and roles
- Keyboard navigation support
- Focus-visible indicators
- Semantic HTML structure
- Minimum 48px touch targets on mobile
- Screen reader friendly

---

## Mobile Optimization

Components are mobile-first and include:

- Touch-friendly controls (48-56px buttons)
- Responsive layouts
- Optimized for touch devices
- Larger touch targets on mobile
- Performance-optimized animations

---

## Browser Support

Components use modern web APIs:
- HTML5 Audio API
- Canvas API (for waveform)
- CSS Grid & Flexbox
- CSS Custom Properties
- RequestAnimationFrame

Supports all modern browsers (Chrome, Firefox, Safari, Edge).

---

## Examples

### Voice Refinement Workflow

```tsx
function VoiceRefinement() {
  return (
    <div>
      <h2>Original Audio</h2>
      <AudioPlayer
        audioUrl="/api/audio/original/123"
        label="Original Transcription"
      />

      <h2>TTS Previews</h2>
      <AudioPlayerWithWaveform
        audioUrl="/api/audio/tts/123?voice=nova"
        label="Voice: Nova"
      />
      <AudioPlayerWithWaveform
        audioUrl="/api/audio/tts/123?voice=alloy"
        label="Voice: Alloy"
      />
    </div>
  );
}
```

### Comparison View

```tsx
function AudioComparison() {
  const [selectedVoice, setSelectedVoice] = useState<string | null>(null);

  return (
    <div className={styles.comparison}>
      {voices.map((voice) => (
        <div key={voice.id}>
          <AudioPlayerWithWaveform
            audioUrl={voice.audioUrl}
            label={voice.name}
            onEnded={() => setSelectedVoice(voice.id)}
          />
          {selectedVoice === voice.id && (
            <button>Use This Voice</button>
          )}
        </div>
      ))}
    </div>
  );
}
```
