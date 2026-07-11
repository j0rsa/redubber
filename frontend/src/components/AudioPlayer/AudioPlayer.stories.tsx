import { useRef } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { AudioPlayer } from './AudioPlayer';
import type { AudioPlayerHandle } from './AudioPlayer';

const meta = {
  title: 'Components/AudioPlayer',
  component: AudioPlayer,
  parameters: {
    layout: 'padded',
    backgrounds: { default: 'light-gray' },
    docs: {
      description: {
        component: 'Audio player with play/pause, seekable progress bar, and time display.',
      },
    },
  },
  argTypes: {
    audioUrl: { control: 'text' },
    autoPlay: { control: 'boolean' },
    label: { control: 'text' },
    onEnded: { action: 'ended' },
  },
} satisfies Meta<typeof AudioPlayer>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    audioUrl: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3',
    label: 'Sample Audio',
  },
};

export const NoAudio: Story = {
  args: {
    audioUrl: undefined,
    label: 'No Audio Available',
  },
};

export const ErrorState: Story = {
  args: {
    audioUrl: 'https://invalid-url.example.com/nonexistent.mp3',
    label: 'Audio with Error',
  },
};

export const OriginalTranscription: Story = {
  args: {
    audioUrl: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3',
    label: 'Original Audio (Transcription)',
  },
};

export const TTSPreview: Story = {
  args: {
    audioUrl: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3',
    label: 'TTS Preview — Nova',
  },
};

export const CompactNoLabel: Story = {
  args: {
    audioUrl: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3',
    label: undefined,
  },
};

/**
 * Six voice players — clicking any one stops all others first.
 * Demonstrates the ref-based exclusive-play pattern used in VoicePreviewGrid.
 */
export const VoiceComparisonGrid: Story = {
  render: () => {
    const voices = [
      { id: 'alloy',   label: 'Alloy — Neutral, balanced',    url: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3' },
      { id: 'echo',    label: 'Echo — Male, clear',            url: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3' },
      { id: 'fable',   label: 'Fable — British, expressive',   url: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3' },
      { id: 'onyx',    label: 'Onyx — Deep, authoritative',    url: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3' },
      { id: 'nova',    label: 'Nova — Warm, engaging female',  url: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3' },
      { id: 'shimmer', label: 'Shimmer — Soft, gentle female', url: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3' },
    ];

    // One ref per player — used to imperatively pause all others on play
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const refs = useRef<Record<string, AudioPlayerHandle | null>>(
      Object.fromEntries(voices.map((v) => [v.id, null]))
    );

    const handlePlayRequest = (activeId: string) => {
      voices.forEach(({ id }) => {
        if (id !== activeId) refs.current[id]?.pause();
      });
    };

    return (
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: '12px',
        maxWidth: '900px',
      }}>
        {voices.map((v) => (
          <AudioPlayer
            key={v.id}
            ref={(el) => { refs.current[v.id] = el; }}
            audioUrl={v.url}
            label={v.label}
            onPlayRequest={() => handlePlayRequest(v.id)}
            onEnded={() => {/* nothing — next doesn't auto-start */}}
          />
        ))}
      </div>
    );
  },
};
