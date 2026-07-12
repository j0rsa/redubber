import type { Meta, StoryObj } from '@storybook/react-vite';
import { Settings } from './Settings';
import type { SettingsProps } from './Settings';
import type { SettingsData } from '../../types/settings';
import { DEFAULT_SETTINGS } from '../../types/settings';

const meta: Meta<typeof Settings> = {
  title: 'Components/Settings',
  component: Settings,
  parameters: {
    layout: 'fullscreen',
    backgrounds: { default: 'dark' },
    docs: {
      description: {
        component:
          'Settings screen: configure OpenAI API key, TTS model, voice analysis model, default voice, working directory, and auto-process toggle.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof Settings>;

// ─── Shared base actions ──────────────────────────────────────────────────────

const baseActions: Pick<SettingsProps, 'onSave'> = {
  onSave: (update) => console.log('onSave called:', update),
};

// ─── Shared settings states ───────────────────────────────────────────────────

const emptySettings: SettingsData = { ...DEFAULT_SETTINGS };

const configuredSettings: SettingsData = {
  openai_api_key: 'sk-...xxxx',
  openai_base_url: '',
  stt_model: 'whisper-1',
  tts_model: 'tts-1',
  voice_analysis_model: 'gpt-4o',
  voice_analysis_audio_model: 'gpt-audio-1',
  default_voice: 'nova',
  projects_root_path: '/Users/jane/Videos',
  working_directory: '/Users/jane/redubber_output',
  auto_process: false,
  tts_concurrency: 20,
  openai_timeout: 60,
  openai_retries: 3,
  tts_speed: 1.25,
  audio_chunk_duration: 900,
};

// ─── Stories ─────────────────────────────────────────────────────────────────

/** Fresh install — all fields empty, no API key configured. */
export const Default: Story = {
  args: {
    ...baseActions,
    settings: emptySettings,
    isSaving: false,
    error: null,
    successMessage: null,
  },
};

/** API key masked as "sk-...xxxx", TTS model tts-1, voice nova, workdir set. */
export const Configured: Story = {
  args: {
    ...baseActions,
    settings: configuredSettings,
    isSaving: false,
    error: null,
    successMessage: null,
  },
};

/** Auto-process enabled — warning banner visible below toggle. */
export const WithAutoProcess: Story = {
  args: {
    ...baseActions,
    settings: { ...configuredSettings, auto_process: true },
    isSaving: false,
    error: null,
    successMessage: null,
  },
};

/** API key field in visible (plain-text) state. */
export const ApiKeyVisible: Story = {
  args: {
    ...baseActions,
    settings: { ...configuredSettings, openai_api_key: 'sk-proj-abcdefghijklmnopqrstuvwxyz' },
    isSaving: false,
    error: null,
    successMessage: null,
  },
};

/** Save in progress — button shows spinner and is disabled. */
export const Saving: Story = {
  args: {
    ...baseActions,
    settings: configuredSettings,
    isSaving: true,
    error: null,
    successMessage: null,
  },
};

/** Save failed — error banner visible. */
export const SaveError: Story = {
  args: {
    ...baseActions,
    settings: configuredSettings,
    isSaving: false,
    error: 'Failed to save settings',
    successMessage: null,
  },
};

/** Save succeeded — success banner visible. */
export const SaveSuccess: Story = {
  args: {
    ...baseActions,
    settings: configuredSettings,
    isSaving: false,
    error: null,
    successMessage: 'Settings saved',
  },
};

/** tts-1-hd selected as the TTS model. */
export const HighQualityModel: Story = {
  args: {
    ...baseActions,
    settings: { ...configuredSettings, tts_model: 'tts-1-hd' },
    isSaving: false,
    error: null,
    successMessage: null,
  },
};

/** Working directory empty — placeholder helper text visible. */
export const EmptyWorkdir: Story = {
  args: {
    ...baseActions,
    settings: { ...configuredSettings, working_directory: '' },
    isSaving: false,
    error: null,
    successMessage: null,
  },
};

/** Processing section with non-default values — tts_speed 1.0, tts_concurrency 10. */
export const ProcessingSection: Story = {
  args: {
    ...baseActions,
    settings: {
      ...configuredSettings,
      tts_speed: 1.0,
      tts_concurrency: 10,
      openai_timeout: 120,
      openai_retries: 5,
      audio_chunk_duration: 600,
    },
    isSaving: false,
    error: null,
    successMessage: null,
  },
};
