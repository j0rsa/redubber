import type { Meta, StoryObj } from '@storybook/react-vite';
import { Settings } from './Settings';
import type { SettingsProps } from './Settings';
import type { SettingsData, HallucinationRuleSetting } from '../../types/settings';
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

const sampleHallucinationRules: HallucinationRuleSetting[] = [
  {
    id: 'known_hallucination_phrase',
    label: 'Known STT phrase',
    description: 'Flag cues that contain common Whisper boilerplate such as “thanks for watching”.',
    enabled: true,
    threshold: null,
    default_threshold: null,
    threshold_min: null,
    threshold_max: null,
    threshold_step: null,
    unit: null,
    comparison: null,
  },
  {
    id: 'excessive_cps',
    label: 'Text too dense',
    description: 'Flag a cue whose text is too dense for its duration.',
    enabled: true,
    threshold: 40,
    default_threshold: 40,
    threshold_min: 5,
    threshold_max: 200,
    threshold_step: 1,
    unit: 'chars/s',
    comparison: 'gt',
  },
  {
    id: 'consecutive_duplicate_segments',
    label: 'Consecutive duplicates',
    description: 'Flag consecutive cues that repeat the exact same text.',
    enabled: false,
    threshold: 3,
    default_threshold: 3,
    threshold_min: 2,
    threshold_max: 20,
    threshold_step: 1,
    unit: 'cues',
    comparison: 'min_count',
  },
  {
    id: 'dominant_word_loop',
    label: 'Dominant word loop',
    description: 'Flag a transcript where one word makes up too large a share of all tokens.',
    enabled: true,
    threshold: 0.5,
    default_threshold: 0.38,
    threshold_min: 0.1,
    threshold_max: 1,
    threshold_step: 0.01,
    unit: 'ratio',
    comparison: 'gte',
  },
];

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
  hallucination_rules: sampleHallucinationRules,
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

/** Hallucination rules with a disabled rule and a custom threshold. */
export const HallucinationRules: Story = {
  args: {
    ...baseActions,
    settings: configuredSettings,
    isSaving: false,
    error: null,
    successMessage: null,
  },
};
