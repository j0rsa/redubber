import type { Meta, StoryObj } from '@storybook/react-vite';
import { QualityWarningBadge } from './QualityWarningBadge';

const meta: Meta<typeof QualityWarningBadge> = {
  title: 'Components/QualityWarningBadge',
  component: QualityWarningBadge,
};

export default meta;
type Story = StoryObj<typeof QualityWarningBadge>;

const issues = [
  {
    rule_id: 'known_hallucination_phrase',
    label: 'Known STT phrase',
    message: "contains common STT hallucination phrase(s): 'thank you for watching'",
    segment_index: 3,
  },
  {
    rule_id: 'excessive_cps',
    label: 'Text too dense',
    message: '45.0 chars/s (>40.0) — text too dense for duration',
    segment_index: 3,
  },
  {
    rule_id: 'transcript_too_dense',
    label: 'Transcript too dense',
    message: 'average 42 chars/s across the file',
    segment_index: null,
  },
];

export const Default: Story = {
  args: { issues },
};

export const Compact: Story = {
  args: { issues, compact: true },
};
