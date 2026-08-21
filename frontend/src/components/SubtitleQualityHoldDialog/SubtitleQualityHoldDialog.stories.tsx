import type { Meta, StoryObj } from '@storybook/react-vite';
import { expect, fn, userEvent, within } from 'storybook/test';
import { SubtitleQualityHoldDialog } from './SubtitleQualityHoldDialog';

const meta: Meta<typeof SubtitleQualityHoldDialog> = {
  title: 'Components/SubtitleQualityHoldDialog',
  component: SubtitleQualityHoldDialog,
  parameters: { layout: 'fullscreen' },
};

export default meta;
type Story = StoryObj<typeof SubtitleQualityHoldDialog>;

export const Default: Story = {
  args: {
    videoFilename: 'lesson.mp4',
    issues: [
      {
        rule_id: 'known_hallucination_phrase',
        label: 'Known STT phrase',
        message: "contains common STT hallucination phrase 'thank you for watching'",
        segment_index: 3,
      },
    ],
    onCancel: fn(),
    onRetry: fn(),
    onEdit: fn(),
    onIgnore: fn(),
  },
  play: async ({ args, canvasElement }) => {
    const canvas = within(canvasElement);
    await expect(canvas.getByText('Subtitle review required')).toBeVisible();
    await expect(canvas.getByRole('button', { name: /Retry transcription/ })).toBeVisible();
    await expect(canvas.getByRole('button', { name: /Edit subtitles/ })).toBeVisible();
    await userEvent.click(canvas.getByRole('button', { name: /Continue anyway/ }));
    await expect(args.onIgnore).toHaveBeenCalledOnce();
  },
};
