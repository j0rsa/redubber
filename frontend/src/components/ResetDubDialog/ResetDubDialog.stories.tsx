import type { Meta, StoryObj } from '@storybook/react-vite';
import { expect, userEvent, within } from 'storybook/test';
import { ResetDubDialog } from './ResetDubDialog';

const meta: Meta<typeof ResetDubDialog> = {
  title: 'Components/ResetDubDialog',
  component: ResetDubDialog,
  parameters: { layout: 'fullscreen' },
};

export default meta;
type Story = StoryObj<typeof ResetDubDialog>;

export const Default: Story = {
  args: {
    videoFilename: '01.mp4',
    currentStage: 'complete',
    onCancel: () => console.log('cancel'),
    onConfirm: (resetTo) => console.log('reset to', resetTo),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await expect(canvas.queryByRole('slider')).not.toBeInTheDocument();
    await userEvent.click(canvas.getByRole('button', { name: 'Mix' }));
    await expect(
      canvas.getByRole('button', { name: 'Mix' }),
    ).toHaveAttribute('aria-current', 'step');
    await expect(
      canvas.getByRole('button', { name: 'Reset to Mix' }),
    ).toBeEnabled();
  },
};

export const Submitting: Story = {
  args: {
    ...Default.args,
    isSubmitting: true,
  },
};

export const WithError: Story = {
  args: {
    ...Default.args,
    errorMessage: 'Video is not in the final redubbed state.',
  },
};

export const BulkSelection: Story = {
  args: {
    ...Default.args,
    videoFilename: '',
    selectionCount: 3,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await expect(canvas.getByText('3 selected videos')).toBeVisible();
    await expect(
      canvas.getByText(/Each video file is always reverted/),
    ).toBeVisible();
  },
};
