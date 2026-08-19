import type { Meta, StoryObj } from '@storybook/react-vite';
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
