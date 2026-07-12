import type { Meta, StoryObj } from '@storybook/react-vite';

const meta: Meta = {
  title: 'Components/OfflineBanner',
  parameters: {
    docs: {
      description: {
        component: 'Fixed banner that appears when the user is offline.',
      },
    },
  },
};

export default meta;

// Mock the component for Storybook since it relies on a hook
const OfflineBannerMock = () => (
  <div
    style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      background: '#dc3545',
      color: 'white',
      padding: '12px',
      textAlign: 'center',
      fontSize: '14px',
      fontWeight: 'bold',
      zIndex: 9999,
      boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
    }}
  >
    You're offline. Some features may not be available.
  </div>
);

type Story = StoryObj<typeof OfflineBannerMock>;

export const Visible: Story = {
  render: () => <OfflineBannerMock />,
};
