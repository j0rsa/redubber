import type { Meta, StoryObj } from '@storybook/react-vite';

const meta: Meta = {
  title: 'Components/InstallPrompt',
  parameters: {
    docs: {
      description: {
        component: 'PWA install prompt that appears after 2 visits.',
      },
    },
  },
};

export default meta;

// Mock the component for Storybook
const InstallPromptMock = ({ onInstall, onDismiss }: { onInstall?: () => void, onDismiss?: () => void }) => (
  <div style={{
    position: 'fixed',
    bottom: '20px',
    left: '50%',
    transform: 'translateX(-50%)',
    background: '#1976d2',
    color: 'white',
    padding: '16px 24px',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    zIndex: 1000
  }}>
    <p style={{ margin: 0, fontSize: '14px' }}>Install Redubber for quick access</p>
    <button
      onClick={onInstall}
      style={{
        background: 'white',
        color: '#1976d2',
        border: 'none',
        padding: '8px 16px',
        borderRadius: '4px',
        cursor: 'pointer',
        fontWeight: 600
      }}
    >
      Install
    </button>
    <button
      onClick={onDismiss}
      style={{
        background: 'transparent',
        color: 'white',
        border: '1px solid white',
        padding: '8px 16px',
        borderRadius: '4px',
        cursor: 'pointer'
      }}
    >
      Not now
    </button>
  </div>
);

type Story = StoryObj<typeof InstallPromptMock>;

export const Default: Story = {
  render: () => (
    <InstallPromptMock
      onInstall={() => console.log('Install clicked')}
      onDismiss={() => console.log('Dismiss clicked')}
    />
  ),
};
