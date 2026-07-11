import type { Preview } from '@storybook/react-vite';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import '../src/index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false, staleTime: Infinity },
  },
});

const preview: Preview = {
  parameters: {
    controls: {
      matchers: {
       color: /(background|color)$/i,
       date: /Date$/i,
      },
    },
    backgrounds: {
      default: 'light',
      values: [
        { name: 'light', value: '#ffffff' },
        { name: 'light-gray', value: '#f5f5f5' },
        { name: 'dark', value: '#121212' },
        { name: 'dark-elevated', value: '#1e1e1e' },
      ],
    },
    viewport: {
      viewports: {
        desktop: {
          name: 'Desktop (1440px)',
          styles: {
            width: '1440px',
            height: '900px',
          },
        },
        desktopLarge: {
          name: 'Desktop Large (1920px)',
          styles: {
            width: '1920px',
            height: '1080px',
          },
        },
        laptop: {
          name: 'Laptop (1280px)',
          styles: {
            width: '1280px',
            height: '800px',
          },
        },
        tablet: {
          name: 'iPad (768px)',
          styles: {
            width: '768px',
            height: '1024px',
          },
        },
        tabletLandscape: {
          name: 'iPad Landscape (1024px)',
          styles: {
            width: '1024px',
            height: '768px',
          },
        },
        mobile1: {
          name: 'iPhone 12 Pro (390px)',
          styles: {
            width: '390px',
            height: '844px',
          },
        },
        mobile2: {
          name: 'iPhone SE (375px)',
          styles: {
            width: '375px',
            height: '667px',
          },
        },
        mobile3: {
          name: 'Samsung Galaxy S21 (360px)',
          styles: {
            width: '360px',
            height: '800px',
          },
        },
      },
      defaultViewport: 'desktop',
    },
    a11y: {
      test: 'todo'
    }
  },
  decorators: [
    (Story, context) => {
      const isFullscreen = context.parameters.layout === 'fullscreen';

      return (
        <QueryClientProvider client={queryClient}>
          <BrowserRouter>
            <div style={{
              padding: isFullscreen ? 0 : '2rem',
              minHeight: '100vh',
              width: '100%',
              boxSizing: 'border-box',
              background: isFullscreen ? 'transparent' : 'transparent',
              display: isFullscreen ? 'flex' : 'block',
              flexDirection: 'column',
            }}>
              <Story />
            </div>
          </BrowserRouter>
        </QueryClientProvider>
      );
    },
  ],
};

export default preview;