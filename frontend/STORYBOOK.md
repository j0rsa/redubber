# Storybook Documentation

## What is Storybook?

Storybook is an interactive development environment and documentation tool for UI components. It allows you to:

- **Develop components in isolation** - Build and test components without running the entire application
- **Document component APIs** - Automatically generate documentation from component props and stories
- **Test different states** - Visualize all possible states of components (loading, error, success, etc.)
- **Share with team** - Create a living component library that designers and developers can reference

## Why We Use Storybook

1. **Faster development** - Work on individual components without complex setup
2. **Better testing** - Easily test edge cases and different prop combinations
3. **Living documentation** - Self-updating docs that always match the codebase
4. **Design system** - Central place to see all UI components and their variants
5. **Quality assurance** - Visual regression testing and accessibility checks

## Getting Started

### Running Storybook

```bash
cd /Users/abochev/code/redubber/frontend
npm run storybook
```

This will start Storybook at [http://localhost:6006](http://localhost:6006)

### Building Static Storybook

```bash
npm run build-storybook
```

This creates a static build in `storybook-static/` that can be deployed anywhere.

## Design System

### Theme

Our design system is defined in `src/styles/theme.ts` and includes:

#### Colors

- **Primary**: `#1976d2` (Blue)
- **Primary Hover**: `#1565c0`
- **Secondary**: `#dc004e` (Pink)
- **Success**: `#2e7d32` (Green)
- **Warning**: `#ed6c02` (Orange)
- **Error**: `#d32f2f` (Red)
- **Background**: `#ffffff` (White)
- **Background Dark**: `#f5f5f5` (Light Gray)
- **Text**: `#212121` (Dark Gray)
- **Text Secondary**: `#757575` (Medium Gray)

#### Gradients

We use modern purple-blue gradients for premium feel:

- **Button/Header Gradient**: `linear-gradient(135deg, #667eea 0%, #764ba2 100%)`

#### Spacing

- **xs**: 4px
- **sm**: 8px
- **md**: 16px
- **lg**: 24px
- **xl**: 32px

#### Border Radius

- **sm**: 4px
- **md**: 8px
- **lg**: 12px

#### Typography

- **Font Family**: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif`
- **H1**: 32px, weight 700
- **H2**: 24px, weight 600
- **H3**: 20px, weight 600
- **Body**: 14px, weight 400
- **Small**: 12px, weight 400

## Component Overview

### FileGrid

**Path**: `src/components/FileGrid.tsx`

Displays video files in a table with metadata, audio streams, subtitles, and pipeline status.

**Features**:
- Purple gradient header
- Hover effects on rows
- Badge styling for audio streams and subtitles
- Integrated pipeline status display
- Action buttons with gradient styling

**Stories**:
- Default - Single video
- WithProgress - Video being processed
- Completed - Finished video with multiple audio streams
- MultipleVideos - Mixed states
- ManyVideos - 10+ videos with random progress
- NoVideos - Empty state
- WithoutActions - Read-only view
- LargeFile - 8GB+ file with multiple streams

### PipelineStatus

**Path**: `src/components/PipelineStatus.tsx`

Real-time progress indicator showing pipeline stage and statistics.

**Features**:
- Animated progress bar with gradient
- Stage chips with pulse animation for running state
- Statistics display (audio chunks, transcripts, TTS segments)
- Color-coded states (pending, running, completed)

**Stories**:
- Pending - Not started
- Starting - Just beginning (5%)
- ExtractingAudio - Audio extraction phase
- Transcribing - Transcription phase (35%)
- Translating - Translation phase (55%)
- GeneratingTTS - TTS generation phase (65%)
- Finalizing - Final processing (90%)
- Complete - Finished (100%)
- AlmostComplete - 99% done
- MinimalData - Without statistics
- WithExternalSubs - Loading external subtitles

### OfflineBanner

**Path**: `src/components/OfflineBanner.tsx`

Fixed top banner that appears when user goes offline.

**Features**:
- Red background (#dc3545)
- Fixed positioning at top
- Only visible when offline (PWA feature)

**Stories**:
- Visible - Shows the offline banner

### InstallPrompt

**Path**: `src/components/InstallPrompt.tsx`

PWA installation prompt that appears after 2 visits.

**Features**:
- Blue gradient background (#1976d2)
- Fixed bottom positioning
- Install and dismiss buttons
- Auto-triggers via `beforeinstallprompt` event

**Stories**:
- Default - Shows the install prompt

### ProjectHub (Page)

**Path**: `src/pages/ProjectHub.tsx`

Main landing page for creating and selecting projects.

**Features**:
- Project creation form
- Grid of existing projects
- Hover effects on project cards
- Timestamps display

**Stories**:
- Default - 3 projects
- Empty - No projects yet
- Loading - Loading state
- SingleProject - One project
- ManyProjects - 10 projects

## CSS Modules

We use CSS modules for scoped styling to prevent class name conflicts.

### FileGrid.module.css

- `.fileGrid` - Main container with shadow and rounded corners
- `.table` - Full-width table
- `.header` - Purple gradient header
- `.row` - Table row with hover effect
- `.cell` - Table cell with padding
- `.badge` - Colored badge for languages/codecs
- `.actionButton` - Gradient button with hover animation

### PipelineStatus.module.css

- `.container` - White card with shadow
- `.header` - Flex header with stage and percentage
- `.progressBar` - Gray background bar
- `.progressFill` - Gradient progress fill
- `.stageChip` - Rounded chip for stage status
- `.stats` - Flex container for statistics
- `@keyframes pulse` - Pulsing animation for running state

## Key Design Patterns

### Modern Aesthetics

1. **Gradients**: Use purple-blue gradients (#667eea → #764ba2) for primary elements
2. **Shadows**: Subtle shadows (0 2px 8px rgba(0,0,0,0.1)) for depth
3. **Rounded Corners**: 8px-12px for modern feel
4. **Hover Effects**: Transform and shadow on buttons/cards
5. **Smooth Transitions**: 0.2s-0.3s ease transitions

### Interactive Elements

1. **Hover States**: All clickable elements have hover effects
2. **Disabled States**: 50% opacity for disabled buttons
3. **Loading States**: Pulse animation for in-progress items
4. **Color Coding**: Different colors for different states (pending/running/completed)

### Responsive Design

- Tables with horizontal scroll
- Flex layouts that wrap on smaller screens
- Max-width containers (1200px) for readability

## Adding New Stories

### Step 1: Create the Story File

Create a file named `ComponentName.stories.tsx` next to your component:

```typescript
import type { Meta, StoryObj } from '@storybook/react-vite';
import { YourComponent } from './YourComponent';

const meta: Meta<typeof YourComponent> = {
  title: 'Category/YourComponent',
  component: YourComponent,
  parameters: {
    docs: {
      description: {
        component: 'Description of what this component does.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof YourComponent>;

export const Default: Story = {
  args: {
    // Your component props here
  },
};
```

### Step 2: Add Multiple States

Create stories for different states:

```typescript
export const Loading: Story = {
  args: {
    isLoading: true,
  },
};

export const Error: Story = {
  args: {
    error: 'Something went wrong',
  },
};

export const WithData: Story = {
  args: {
    data: [...mockData],
  },
};
```

### Step 3: Use Decorators (if needed)

Wrap stories with context providers:

```typescript
export const WithRouter: Story = {
  decorators: [
    (Story) => (
      <BrowserRouter>
        <Story />
      </BrowserRouter>
    ),
  ],
};
```

## Storybook Addons

### Installed Addons

1. **@storybook/addon-docs** - Auto-generated documentation
2. **@storybook/addon-a11y** - Accessibility testing
3. **@storybook/addon-vitest** - Vitest integration for component tests
4. **@chromatic-com/storybook** - Visual regression testing
5. **@storybook/addon-mcp** - MCP integration

### Addon Controls

Use the Controls panel in Storybook UI to:
- Change prop values in real-time
- Test different combinations
- See how components respond to different inputs

### Accessibility Panel

The a11y addon automatically checks for:
- Color contrast issues
- Missing alt text
- Keyboard navigation
- ARIA labels
- Semantic HTML

## Best Practices

### 1. Name Stories Clearly

```typescript
// Good
export const WithLongContent: Story = { ... }
export const LoadingState: Story = { ... }

// Bad
export const Story1: Story = { ... }
export const Test: Story = { ... }
```

### 2. Use Mock Data Generators

```typescript
const createMockVideo = (overrides: Partial<VideoFile> = {}): VideoFile => ({
  id: 1,
  filename: 'video.mp4',
  ...overrides,
});
```

### 3. Document Complex Props

Use the `description` parameter:

```typescript
export const ComplexExample: Story = {
  args: { ... },
  parameters: {
    docs: {
      description: {
        story: 'This story demonstrates how the component handles multiple audio streams...',
      },
    },
  },
};
```

### 4. Test Edge Cases

Create stories for:
- Empty states
- Loading states
- Error states
- Maximum data scenarios
- Minimum data scenarios
- Long text/overflow scenarios

### 5. Keep Stories Independent

Each story should work in isolation without depending on other stories.

## Troubleshooting

### Stories Not Showing Up

1. Check file naming: Must end with `.stories.tsx`
2. Check export: Must have `export default meta`
3. Restart Storybook: `Ctrl+C` and `npm run storybook`

### TypeScript Errors

1. Make sure types are imported from correct paths
2. Use `type` instead of `interface` for story types
3. Check that component props match story args

### Styling Issues

1. Import CSS modules: `import styles from './Component.module.css'`
2. Check that `index.css` is imported in `.storybook/preview.tsx`
3. Verify class names match between CSS and JSX

### Mock Data Not Working

1. Use generators for consistent data
2. Make sure mock data matches TypeScript interfaces
3. Check for required vs. optional properties

## Resources

- [Storybook Documentation](https://storybook.js.org/docs)
- [Component Story Format (CSF)](https://storybook.js.org/docs/api/csf)
- [Storybook Addons](https://storybook.js.org/addons)
- [Best Practices](https://storybook.js.org/docs/writing-stories/introduction)

## Screenshots

After running Storybook, you can find:

- **Canvas view**: Interactive component playground
- **Docs view**: Auto-generated documentation
- **Controls panel**: Real-time prop editing
- **Actions panel**: Event logging
- **Accessibility panel**: a11y violations

---

**Last Updated**: 2026-07-07  
**Storybook Version**: 10.4.6
