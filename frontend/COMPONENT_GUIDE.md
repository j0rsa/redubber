# Component Guide

Quick reference for all Storybook components and their usage.

## Component Hierarchy

```
App
├── Pages
│   ├── ProjectHub
│   │   ├── InstallPrompt
│   │   └── Project Cards
│   │
│   ├── ProjectDetail
│   │   ├── FileGrid
│   │   │   └── PipelineStatus
│   │   └── Project Settings
│   │
│   └── JobMonitor
│       └── Task Status
│
└── Global Components
    ├── OfflineBanner
    └── InstallPrompt
```

## Components

### FileGrid

**Purpose**: Display videos in a table with metadata and actions

**Props**:
```typescript
interface FileGridProps {
  videos: VideoFile[];
  onRedub?: (videoPath: string) => void;
}
```

**Features**:
- Purple gradient header
- Sortable columns (filename, duration, size)
- Audio stream badges
- Subtitle indicators
- Pipeline status integration
- Action buttons

**When to use**:
- Project detail pages
- Video list views
- Any tabular video display

**Storybook location**: `Components/FileGrid`

**Example**:
```tsx
<FileGrid
  videos={videoList}
  onRedub={(path) => handleRedub(path)}
/>
```

---

### PipelineStatus

**Purpose**: Show real-time progress of video processing pipeline

**Props**:
```typescript
interface PipelineStatusProps {
  status: PipelineStatus;
}

interface PipelineStatus {
  progress: number; // 0-100
  current_stage: string;
  is_complete: boolean;
  audio_chunks?: number;
  transcripts?: number;
  tts_segments?: number;
}
```

**Features**:
- Animated progress bar with gradient
- Stage chips (pending/running/completed)
- Statistics display
- Pulse animation for running state
- Color-coded states

**When to use**:
- Inside FileGrid for each video
- Job monitoring pages
- Any progress indication

**Storybook location**: `Components/PipelineStatus`

**Example**:
```tsx
<PipelineStatus
  status={{
    progress: 65,
    current_stage: 'Generating TTS',
    is_complete: false,
    audio_chunks: 150,
    transcripts: 150,
    tts_segments: 98,
  }}
/>
```

---

### OfflineBanner

**Purpose**: Alert users when they're offline (PWA feature)

**Props**: None (uses `useOnlineStatus` hook)

**Features**:
- Fixed top positioning
- Red background
- Auto-shows/hides based on connection
- High z-index (9999)

**When to use**:
- Root level of app
- Automatically managed

**Storybook location**: `Components/OfflineBanner`

**Example**:
```tsx
<OfflineBanner />
```

---

### InstallPrompt

**Purpose**: Prompt users to install PWA after 2 visits

**Props**: None (manages state internally)

**Features**:
- Fixed bottom positioning
- Blue gradient background
- Install/dismiss actions
- Visit tracking (localStorage)
- Native browser prompt integration

**When to use**:
- Root level of app
- ProjectHub page

**Storybook location**: `Components/InstallPrompt`

**Example**:
```tsx
<InstallPrompt />
```

---

## Pages

### ProjectHub

**Purpose**: Main landing page for project management

**Features**:
- Project creation form
- Project grid/list
- Navigation to project details
- Install prompt integration

**Sections**:
1. **Header**: "Redubber Projects"
2. **Create Form**: Path input + create button
3. **Project List**: Clickable project cards

**Storybook location**: `Pages/ProjectHub`

**States**:
- Default (with projects)
- Empty (no projects)
- Loading
- Single project
- Many projects

---

### ProjectDetail

**Purpose**: View and manage videos within a project

**Features**:
- Project header with back button
- Project settings display
- Scan videos button
- FileGrid integration
- Redub action handling

**Sections**:
1. **Header**: Project name, path, back button
2. **Settings**: Voice config, instructions
3. **Videos**: FileGrid with all videos

**Storybook location**: `Pages/ProjectDetail`

**States**:
- Default (with videos)
- Loading
- Empty project
- Single video
- Mixed progress
- Many videos

---

## Styling

### CSS Modules

All components use CSS Modules for scoped styling:

```tsx
import styles from './Component.module.css';

<div className={styles.container}>
  <button className={styles.actionButton}>
    Click me
  </button>
</div>
```

### Common Patterns

#### Conditional Classes

```tsx
<span className={`${styles.badge} ${styles.running}`}>
  Status
</span>
```

#### Inline Styles (sparingly)

```tsx
<div style={{ width: `${progress}%` }} />
```

### Theme Usage

Import theme for consistency:

```tsx
import { theme } from '../styles/theme';

<div style={{ color: theme.colors.primary }}>
  Text
</div>
```

---

## State Management

### Component State

Use `useState` for local component state:

```tsx
const [isOpen, setIsOpen] = useState(false);
```

### Global State (Zustand)

```tsx
import { useUIStore } from '../stores/uiStore';

const currentProjectId = useUIStore((state) => state.currentProjectId);
const setCurrentProjectId = useUIStore((state) => state.setCurrentProjectId);
```

### Server State (React Query)

```tsx
import { useProjects } from '../hooks/useProjects';

const { data, isLoading, error } = useProjects();
```

---

## Testing with Storybook

### Visual Testing

1. **Run Storybook**: `npm run storybook`
2. **Navigate to component**
3. **Check all stories**
4. **Use Controls panel** to test props
5. **Check Accessibility panel** for a11y issues

### Interaction Testing

Use the Controls panel to:
- Change prop values
- Test edge cases
- Verify responsive behavior
- Check different states

### Accessibility Testing

The a11y addon automatically checks:
- Color contrast
- Keyboard navigation
- ARIA labels
- Semantic HTML

---

## Common Patterns

### Loading State

```tsx
{isLoading ? (
  <div>Loading...</div>
) : (
  <Component data={data} />
)}
```

### Error State

```tsx
{error ? (
  <div style={{ color: 'red' }}>
    Error: {error.message}
  </div>
) : (
  <Component />
)}
```

### Empty State

```tsx
{items.length === 0 ? (
  <div style={{ textAlign: 'center', padding: '48px' }}>
    <p>No items found</p>
  </div>
) : (
  <ItemList items={items} />
)}
```

### Conditional Rendering

```tsx
{showDetails && <DetailPanel />}
{items.map(item => <ItemCard key={item.id} {...item} />)}
```

---

## Best Practices

### Props

1. **Use TypeScript interfaces** for type safety
2. **Provide defaults** for optional props
3. **Keep props minimal** - don't pass what you don't need
4. **Use callbacks** for actions (onRedub, onScan)

### Styling

1. **CSS Modules first** - scoped styles prevent conflicts
2. **Use theme values** - maintain consistency
3. **Avoid inline styles** - except for dynamic values
4. **Mobile-first** - design for small screens first

### Performance

1. **Memoize expensive calculations** with `useMemo`
2. **Memoize callbacks** with `useCallback`
3. **Use React.memo** for components that render often
4. **Lazy load** heavy components

### Accessibility

1. **Semantic HTML** - use correct elements
2. **ARIA labels** - for screen readers
3. **Keyboard navigation** - test without mouse
4. **Color contrast** - check in Storybook

---

## File Structure

```
src/
├── components/
│   ├── FileGrid.tsx
│   ├── FileGrid.module.css
│   ├── FileGrid.stories.tsx
│   ├── PipelineStatus.tsx
│   ├── PipelineStatus.module.css
│   ├── PipelineStatus.stories.tsx
│   ├── OfflineBanner.tsx
│   ├── OfflineBanner.stories.tsx
│   ├── InstallPrompt.tsx
│   └── InstallPrompt.stories.tsx
│
├── pages/
│   ├── ProjectHub.tsx
│   ├── ProjectHub.stories.tsx
│   ├── ProjectDetail.tsx
│   └── ProjectDetail.stories.tsx
│
├── styles/
│   └── theme.ts
│
├── types/
│   └── index.ts
│
├── hooks/
│   ├── useProjects.ts
│   ├── useVideos.ts
│   └── useTasks.ts
│
└── stores/
    └── uiStore.ts
```

---

## Quick Commands

```bash
# Run Storybook
npm run storybook

# Build Storybook (static)
npm run build-storybook

# Run tests with Vitest
npm run test

# Type check
npx tsc --noEmit

# Lint
npm run lint
```

---

## Resources

- [Storybook Docs](https://storybook.js.org/docs)
- [React Query Docs](https://tanstack.com/query/latest)
- [Zustand Docs](https://github.com/pmndrs/zustand)
- [CSS Modules Guide](https://github.com/css-modules/css-modules)

---

**Last Updated**: 2026-07-07
