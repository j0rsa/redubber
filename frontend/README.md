# Redubber v2.0 Frontend

React + TypeScript + Vite PWA for managing video redubbing projects.

## Features

- **Project Management**: Create and browse redubbing projects
- **Video Discovery**: Scan directories for video files
- **Pipeline Monitoring**: Real-time progress tracking with automatic polling
- **Task Management**: Submit redub jobs and monitor their status

## Tech Stack

- **React 19** with TypeScript 6
- **React Router 7** for routing
- **TanStack Query 5** for server state management
- **Zustand** for UI state (theme, current project)
- **Axios** for HTTP client
- **Vite 8** for build tooling

## Development

```bash
# Install dependencies
npm install

# Start dev server (runs on localhost:5173)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

## Architecture

### Directory Structure

```
src/
├── api/           # API client configuration
│   └── client.ts
├── hooks/         # React Query hooks
│   ├── useProjects.ts
│   ├── useVideos.ts
│   └── useTasks.ts
├── pages/         # Route components
│   ├── ProjectHub.tsx
│   ├── ProjectDetail.tsx
│   └── JobMonitor.tsx
├── components/    # Reusable UI components
│   ├── FileGrid.tsx
│   └── PipelineStatus.tsx
├── stores/        # Zustand stores
│   └── uiStore.ts
├── types/         # TypeScript type definitions
│   └── index.ts
└── main.tsx       # App entry point
```

### Routes

- `/` - Project hub (list/create projects)
- `/project/:id` - Project detail (video list, scan, redub)
- `/job/:taskId` - Job monitor (real-time task status)

### API Integration

The frontend connects to the FastAPI backend via Vite proxy:
- Frontend: `http://localhost:5173`
- API proxy: `/api` → `http://localhost:8000/api`

### State Management

**React Query** manages server state:
- Automatic caching and background refetching
- Optimistic updates for mutations
- Polling for active jobs (2s interval)

**Zustand** manages UI state:
- Current project ID
- Theme (light/dark)
- File filters
- Modal visibility

### Type Safety

All API responses are strongly typed using TypeScript interfaces defined in `src/types/index.ts`. The build will fail if types don't match.

## Key Components

### FileGrid

Table displaying video files with:
- Metadata (duration, size, audio streams, subtitles)
- Pipeline status badges
- Redub action buttons

### PipelineStatus

Progress bar component showing:
- Current stage (audio extraction, transcription, etc.)
- Progress percentage
- Status badge (pending/in progress/complete)
- Detailed metrics (chunks, transcripts, TTS segments)

## React Query Hooks

### useProjects

```typescript
const { data: projects } = useProjects();
const { data: project } = useProject(id);
const createProject = useCreateProject();
```

### useVideos

```typescript
const { data: videos } = useVideos(projectId);
const scanVideos = useScanVideos();
```

### useTasks

```typescript
const { data: tasks } = useTasks();
const { data: task } = useTask(taskId);  // Auto-polls if running
const submitRedub = useSubmitRedub();
const cancelTask = useCancelTask();
```

## API Endpoints Used

- `GET /api/projects` - List projects
- `POST /api/projects` - Create project
- `GET /api/projects/:id` - Get project details
- `GET /api/projects/:id/videos` - List videos
- `POST /api/projects/:id/scan` - Scan for videos
- `POST /api/redub` - Submit redub job
- `GET /api/tasks` - List tasks
- `GET /api/tasks/:id` - Get task status
- `POST /api/tasks/:id/cancel` - Cancel task

## Performance Optimizations

1. **Automatic polling**: Tasks poll every 2s only when status is `running` or `queued`
2. **Stale time**: 5s default to reduce unnecessary refetches
3. **Disabled window focus refetch**: Prevents jarring refetches on tab switch
4. **Query invalidation**: Mutations automatically invalidate related queries

## Future Enhancements

- [ ] Add voice settings modal
- [ ] Implement file filters (search, language, status)
- [ ] Add dark mode theme toggle
- [ ] PWA offline support
- [ ] Batch operations
- [ ] Download completed videos
