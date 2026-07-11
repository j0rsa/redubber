import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ProjectHub } from './pages/ProjectHub';
import { NewProject } from './pages/NewProject';
import { ProjectDetail } from './pages/ProjectDetail';
import { JobMonitor } from './pages/JobMonitor';
import { OfflineBanner } from './components/OfflineBanner';
import { TasksContainer } from './components/TasksPanel/TasksContainer';
import { registerServiceWorker } from './serviceWorkerRegistration';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5000,
    },
  },
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <OfflineBanner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<ProjectHub />} />
          <Route path="/new-project" element={<NewProject />} />
          <Route path="/project/:id" element={<ProjectDetail />} />
          <Route path="/job/:taskId" element={<JobMonitor />} />
        </Routes>
        <TasksContainer />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>
);

// Register service worker for PWA functionality
registerServiceWorker();
