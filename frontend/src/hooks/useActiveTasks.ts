import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import type { TaskStatus } from '../types';

export interface ActiveTasksResult {
  activeTasks: TaskStatus[];
  activeCount: number;
  hasActive: boolean;
}

const RECENT_FAILURE_WINDOW_MS = 5 * 60 * 1000;

/**
 * Polls GET /api/tasks every 3 seconds when work is running, every 10s otherwise.
 * Returns active (queued/running) tasks plus any failed tasks from the last 5 minutes.
 */
export const useActiveTasks = (): ActiveTasksResult => {
  const { data } = useQuery<TaskStatus[]>({
    queryKey: ['tasks'],
    queryFn: async () => {
      const { data } = await apiClient.get('/tasks');
      return data;
    },
    refetchInterval: (query) => {
      const tasks: TaskStatus[] | undefined = query.state.data;
      if (!tasks) return 10000;
      const hasTTSRunning = tasks.some(
        (t) => t.status === 'running' && t.stage?.toLowerCase().includes('tts')
      );
      const hasActive = tasks.some(
        (t) => t.status === 'queued' || t.status === 'running'
      );
      if (hasTTSRunning) return 1000;  // 1s during TTS — most visually active stage
      if (hasActive) return 3000;
      return 10000;
    },
  });

  const now = Date.now();
  const activeTasks = (data ?? []).filter((t) => {
    if (t.status === 'queued' || t.status === 'running') return true;
    if (t.status === 'failed') {
      const created = new Date(t.created_at).getTime();
      return now - created < RECENT_FAILURE_WINDOW_MS;
    }
    return false;
  });

  const activeCount = (data ?? []).filter(
    (t) => t.status === 'queued' || t.status === 'running'
  ).length;

  return {
    activeTasks,
    activeCount,
    hasActive: activeCount > 0,
  };
};
