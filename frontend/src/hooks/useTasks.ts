import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import type { TaskStatus, RedubRequest } from '../types';

export const useTasks = () => {
  return useQuery<TaskStatus[]>({
    queryKey: ['tasks'],
    queryFn: async () => {
      const { data } = await apiClient.get('/tasks');
      return data;
    }
  });
};

export const useTask = (taskId: string | null) => {
  return useQuery<TaskStatus>({
    queryKey: ['task', taskId],
    queryFn: async () => {
      if (!taskId) throw new Error('No task ID');
      const { data } = await apiClient.get(`/tasks/${taskId}`);
      return data;
    },
    enabled: !!taskId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      // Poll every 2 seconds if running or queued
      return status === 'running' || status === 'queued' ? 2000 : false;
    }
  });
};

export const useSubmitRedub = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (request: RedubRequest) => {
      const { data } = await apiClient.post('/redub', request);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    }
  });
};

export const useCancelTask = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (taskId: string) => {
      const { data } = await apiClient.post(`/tasks/${taskId}/cancel`);
      return data;
    },
    onSuccess: (_, taskId) => {
      queryClient.invalidateQueries({ queryKey: ['task', taskId] });
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    }
  });
};
