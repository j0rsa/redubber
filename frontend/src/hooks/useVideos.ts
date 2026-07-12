import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import type { VideoFile } from '../types';

export const useVideos = (projectId: number | null, hasRunningJobs = false) => {
  return useQuery<VideoFile[]>({
    queryKey: ['videos', projectId],
    queryFn: async () => {
      if (!projectId) throw new Error('No project ID');
      const { data } = await apiClient.get(`/projects/${projectId}/videos`);
      return data;
    },
    enabled: !!projectId,
    // Poll every 3s while jobs are running so pipeline_status stays current
    refetchInterval: hasRunningJobs ? 3000 : false,
  });
};

export const useScanVideos = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (projectId: number) => {
      const { data } = await apiClient.post(`/projects/${projectId}/scan`);
      return data;
    },
    onSuccess: (_, projectId) => {
      queryClient.invalidateQueries({ queryKey: ['videos', projectId] });
    }
  });
};
