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

async function waitForScanIdle(projectId: number, timeoutMs = 120_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const { data } = await apiClient.get<{ status: string }>(
      `/projects/${projectId}/scan`,
    );
    if (data.status !== 'running') {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

export const useScanVideos = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (projectId: number) => {
      const { data } = await apiClient.post(`/projects/${projectId}/scan`);
      // Wait until background scan finishes so duration/size aggregates are ready.
      await waitForScanIdle(projectId);
      return data;
    },
    onSuccess: async (_, projectId) => {
      await queryClient.invalidateQueries({ queryKey: ['videos', projectId] });
      await queryClient.invalidateQueries({ queryKey: ['projects'] });
      await queryClient.invalidateQueries({ queryKey: ['project', projectId] });
    },
  });
};
