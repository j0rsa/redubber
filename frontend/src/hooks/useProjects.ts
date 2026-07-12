import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import type { Project, ProjectCreate } from '../types';

export const useProjects = () => {
  return useQuery<Project[]>({
    queryKey: ['projects'],
    queryFn: async () => {
      const { data } = await apiClient.get('/projects');
      return data;
    }
  });
};

export const useProject = (id: number | null) => {
  return useQuery<Project>({
    queryKey: ['project', id],
    queryFn: async () => {
      if (!id) throw new Error('No project ID');
      const { data } = await apiClient.get(`/projects/${id}`);
      return data;
    },
    enabled: !!id
  });
};

export const useCreateProject = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (project: ProjectCreate) => {
      const { data } = await apiClient.post('/projects', project);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    }
  });
};

export const useDeleteProject = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (projectId: number) => {
      await apiClient.delete(`/projects/${projectId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    }
  });
};
