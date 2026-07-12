import { useState, useCallback } from 'react';
import { apiClient } from '../api/client';
import type { FileNode } from '../components/FileBrowser';

interface UseFileBrowserReturn {
  currentPath: string;
  nodes: FileNode[];
  isLoading: boolean;
  error: string | null;
  navigate: (path: string) => Promise<void>;
  navigateUp: () => void;
  canNavigateUp: boolean;
}

export const useFileBrowser = (initialPath: string = '/'): UseFileBrowserReturn => {
  const [currentPath, setCurrentPath] = useState(initialPath);
  const [nodes, setNodes] = useState<FileNode[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const navigate = useCallback(async (path: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const { data } = await apiClient.get('/filesystem/browse', {
        params: { path },
      });
      setCurrentPath(data.path);
      setNodes(data.nodes);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load directory';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const navigateUp = useCallback(() => {
    const parent = currentPath.split('/').slice(0, -1).join('/') || '/';
    navigate(parent);
  }, [currentPath, navigate]);

  const canNavigateUp = currentPath !== '/' && currentPath.split('/').length > 1;

  return { currentPath, nodes, isLoading, error, navigate, navigateUp, canNavigateUp };
};
