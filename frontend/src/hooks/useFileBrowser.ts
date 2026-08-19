import { useState, useCallback, useEffect, useRef } from 'react';
import { apiClient } from '../api/client';
import type { FileNode } from '../components/FileBrowser';

interface UseFileBrowserReturn {
  currentPath: string;
  nodes: FileNode[];
  isLoading: boolean;
  isSearching: boolean;
  error: string | null;
  searchQuery: string;
  isSearchMode: boolean;
  navigate: (path: string) => Promise<void>;
  navigateUp: () => void;
  canNavigateUp: boolean;
  setSearchQuery: (query: string) => void;
  clearSearch: () => void;
}

const SEARCH_DEBOUNCE_MS = 300;

export const useFileBrowser = (
  initialPath: string = '/',
  searchRoot: string = initialPath,
): UseFileBrowserReturn => {
  const [currentPath, setCurrentPath] = useState(initialPath);
  const [nodes, setNodes] = useState<FileNode[]>([]);
  const [searchResults, setSearchResults] = useState<FileNode[]>([]);
  const [searchQuery, setSearchQueryState] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const searchDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const searchRequestIdRef = useRef(0);

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

  const runSearch = useCallback(async (query: string) => {
    const trimmed = query.trim();
    if (!trimmed) {
      setSearchResults([]);
      setIsSearching(false);
      return;
    }

    const requestId = ++searchRequestIdRef.current;
    setIsSearching(true);
    setError(null);

    try {
      const { data } = await apiClient.get('/filesystem/search', {
        params: {
          q: trimmed,
          root: searchRoot,
        },
      });

      if (requestId !== searchRequestIdRef.current) {
        return;
      }

      setSearchResults(data.nodes);
    } catch (err: unknown) {
      if (requestId !== searchRequestIdRef.current) {
        return;
      }

      const msg = err instanceof Error ? err.message : 'Failed to search folders';
      setError(msg);
      setSearchResults([]);
    } finally {
      if (requestId === searchRequestIdRef.current) {
        setIsSearching(false);
      }
    }
  }, [searchRoot]);

  const setSearchQuery = useCallback((query: string) => {
    setSearchQueryState(query);

    if (searchDebounceRef.current) {
      clearTimeout(searchDebounceRef.current);
    }

    const trimmed = query.trim();
    if (!trimmed) {
      searchRequestIdRef.current += 1;
      setSearchResults([]);
      setIsSearching(false);
      return;
    }

    setIsSearching(true);
    searchDebounceRef.current = setTimeout(() => {
      void runSearch(trimmed);
    }, SEARCH_DEBOUNCE_MS);
  }, [runSearch]);

  const clearSearch = useCallback(() => {
    if (searchDebounceRef.current) {
      clearTimeout(searchDebounceRef.current);
    }
    searchRequestIdRef.current += 1;
    setSearchQueryState('');
    setSearchResults([]);
    setIsSearching(false);
  }, []);

  useEffect(() => {
    return () => {
      if (searchDebounceRef.current) {
        clearTimeout(searchDebounceRef.current);
      }
    };
  }, []);

  const navigateUp = useCallback(() => {
    clearSearch();
    const parent = currentPath.split('/').slice(0, -1).join('/') || '/';
    void navigate(parent);
  }, [clearSearch, currentPath, navigate]);

  const canNavigateUp = currentPath !== '/' && currentPath.split('/').length > 1;
  const isSearchMode = searchQuery.trim().length > 0;

  return {
    currentPath,
    nodes: isSearchMode ? searchResults : nodes,
    isLoading,
    isSearching,
    error,
    searchQuery,
    isSearchMode,
    navigate: async (path: string) => {
      clearSearch();
      await navigate(path);
    },
    navigateUp,
    canNavigateUp,
    setSearchQuery,
    clearSearch,
  };
};
