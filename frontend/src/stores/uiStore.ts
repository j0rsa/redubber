/**
 * Zustand store for UI state
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIState {
  // Current project
  currentProjectId: number | null;
  setCurrentProjectId: (id: number | null) => void;

  // Modal visibility
  voiceRefinementModalOpen: boolean;
  setVoiceRefinementModalOpen: (open: boolean) => void;

  // Filters
  fileFilters: {
    search: string;
    language?: string;
    status?: string;
  };
  setFileFilters: (filters: Partial<UIState['fileFilters']>) => void;

  // Per-project hide-completed preference
  hideCompletedByProject: Record<number, boolean>;
  setHideCompleted: (projectId: number, hide: boolean) => void;
  getHideCompleted: (projectId: number) => boolean;

  // Theme
  theme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark') => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      // Current project
      currentProjectId: null,
      setCurrentProjectId: (id) => set({ currentProjectId: id }),

      // Modal visibility
      voiceRefinementModalOpen: false,
      setVoiceRefinementModalOpen: (open) => set({ voiceRefinementModalOpen: open }),

      // Filters
      fileFilters: {
        search: '',
      },
      setFileFilters: (filters) =>
        set((state) => ({
          fileFilters: { ...state.fileFilters, ...filters },
        })),

      // Per-project hide-completed
      hideCompletedByProject: {},
      setHideCompleted: (projectId, hide) =>
        set((state) => ({
          hideCompletedByProject: { ...state.hideCompletedByProject, [projectId]: hide },
        })),
      getHideCompleted: (projectId) =>
        (useUIStore.getState().hideCompletedByProject[projectId] ?? false),

      // Theme
      theme: 'light',
      setTheme: (theme) => set({ theme }),
    }),
    {
      name: 'redubber-ui-state',
      partialize: (state) => ({
        currentProjectId: state.currentProjectId,
        theme: state.theme,
        hideCompletedByProject: state.hideCompletedByProject,
      }),
    }
  )
);
