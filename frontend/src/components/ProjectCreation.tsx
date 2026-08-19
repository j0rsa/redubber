import { useState, useEffect } from 'react';
import { FileBrowser, type FileNode } from './FileBrowser';
import styles from './ProjectCreation.module.css';

interface ProjectCreationProps {
  initialPath?: string;
  nodes: FileNode[];
  onLoadDirectory?: (path: string) => void;
  onCreateProject: (path: string, name: string) => void;
  onCancel?: () => void;
  isLoading?: boolean;
  isSearching?: boolean;
  searchQuery?: string;
  onSearchQueryChange?: (query: string) => void;
  isSearchMode?: boolean;
}

export const ProjectCreation = ({
  initialPath = '/',
  nodes,
  onLoadDirectory,
  onCreateProject,
  onCancel,
  isLoading = false,
  isSearching = false,
  searchQuery = '',
  onSearchQueryChange,
  isSearchMode = false,
}: ProjectCreationProps) => {
  const [selectedPath, setSelectedPath] = useState<string>('');
  const [currentPath, setCurrentPath] = useState<string>(initialPath);
  const [projectName, setProjectName] = useState<string>('');

  useEffect(() => {
    if (!isSearchMode) {
      setCurrentPath(initialPath);
    }
  }, [initialPath, isSearchMode]);

  const handleSelectPath = (path: string) => {
    setSelectedPath(path);
    setProjectName(path.split('/').filter(Boolean).pop() ?? '');
  };

  const handleNavigate = (path: string) => {
    setCurrentPath(path);
    onLoadDirectory?.(path);
  };

  const handleNavigateUp = () => {
    const parts = currentPath.split('/').filter(Boolean);
    const parent = parts.length > 0 ? '/' + parts.slice(0, -1).join('/') : '/';
    handleNavigate(parent || '/');
  };

  const handleCreate = () => {
    if (selectedPath && projectName.trim()) {
      onCreateProject(selectedPath, projectName.trim());
    }
  };

  const canNavigateUp = currentPath !== '/';
  const canCreate = !!selectedPath && projectName.trim().length > 0;
  const browserLoading = isLoading && !isSearchMode;
  const showSearchSpinner = isSearching && searchQuery.trim().length > 0;

  return (
    <div className={styles.container}>
      <div className={styles.breadcrumb}>
        <button
          className={styles.breadcrumbButton}
          onClick={handleNavigateUp}
          disabled={!canNavigateUp || isLoading}
        >
          ⬆ Up
        </button>
        <span className={styles.currentPath}>
          {isSearchMode ? `Search: "${searchQuery.trim()}"` : currentPath}
        </span>
        {onSearchQueryChange && (
          <div className={styles.searchBox}>
            <input
              type="search"
              className={styles.searchInput}
              value={searchQuery}
              onChange={(e) => onSearchQueryChange(e.target.value)}
              placeholder="Search folders…"
              disabled={isLoading && !isSearchMode}
              aria-label="Search folders"
            />
            {showSearchSpinner && <span className={styles.searchSpinner} aria-hidden="true" />}
          </div>
        )}
      </div>

      <div className={styles.browserWrapper}>
        {browserLoading ? (
          <div className={styles.loading}>
            <div className={styles.spinner} />
            <p>Loading…</p>
          </div>
        ) : (
          <FileBrowser
            rootPath={currentPath}
            nodes={nodes}
            selectedPath={selectedPath}
            onSelectPath={handleSelectPath}
            onNavigate={handleNavigate}
            searchMode={isSearchMode}
            emptyMessage={
              isSearchMode
                ? isSearching
                  ? 'Searching…'
                  : 'No matching folders found'
                : 'No files or folders found'
            }
          />
        )}
      </div>

      <div className={styles.sidebar}>
        <div className={styles.form}>
          <p className={styles.sidebarTitle}>Create Project</p>
          <p className={styles.sidebarHint}>
            Browse or search for a folder, select it, then name your project and click Create.
          </p>

          <div className={styles.field}>
            <label className={styles.label}>Project Name</label>
            <input
              type="text"
              className={styles.input}
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              placeholder="my-project"
              disabled={isLoading}
            />
            <span className={styles.hint}>Auto-filled from the selected folder name</span>
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Selected Folder</label>
            <div className={styles.selectedPath}>
              {selectedPath || 'No folder selected'}
            </div>
          </div>
        </div>

        <div className={styles.actions}>
          <button
            className={styles.createButton}
            onClick={handleCreate}
            disabled={!canCreate || isLoading}
          >
            {isLoading ? 'Creating…' : 'Create Project'}
          </button>
          {onCancel && (
            <button
              className={styles.cancelButton}
              onClick={onCancel}
              disabled={isLoading}
            >
              Cancel
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
