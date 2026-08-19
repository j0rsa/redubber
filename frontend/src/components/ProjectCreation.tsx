import { useState, useEffect, useMemo } from 'react';
import { FileBrowser, type FileNode } from './FileBrowser';
import { fuzzyMatchFilter } from '../utils/fuzzyMatch';
import styles from './ProjectCreation.module.css';

interface ProjectCreationProps {
  initialPath?: string;
  nodes: FileNode[];
  onLoadDirectory?: (path: string) => void;
  onCreateProject: (path: string, name: string) => void;
  onCancel?: () => void;
  isLoading?: boolean;
}

export const ProjectCreation = ({
  initialPath = '/',
  nodes,
  onLoadDirectory,
  onCreateProject,
  onCancel,
  isLoading = false,
}: ProjectCreationProps) => {
  const [selectedPath, setSelectedPath] = useState<string>('');
  const [currentPath, setCurrentPath] = useState<string>(initialPath);
  const [projectName, setProjectName] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    setCurrentPath(initialPath);
  }, [initialPath]);

  const isFiltering = searchQuery.trim().length > 0;

  const visibleNodes = useMemo(() => {
    const folders = nodes.filter((node) => node.type === 'directory');
    if (!isFiltering) {
      return nodes;
    }
    return fuzzyMatchFilter(folders, searchQuery, (node) => node.name);
  }, [nodes, searchQuery, isFiltering]);

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
        <span className={styles.currentPath}>{currentPath}</span>
        <div className={styles.searchBox}>
          <input
            type="search"
            className={styles.searchInput}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter folders…"
            disabled={isLoading}
            aria-label="Filter folders in this directory"
          />
        </div>
      </div>

      <div className={styles.browserWrapper}>
        {isLoading ? (
          <div className={styles.loading}>
            <div className={styles.spinner} />
            <p>Loading…</p>
          </div>
        ) : (
          <FileBrowser
            rootPath={currentPath}
            nodes={visibleNodes}
            selectedPath={selectedPath}
            onSelectPath={handleSelectPath}
            onNavigate={handleNavigate}
            emptyMessage={
              isFiltering ? 'No matching folders in this directory' : 'No files or folders found'
            }
          />
        )}
      </div>

      <div className={styles.sidebar}>
        <div className={styles.form}>
          <p className={styles.sidebarTitle}>Create Project</p>
          <p className={styles.sidebarHint}>
            Browse into a folder, filter subfolders by name, select one, then click Create.
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
