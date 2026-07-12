import { useState } from 'react';
import styles from './FileBrowser.module.css';

export interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: FileNode[];
  size?: number;
  modified?: string;
}

interface FileBrowserProps {
  rootPath: string;
  nodes: FileNode[];
  selectedPath?: string;
  onSelectPath: (path: string) => void;
  onNavigate?: (path: string) => void;
}

export const FileBrowser = ({
  rootPath,
  nodes,
  selectedPath,
  onSelectPath,
  onNavigate,
}: FileBrowserProps) => {
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set([rootPath]));

  const toggleExpand = (path: string) => {
    const newExpanded = new Set(expandedPaths);
    if (newExpanded.has(path)) {
      newExpanded.delete(path);
    } else {
      newExpanded.add(path);
    }
    setExpandedPaths(newExpanded);
  };

  const handleNodeClick = (node: FileNode) => {
    if (node.type === 'directory') {
      toggleExpand(node.path);
      onSelectPath(node.path);
    }
  };

  const handleNodeDoubleClick = (node: FileNode) => {
    if (node.type === 'directory' && onNavigate) {
      onNavigate(node.path);
    }
  };

  const renderNode = (node: FileNode, depth: number = 0) => {
    const isExpanded = expandedPaths.has(node.path);
    const isSelected = selectedPath === node.path;
    const isDirectory = node.type === 'directory';

    return (
      <div key={node.path} className={styles.nodeWrapper}>
        <div
          className={`${styles.node} ${isSelected ? styles.selected : ''} ${
            isDirectory ? styles.directory : styles.file
          }`}
          style={{ paddingLeft: `${depth * 20 + 8}px` }}
          onClick={() => handleNodeClick(node)}
          onDoubleClick={() => handleNodeDoubleClick(node)}
        >
          <span className={styles.icon}>
            {isDirectory ? '📁' : '📄'}
          </span>
          <span className={styles.name}>{node.name}</span>
          {!isDirectory && node.size !== undefined && (
            <span className={styles.size}>{formatSize(node.size)}</span>
          )}
        </div>
        {isDirectory && isExpanded && node.children && (
          <div className={styles.children}>
            {node.children.map((child) => renderNode(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={styles.container}>
      <div className={styles.nodes}>
        {nodes.length === 0 ? (
          <div className={styles.empty}>No files or folders found</div>
        ) : (
          nodes.map((node) => renderNode(node, 0))
        )}
      </div>
    </div>
  );
};

const formatSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
};
