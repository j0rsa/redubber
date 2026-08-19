import type { VideoFile } from '../types';

export interface VideoFolderGroup {
  folder: string;
  videos: VideoFile[];
}

function normalizePath(path: string): string {
  return path.replace(/\\/g, '/').replace(/\/$/, '');
}

/** Relative folder path from the project root, or "." for files at project root. */
export function getVideoFolderKey(video: VideoFile, projectPath?: string): string {
  const filePath = normalizePath(video.path);
  const dir = filePath.slice(0, filePath.lastIndexOf('/'));

  if (projectPath) {
    const root = normalizePath(projectPath);
    if (dir === root) {
      return '.';
    }
    if (dir.startsWith(`${root}/`)) {
      return dir.slice(root.length + 1);
    }
  }

  const parts = dir.split('/').filter(Boolean);
  return parts[parts.length - 1] ?? '.';
}

export function formatFolderLabel(folder: string): string {
  return folder === '.' ? 'Project root' : folder;
}

export function groupVideosByFolder(
  videos: VideoFile[],
  projectPath?: string,
): VideoFolderGroup[] {
  const groups = new Map<string, VideoFile[]>();

  for (const video of videos) {
    const folder = getVideoFolderKey(video, projectPath);
    const list = groups.get(folder) ?? [];
    list.push(video);
    groups.set(folder, list);
  }

  const compareNames = (a: string, b: string) =>
    a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });

  return [...groups.entries()]
    .sort(([folderA], [folderB]) => {
      if (folderA === '.') return -1;
      if (folderB === '.') return 1;
      return compareNames(folderA, folderB);
    })
    .map(([folder, folderVideos]) => ({
      folder,
      videos: [...folderVideos].sort((a, b) => compareNames(a.filename, b.filename)),
    }));
}
