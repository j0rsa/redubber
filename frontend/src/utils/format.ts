/** Format a duration in seconds as M:SS or H:MM:SS when ≥ 1 hour. */
export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds || 0));
  const hours = Math.floor(total / 3600);
  const mins = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/** Format a size in megabytes as MB or GB. */
export function formatSize(mb: number): string {
  const value = mb || 0;
  return value >= 1000 ? `${(value / 1024).toFixed(1)} GB` : `${value.toFixed(1)} MB`;
}
