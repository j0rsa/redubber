/**
 * Returns the final path segment (filename) from a full file path.
 * Works with both forward-slash and backslash separators.
 */
export const basename = (path: string): string => {
  const parts = path.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1] || path;
};
