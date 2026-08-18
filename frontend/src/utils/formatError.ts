/** Normalize pipeline error strings for display (preserve intentional line breaks). */
export function formatPipelineError(error: string): string {
  const normalized = error.replace(/\r\n/g, '\n').trim();
  if (!normalized.includes('\n') && normalized.includes(' - ')) {
    // Some legacy errors join findings on one line with " - " separators.
    const headerMatch = normalized.match(/^(.+?:)\s*(.+)$/);
    if (headerMatch) {
      const [, header, body] = headerMatch;
      const items = body.split(/\s+-\s+/).map((part) => part.trim()).filter(Boolean);
      if (items.length > 1) {
        return [header, ...items.map((item) => `- ${item}`)].join('\n');
      }
    }
  }
  return normalized;
}

/** Short single-line preview for narrow layouts (mobile). */
export function truncatePipelineError(error: string, maxLength = 80): string {
  const singleLine = formatPipelineError(error).replace(/\s*\n+\s*/g, ' ').trim();
  return singleLine.length > maxLength ? `${singleLine.slice(0, maxLength)}…` : singleLine;
}
