/** Lightweight fuzzy matching for folder name filtering. */

export function fuzzyScore(query: string, text: string): number {
  const q = query.trim().toLowerCase();
  const t = text.toLowerCase();
  if (!q || !t) return 0;

  if (t.includes(q)) {
    return 100 + (q.length / t.length) * 50;
  }

  let qi = 0;
  let score = 0;
  let prevMatch = -1;
  for (let ti = 0; ti < t.length; ti += 1) {
    if (qi < q.length && t[ti] === q[qi]) {
      score += 1;
      if (prevMatch === ti - 1) score += 2;
      if (qi === 0 && ti === 0) score += 5;
      prevMatch = ti;
      qi += 1;
    }
  }

  return qi === q.length ? score : 0;
}

export function fuzzyMatchFilter<T>(
  items: T[],
  query: string,
  getText: (item: T) => string,
): T[] {
  const q = query.trim();
  if (!q) return items;

  return items
    .map((item) => ({ item, score: fuzzyScore(q, getText(item)) }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score || getText(a.item).localeCompare(getText(b.item)))
    .map(({ item }) => item);
}
