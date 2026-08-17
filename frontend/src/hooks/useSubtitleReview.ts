import { useCallback, useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import type { SubtitleReviewData } from '../components/SubtitleReview/types';

interface UseSubtitleReviewOptions {
  projectId: number | null;
  videoId: number | null;
}

export const useSubtitleReview = ({ projectId, videoId }: UseSubtitleReviewOptions) => {
  const [data, setData] = useState<SubtitleReviewData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchReview = useCallback(async () => {
    if (!projectId || !videoId) return;
    setLoading(true);
    setError(null);
    try {
      const { data: body } = await apiClient.get<SubtitleReviewData>(
        `/projects/${projectId}/videos/${videoId}/subtitle-review`,
      );
      setData(body);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load subtitles';
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || message);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [projectId, videoId]);

  useEffect(() => {
    if (!projectId || !videoId) {
      setData(null);
      setError(null);
      return;
    }
    void fetchReview();
  }, [projectId, videoId, fetchReview]);

  return { data, loading, error, reload: fetchReview };
};
