import { useCallback, useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import type { SubtitleReviewData } from '../components/SubtitleReview/types';

interface UseSubtitleReviewOptions {
  projectId: number | null;
  videoId: number | null;
  srtPath?: string | null;
}

export const useSubtitleReview = ({
  projectId,
  videoId,
  srtPath = null,
}: UseSubtitleReviewOptions) => {
  const [data, setData] = useState<SubtitleReviewData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savingCueIndex, setSavingCueIndex] = useState<number | null>(null);

  const fetchReview = useCallback(async () => {
    if (!projectId || !videoId) return;
    setLoading(true);
    setError(null);
    try {
      const params: Record<string, string> = {};
      if (srtPath) {
        params.srt_path = srtPath;
      }
      const { data: body } = await apiClient.get<SubtitleReviewData>(
        `/projects/${projectId}/videos/${videoId}/subtitle-review`,
        { params },
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
  }, [projectId, videoId, srtPath]);

  const saveCue = useCallback(
    async (index: number, text: string) => {
      if (!projectId || !videoId) return;
      setSavingCueIndex(index);
      try {
        const { data: body } = await apiClient.patch<SubtitleReviewData>(
          `/projects/${projectId}/videos/${videoId}/subtitle-review/cues/${index}`,
          { text, srt_path: srtPath },
        );
        setData(body);
      } finally {
        setSavingCueIndex(null);
      }
    },
    [projectId, videoId, srtPath],
  );

  useEffect(() => {
    if (!projectId || !videoId) {
      setData(null);
      setError(null);
      return;
    }
    void fetchReview();
  }, [projectId, videoId, fetchReview]);

  return { data, loading, error, reload: fetchReview, saveCue, savingCueIndex };
};
