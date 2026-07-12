import { useState, useCallback } from 'react';
import type {
  TranscriptionSegment,
  VoiceInstructions,
  VoicePreview,
  VoiceSettings,
  SegmentFilter,
} from '../components/VoiceRefinement/types';

const DEFAULT_SAMPLE_SIZE = 20;

const DEFAULT_FILTER: SegmentFilter = {
  search: '',
  minDuration: 0,
  maxDuration: 0,
};

interface UseVoiceRefinementOptions {
  projectId: number;
  onSave?: (settings: VoiceSettings) => void;
  /** First video path to transcribe when no segments exist yet */
  firstVideoPath?: string;
}

interface UseVoiceRefinementReturn {
  // State
  segments: TranscriptionSegment[];
  selectedSegment: TranscriptionSegment | null;
  voiceInstructions: string;
  voiceInstructionsData: VoiceInstructions | null;
  previews: VoicePreview[];
  selectedVoice: string | null;

  // Filter state
  filter: SegmentFilter;

  // Segments metadata
  totalCandidates: number | undefined;
  hasMore: boolean;

  // Loading states
  loadingSegments: boolean;
  analyzingVoice: boolean;
  generatingPreviews: boolean;
  saving: boolean;

  // Error states
  error: string | null;

  // Actions
  fetchSegments: () => Promise<void>;
  loadMoreSegments: () => Promise<void>;
  updateFilter: (filter: Partial<SegmentFilter>) => void;
  selectSegment: (segment: TranscriptionSegment) => void;
  analyzeVoice: () => Promise<void>;
  regenerateInstructions: (feedback?: string) => Promise<void>;
  updateInstructions: (instructions: string) => void;
  generatePreviews: () => Promise<void>;
  selectVoice: (voice: string) => void;
  saveSettings: () => Promise<void>;
  transcribeFirstVideo: () => Promise<void>;
  transcribing: boolean;
}

export const useVoiceRefinement = ({
  projectId,
  onSave,
  firstVideoPath,
}: UseVoiceRefinementOptions): UseVoiceRefinementReturn => {
  // Segment state
  const [segments, setSegments] = useState<TranscriptionSegment[]>([]);
  const [totalCandidates, setTotalCandidates] = useState<number | undefined>(undefined);
  const [hasMore, setHasMore] = useState(false);
  const [offset, setOffset] = useState(0);
  const [filter, setFilter] = useState<SegmentFilter>(DEFAULT_FILTER);

  // Voice state
  const [selectedSegment, setSelectedSegment] = useState<TranscriptionSegment | null>(null);
  const [voiceInstructions, setVoiceInstructions] = useState('');
  const [voiceInstructionsData, setVoiceInstructionsData] = useState<VoiceInstructions | null>(null);
  const [previews, setPreviews] = useState<VoicePreview[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<string | null>(null);

  // Loading states
  const [loadingSegments, setLoadingSegments] = useState(false);
  const [analyzingVoice, setAnalyzingVoice] = useState(false);
  const [generatingPreviews, setGeneratingPreviews] = useState(false);
  const [saving, setSaving] = useState(false);

  // Error states
  const [error, setError] = useState<string | null>(null);

  // Transcription-only state
  const [transcribing, setTranscribing] = useState(false);

  /** Build URL with filter + pagination params */
  const buildSegmentsUrl = (currentFilter: SegmentFilter, currentOffset: number): string => {
    const params = new URLSearchParams();
    params.set('sample', String(DEFAULT_SAMPLE_SIZE));
    params.set('offset', String(currentOffset));
    if (currentFilter.search.trim()) params.set('search', currentFilter.search.trim());
    if (currentFilter.minDuration > 0) params.set('min_duration', String(currentFilter.minDuration));
    if (currentFilter.maxDuration > 0) params.set('max_duration', String(currentFilter.maxDuration));
    return `/api/projects/${projectId}/transcription-segments?${params.toString()}`;
  };

  /** Fetch segments, replacing the list (used on initial load and filter changes) */
  const fetchSegments = useCallback(async () => {
    setLoadingSegments(true);
    setError(null);

    try {
      const url = buildSegmentsUrl(filter, 0);
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Failed to fetch segments: ${response.statusText}`);
      }

      const data = await response.json();
      setSegments(data.segments ?? []);
      setTotalCandidates(data.total_candidates);
      setHasMore(data.has_more ?? false);
      setOffset(data.returned ?? 0);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch segments';
      setError(message);
      console.error('Error fetching segments:', err);
    } finally {
      setLoadingSegments(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, filter]);

  /** Append the next page of segments */
  const loadMoreSegments = useCallback(async () => {
    if (loadingSegments) return;
    setLoadingSegments(true);
    setError(null);

    try {
      const url = buildSegmentsUrl(filter, offset);
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Failed to load more segments: ${response.statusText}`);
      }

      const data = await response.json();
      setSegments((prev) => [...prev, ...(data.segments ?? [])]);
      setTotalCandidates(data.total_candidates);
      setHasMore(data.has_more ?? false);
      setOffset((prev) => prev + (data.returned ?? 0));
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load more segments';
      setError(message);
      console.error('Error loading more segments:', err);
    } finally {
      setLoadingSegments(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, filter, offset, loadingSegments]);

  /** Update filter, reset pagination, and re-fetch */
  const updateFilter = useCallback(
    (newFilter: Partial<SegmentFilter>) => {
      setFilter((prev) => ({ ...prev, ...newFilter }));
      setOffset(0);
      setSegments([]);
      setHasMore(false);

      // Trigger a fetch with the new filter immediately
      setLoadingSegments(true);
      setError(null);

      const url = buildSegmentsUrl(newFilter, 0);
      fetch(url)
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Failed to fetch segments: ${response.statusText}`);
          }
          return response.json();
        })
        .then((data) => {
          setSegments(data.segments ?? []);
          setTotalCandidates(data.total_candidates);
          setHasMore(data.has_more ?? false);
          setOffset(data.returned ?? 0);
        })
        .catch((err) => {
          const message = err instanceof Error ? err.message : 'Failed to fetch segments';
          setError(message);
          console.error('Error fetching segments:', err);
        })
        .finally(() => {
          setLoadingSegments(false);
        });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [projectId]
  );

  // Select segment
  const selectSegment = useCallback((segment: TranscriptionSegment) => {
    setSelectedSegment(segment);
    setVoiceInstructions('');
    setVoiceInstructionsData(null);
    setPreviews([]);
    setSelectedVoice(null);
    setError(null);
  }, []);

  // Analyze voice with LLM
  const analyzeVoice = useCallback(async () => {
    if (!selectedSegment) {
      setError('Please select a segment first');
      return;
    }

    setAnalyzingVoice(true);
    setError(null);

    try {
      const response = await fetch(`/api/projects/${projectId}/voice-instructions/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          segment_id: selectedSegment.id,
          original_text: selectedSegment.original_text,
          translated_text: selectedSegment.translated_text,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to analyze voice: ${response.statusText}`);
      }

      const data = await response.json();
      setVoiceInstructions(data.voice_instructions);
      setVoiceInstructionsData(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to analyze voice';
      setError(message);
      console.error('Error analyzing voice:', err);
    } finally {
      setAnalyzingVoice(false);
    }
  }, [projectId, selectedSegment]);

  // Regenerate instructions with feedback
  const regenerateInstructions = useCallback(async (feedback?: string) => {
    if (!selectedSegment) {
      setError('Please select a segment first');
      return;
    }

    setAnalyzingVoice(true);
    setError(null);

    try {
      const response = await fetch(`/api/projects/${projectId}/voice-instructions/regenerate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          segment_id: selectedSegment.id,
          original_text: selectedSegment.original_text,
          translated_text: selectedSegment.translated_text,
          previous_instructions: voiceInstructions,
          user_feedback: feedback ?? '',
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to regenerate instructions: ${response.statusText}`);
      }

      const data = await response.json();
      setVoiceInstructions(data.voice_instructions);
      setVoiceInstructionsData(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to regenerate instructions';
      setError(message);
      console.error('Error regenerating instructions:', err);
    } finally {
      setAnalyzingVoice(false);
    }
  }, [projectId, selectedSegment, voiceInstructions]);

  // Update instructions manually
  const updateInstructions = useCallback((instructions: string) => {
    setVoiceInstructions(instructions);
  }, []);

  // Generate TTS previews for all voices
  const generatePreviews = useCallback(async () => {
    if (!selectedSegment || !voiceInstructions) {
      setError('Please select a segment and generate instructions first');
      return;
    }

    setGeneratingPreviews(true);
    setError(null);

    try {
      const response = await fetch(`/api/projects/${projectId}/voice-previews/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          translated_text: selectedSegment.translated_text,
          voice_instructions: voiceInstructions,
          voices: ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to generate previews: ${response.statusText}`);
      }

      const data = await response.json();
      setPreviews(data.previews ?? []);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to generate previews';
      setError(message);
      console.error('Error generating previews:', err);
    } finally {
      setGeneratingPreviews(false);
    }
  }, [projectId, selectedSegment, voiceInstructions]);

  // Select voice
  const selectVoice = useCallback((voice: string) => {
    setSelectedVoice(voice);
  }, []);

  // Save voice settings
  const saveSettings = useCallback(async () => {
    if (!selectedVoice || !voiceInstructions || !selectedSegment) {
      setError('Please select a voice before saving');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const response = await fetch(`/api/projects/${projectId}/voice-settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          voice: selectedVoice,
          voice_instructions: voiceInstructions,
          segment_used: selectedSegment.id,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to save settings: ${response.statusText}`);
      }

      await response.json();

      if (onSave) {
        onSave({
          voice: selectedVoice,
          voice_instructions: voiceInstructions,
          segment_used: selectedSegment.id,
        });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save settings';
      setError(message);
      console.error('Error saving settings:', err);
    } finally {
      setSaving(false);
    }
  }, [projectId, selectedVoice, voiceInstructions, selectedSegment, onSave]);

  const transcribeFirstVideo = useCallback(async () => {
    if (!firstVideoPath) {
      setError('No video available to transcribe');
      return;
    }
    setTranscribing(true);
    setError(null);
    try {
      const res = await fetch(`/api/projects/${projectId}/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: firstVideoPath, project_id: projectId }),
      });
      if (!res.ok) throw new Error(`Transcription request failed: ${res.statusText}`);
      const { task_id } = await res.json();

      // Poll until done
      while (true) {
        await new Promise((r) => setTimeout(r, 2000));
        const statusRes = await fetch(`/api/tasks/${task_id}`);
        if (!statusRes.ok) break;
        const taskStatus = await statusRes.json();
        if (taskStatus.status === 'completed') break;
        if (taskStatus.status === 'failed') {
          throw new Error(taskStatus.error ?? 'Transcription failed');
        }
      }

      // Reload segments now that .seg files exist
      await fetchSegments();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Transcription failed');
    } finally {
      setTranscribing(false);
    }
  }, [firstVideoPath, projectId, fetchSegments]);

  return {
    // State
    segments,
    selectedSegment,
    voiceInstructions,
    voiceInstructionsData,
    previews,
    selectedVoice,

    // Filter
    filter,
    totalCandidates,
    hasMore,

    // Loading states
    loadingSegments,
    analyzingVoice,
    generatingPreviews,
    saving,

    // Error state
    error,

    // Actions
    fetchSegments,
    loadMoreSegments,
    updateFilter,
    selectSegment,
    analyzeVoice,
    regenerateInstructions,
    updateInstructions,
    generatePreviews,
    selectVoice,
    saveSettings,
    transcribeFirstVideo,
    transcribing,
  };
};
