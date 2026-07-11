import { useEffect } from 'react';
import { useVoiceRefinement } from '../../hooks/useVoiceRefinement';
import { VoiceRefinementView } from './VoiceRefinementView';
import type { VoiceSettings } from './types';

interface VoiceRefinementProps {
  projectId: number;
  isOpen: boolean;
  onClose: () => void;
  onSave?: (settings: VoiceSettings) => void;
  /** Path of the first available video, for transcription-only run */
  firstVideoPath?: string;
}

export const VoiceRefinement = ({
  projectId,
  isOpen,
  onClose,
  onSave,
  firstVideoPath,
}: VoiceRefinementProps) => {
  const {
    segments,
    selectedSegment,
    voiceInstructions,
    voiceInstructionsData,
    previews,
    selectedVoice,
    filter,
    totalCandidates,
    hasMore,
    loadingSegments,
    analyzingVoice,
    generatingPreviews,
    saving,
    error,
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
  } = useVoiceRefinement({ projectId, onSave, firstVideoPath });

  useEffect(() => {
    if (isOpen && segments.length === 0) {
      fetchSegments();
    }
  }, [isOpen, segments.length, fetchSegments]);

  useEffect(() => {
    if (voiceInstructions && previews.length === 0 && !generatingPreviews) {
      generatePreviews();
    }
  }, [voiceInstructions, previews.length, generatingPreviews, generatePreviews]);

  const handleSave = async () => {
    await saveSettings();
    if (!error) {
      onClose();
    }
  };

  if (!isOpen) {
    return null;
  }

  return (
    <VoiceRefinementView
      segments={segments}
      selectedSegment={selectedSegment}
      loadingSegments={loadingSegments}
      onSelectSegment={selectSegment}
      filter={filter}
      totalCandidates={totalCandidates}
      hasMore={hasMore}
      onFilterChange={updateFilter}
      onLoadMore={loadMoreSegments}
      voiceInstructions={voiceInstructions}
      voiceInstructionsData={voiceInstructionsData}
      analyzingVoice={analyzingVoice}
      onAnalyze={analyzeVoice}
      onRegenerate={regenerateInstructions}
      onUpdateInstructions={updateInstructions}
      previews={previews}
      selectedVoice={selectedVoice}
      generatingPreviews={generatingPreviews}
      onSelectVoice={selectVoice}
      onGeneratePreviews={generatePreviews}
      speakerGender={voiceInstructionsData?.detected_characteristics?.speaker_gender}
      saving={saving}
      error={error}
      onSave={handleSave}
      onClose={onClose}
      onTranscribe={firstVideoPath ? transcribeFirstVideo : undefined}
      transcribing={transcribing}
    />
  );
};
