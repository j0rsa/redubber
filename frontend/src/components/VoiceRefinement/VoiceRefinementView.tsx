import type {
  TranscriptionSegment,
  VoiceInstructions,
  VoicePreview,
  SegmentFilter,
} from './types';
import { SegmentSelector } from './SegmentSelector';
import { VoiceAnalyzer } from './VoiceAnalyzer';
import { VoicePreviewGrid } from './VoicePreviewGrid';
import styles from './VoiceRefinement.module.css';

export interface VoiceRefinementViewProps {
  // Segment step
  segments: TranscriptionSegment[];
  selectedSegment: TranscriptionSegment | null;
  loadingSegments: boolean;
  onSelectSegment: (segment: TranscriptionSegment) => void;
  // Filter
  filter: SegmentFilter;
  totalCandidates?: number;
  hasMore: boolean;
  onFilterChange: (filter: SegmentFilter) => void;
  onLoadMore: () => void;

  // Analysis step
  voiceInstructions: string;
  voiceInstructionsData: VoiceInstructions | null;
  analyzingVoice: boolean;
  onAnalyze: () => Promise<void>;
  onRegenerate: (feedback?: string) => Promise<void>;
  onUpdateInstructions: (instructions: string) => void;

  // Preview step
  previews: VoicePreview[];
  selectedVoice: string | null;
  generatingPreviews: boolean;
  onSelectVoice: (voice: string) => void;
  onGeneratePreviews: () => Promise<void>;
  speakerGender?: 'male' | 'female' | 'unknown';

  // Footer
  saving: boolean;
  error: string | null;
  onSave: () => void;
  onClose: () => void;

  // Transcription shortcut
  onTranscribe?: () => Promise<void>;
  transcribing?: boolean;
}

export const VoiceRefinementView = ({
  segments,
  selectedSegment,
  loadingSegments,
  onSelectSegment,
  filter,
  totalCandidates,
  hasMore,
  onFilterChange,
  onLoadMore,
  voiceInstructions,
  voiceInstructionsData,
  analyzingVoice,
  onAnalyze,
  onRegenerate,
  onUpdateInstructions,
  previews,
  selectedVoice,
  generatingPreviews,
  onSelectVoice,
  onGeneratePreviews,
  speakerGender,
  saving,
  error,
  onSave,
  onClose,
  onTranscribe,
  transcribing = false,
}: VoiceRefinementViewProps) => {
  const canSave = selectedVoice && voiceInstructions && !saving;

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h2 className={styles.title}>Voice Refinement</h2>
          <button className={styles.closeButton} onClick={onClose}>
            ✕
          </button>
        </div>

        <div className={styles.content}>
          {error && (
            <div className={styles.errorBanner}>
              <span className={styles.errorIcon}>⚠️</span>
              <span className={styles.errorText}>{error}</span>
            </div>
          )}

          <SegmentSelector
            segments={segments}
            selectedSegment={selectedSegment}
            onSelectSegment={onSelectSegment}
            isLoading={loadingSegments || transcribing}
            filter={filter}
            totalCandidates={totalCandidates}
            hasMore={hasMore}
            onFilterChange={onFilterChange}
            onLoadMore={onLoadMore}
            onTranscribe={onTranscribe}
            transcribing={transcribing}
          />

          <VoiceAnalyzer
            selectedSegment={selectedSegment}
            voiceInstructions={voiceInstructions}
            voiceInstructionsData={voiceInstructionsData}
            onAnalyze={onAnalyze}
            onRegenerate={onRegenerate}
            onUpdateInstructions={onUpdateInstructions}
            isAnalyzing={analyzingVoice}
          />

          <VoicePreviewGrid
            previews={previews}
            selectedVoice={selectedVoice}
            onSelectVoice={onSelectVoice}
            onGeneratePreviews={onGeneratePreviews}
            isGenerating={generatingPreviews}
            hasInstructions={!!voiceInstructions}
            speakerGender={speakerGender}
          />
        </div>

        <div className={styles.footer}>
          <button className={styles.cancelButton} onClick={onClose}>
            Cancel
          </button>
          <button
            className={styles.saveButton}
            onClick={onSave}
            disabled={!canSave}
          >
            {saving ? (
              <>
                <span className={styles.buttonSpinner} />
                Saving...
              </>
            ) : (
              'Save Voice Settings'
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
