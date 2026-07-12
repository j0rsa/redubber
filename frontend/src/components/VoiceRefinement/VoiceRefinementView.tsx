import { useState } from 'react';
import { SegmentSelector } from './SegmentSelector';
import { VoiceAnalyzer } from './VoiceAnalyzer';
import { VoicePreviewGrid } from './VoicePreviewGrid';
import { ManualVoiceTab } from './ManualVoiceTab';
import type { TranscriptionSegment, VoiceInstructions, VoicePreview, SegmentFilter } from './types';
import styles from './VoiceRefinement.module.css';

export interface VoiceRefinementViewProps {
  // Segment state
  segments: TranscriptionSegment[];
  selectedSegment: TranscriptionSegment | null;
  loadingSegments: boolean;
  onSelectSegment: (segment: TranscriptionSegment) => void;
  filter: SegmentFilter;
  totalCandidates?: number;
  hasMore: boolean;
  onFilterChange: (filter: Partial<SegmentFilter>) => void;
  onLoadMore: () => void;

  // Voice instructions
  voiceInstructions: string;
  voiceInstructionsData: VoiceInstructions | null;
  analyzingVoice: boolean;
  onAnalyze: () => Promise<void>;
  onRegenerate: (feedback?: string) => Promise<void>;
  onUpdateInstructions: (instructions: string) => void;

  // Previews
  previews: VoicePreview[];
  selectedVoice: string | null;
  generatingPreviews: boolean;
  onSelectVoice: (voice: string) => void;
  onGeneratePreviews: () => Promise<void>;
  speakerGender?: 'male' | 'female' | 'unknown';

  // Actions
  saving: boolean;
  error: string | null;
  onSave: () => void;
  onClose: () => void;
  onTranscribe?: () => Promise<void>;
  transcribing?: boolean;
}

type Mode = 'interactive' | 'manual';

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
  const [mode, setMode] = useState<Mode>('interactive');

  const canSave = selectedVoice && voiceInstructions && !saving;

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>

        {/* ── Header ── */}
        <div className={styles.header}>
          <h2 className={styles.title}>Voice Customization</h2>
          <button className={styles.closeButton} onClick={onClose}>✕</button>
        </div>

        {/* ── Mode tabs ── */}
        <div className={styles.tabs}>
          <button
            className={`${styles.tab} ${mode === 'interactive' ? styles.tabActive : ''}`}
            onClick={() => setMode('interactive')}
          >
            🔬 Interactive
          </button>
          <button
            className={`${styles.tab} ${mode === 'manual' ? styles.tabActive : ''}`}
            onClick={() => setMode('manual')}
          >
            ✎ Manual
          </button>
        </div>

        {/* ── Content ── */}
        <div className={styles.content}>
          {error && (
            <div className={styles.errorBanner}>
              <span className={styles.errorIcon}>⚠️</span>
              <span className={styles.errorText}>{error}</span>
            </div>
          )}

          {mode === 'interactive' ? (
            <>
              <SegmentSelector
                segments={segments}
                selectedSegment={selectedSegment}
                isLoading={loadingSegments || transcribing}
                filter={filter}
                totalCandidates={totalCandidates}
                hasMore={hasMore}
                onSelectSegment={onSelectSegment}
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
            </>
          ) : (
            <ManualVoiceTab
              voiceInstructions={voiceInstructions}
              selectedVoice={selectedVoice}
              previews={previews}
              generatingPreviews={generatingPreviews}
              onUpdateInstructions={onUpdateInstructions}
              onSelectVoice={onSelectVoice}
              onGeneratePreviews={onGeneratePreviews}
            />
          )}
        </div>

        {/* ── Footer ── */}
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
