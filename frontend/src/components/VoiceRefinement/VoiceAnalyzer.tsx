import { useState } from 'react';
import type { TranscriptionSegment, VoiceInstructions } from './types';
import styles from './VoiceRefinement.module.css';

interface VoiceAnalyzerProps {
  selectedSegment: TranscriptionSegment | null;
  voiceInstructions: string;
  voiceInstructionsData: VoiceInstructions | null;
  onAnalyze: () => Promise<void>;
  onRegenerate: (feedback?: string) => Promise<void>;
  onUpdateInstructions: (instructions: string) => void;
  isAnalyzing?: boolean;
}

export const VoiceAnalyzer = ({
  selectedSegment,
  voiceInstructions,
  voiceInstructionsData,
  onAnalyze,
  onRegenerate,
  onUpdateInstructions,
  isAnalyzing = false,
}: VoiceAnalyzerProps) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedInstructions, setEditedInstructions] = useState('');
  const [feedback, setFeedback] = useState('');
  const [showFeedbackInput, setShowFeedbackInput] = useState(false);

  const handleEdit = () => {
    setIsEditing(true);
    setEditedInstructions(voiceInstructions);
  };

  const handleSaveEdit = () => {
    onUpdateInstructions(editedInstructions);
    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditedInstructions('');
  };

  const handleRegenerate = async () => {
    if (showFeedbackInput && feedback.trim()) {
      await onRegenerate(feedback);
      setFeedback('');
      setShowFeedbackInput(false);
    } else if (!showFeedbackInput) {
      await onRegenerate();
    }
  };

  const characterCount = voiceInstructions.length;
  const isLongInstructions = characterCount > 500;

  return (
    <div className={styles.section}>
      <h3 className={styles.sectionTitle}>Step 2: Voice Instructions</h3>

      {!voiceInstructions ? (
        <>
          <p className={styles.sectionDescription}>
            Analyze a segment with AI, or type instructions manually.
          </p>
          <div className={styles.instructionsActions}>
            <button
              className={styles.analyzeButton}
              onClick={onAnalyze}
              disabled={isAnalyzing || !selectedSegment}
              title={!selectedSegment ? 'Select a segment first' : undefined}
            >
              {isAnalyzing ? (
                <>
                  <span className={styles.buttonSpinner} />
                  Analyzing...
                </>
              ) : (
                <>🔬 Analyze with AI</>
              )}
            </button>
            <button
              className={styles.secondaryButton}
              onClick={handleEdit}
            >
              ✎ Enter manually
            </button>
          </div>
          {isEditing && (
            <div className={styles.instructionsContainer}>
              <textarea
                className={styles.instructionsTextarea}
                value={editedInstructions}
                onChange={(e) => setEditedInstructions(e.target.value)}
                placeholder="e.g., Speak with a warm, professional tone at a moderate pace. Clear enunciation with a slight accent."
                rows={8}
                autoFocus
              />
              <div className={styles.instructionsActions}>
                <button className={styles.secondaryButton} onClick={handleCancelEdit}>
                  Cancel
                </button>
                <button
                  className={styles.primaryButton}
                  onClick={handleSaveEdit}
                  disabled={!editedInstructions.trim()}
                >
                  Apply Instructions
                </button>
              </div>
            </div>
          )}
        </>
      ) : (
        <>
          <p className={styles.sectionDescription}>
            Review and edit the generated voice instructions
          </p>

          {voiceInstructionsData?.detected_characteristics && (
            <div className={styles.characteristics}>
              <div className={styles.characteristicChips}>
                {Object.entries(voiceInstructionsData.detected_characteristics).map(
                  ([key, value]) => (
                    <div key={key} className={styles.chip}>
                      <span className={styles.chipLabel}>{key}:</span>{' '}
                      <span className={styles.chipValue}>{value}</span>
                    </div>
                  )
                )}
              </div>
            </div>
          )}

          <div className={styles.instructionsContainer}>
            {isEditing ? (
              <textarea
                className={styles.instructionsTextarea}
                value={editedInstructions}
                onChange={(e) => setEditedInstructions(e.target.value)}
                placeholder="Enter voice instructions..."
                rows={8}
              />
            ) : (
              <div className={styles.instructionsDisplay}>
                {voiceInstructions}
              </div>
            )}

            <div className={styles.instructionsFooter}>
              <span
                className={`${styles.characterCount} ${
                  isLongInstructions ? styles.characterCountWarning : ''
                }`}
              >
                {characterCount} characters
                {isLongInstructions && ' (may be truncated)'}
              </span>
            </div>
          </div>

          <div className={styles.instructionsActions}>
            {isEditing ? (
              <>
                <button
                  className={styles.secondaryButton}
                  onClick={handleCancelEdit}
                >
                  Cancel
                </button>
                <button
                  className={styles.primaryButton}
                  onClick={handleSaveEdit}
                >
                  Save Changes
                </button>
              </>
            ) : (
              <>
                <button
                  className={styles.secondaryButton}
                  onClick={handleEdit}
                >
                  ✎ Edit
                </button>
                <button
                  className={styles.secondaryButton}
                  onClick={() => setShowFeedbackInput(!showFeedbackInput)}
                  disabled={isAnalyzing}
                >
                  ↻ Regenerate
                </button>
              </>
            )}
          </div>

          {showFeedbackInput && (
            <div className={styles.feedbackContainer}>
              <label className={styles.feedbackLabel}>
                Optional feedback for regeneration:
              </label>
              <input
                type="text"
                className={styles.feedbackInput}
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                placeholder="e.g., Make it more energetic and engaging"
              />
              <button
                className={styles.primaryButton}
                onClick={handleRegenerate}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? (
                  <>
                    <span className={styles.buttonSpinner} />
                    Regenerating...
                  </>
                ) : (
                  'Generate New Instructions'
                )}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};
