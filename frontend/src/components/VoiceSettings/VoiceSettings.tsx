import { useState } from 'react';
import type { Project } from '../../types';
import { getVoiceIcon, getVoiceDisplayName } from '../../constants/voices';
import styles from './VoiceSettings.module.css';

interface VoiceSettingsProps {
  project: Project;
  onOpenRefinement: () => void;
}

export const VoiceSettings = ({ project, onOpenRefinement }: VoiceSettingsProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editedInstructions, setEditedInstructions] = useState(project.voice_instructions);

  const hasInstructions = project.voice_instructions && project.voice_instructions.trim().length > 0;
  const instructionsPreview = project.voice_instructions?.substring(0, 100);
  const shouldShowExpand = project.voice_instructions && project.voice_instructions.length > 100;

  const handleSaveInstructions = async () => {
    try {
      const response = await fetch(`/api/projects/${project.id}/voice-settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          voice: project.voice,
          instructions: editedInstructions,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to update voice instructions');
      }

      setIsEditing(false);
      // Update would normally trigger a refetch via React Query
      window.location.reload(); // Simple approach for now
    } catch (err) {
      console.error('Failed to save voice instructions:', err);
      alert('Failed to save voice instructions');
    }
  };

  const handleClearSettings = async () => {
    if (!confirm('Clear voice settings and reset to defaults?')) {
      return;
    }

    try {
      const response = await fetch(`/api/projects/${project.id}/voice-settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          voice: 'alloy',
          instructions: '',
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to clear voice settings');
      }

      window.location.reload();
    } catch (err) {
      console.error('Failed to clear voice settings:', err);
      alert('Failed to clear voice settings');
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h2 className={styles.title}>Voice Settings</h2>
        <button onClick={onOpenRefinement} className={styles.refineButton}>
          <span className={styles.buttonIcon}>✨</span>
          Refine Voice
        </button>
      </div>

      <div className={styles.content}>
        {/* Current Voice */}
        <div className={styles.section}>
          <div className={styles.label}>Selected Voice</div>
          <div className={styles.voiceDisplay}>
            <span className={styles.voiceIcon}>{getVoiceIcon(project.voice)}</span>
            <span className={styles.voiceName}>{getVoiceDisplayName(project.voice)}</span>
          </div>
        </div>

        {/* Voice Instructions */}
        <div className={styles.section}>
          <div className={styles.labelRow}>
            <div className={styles.label}>Voice Instructions</div>
            {hasInstructions && !isEditing && (
              <div className={styles.actionButtons}>
                <button onClick={() => setIsEditing(true)} className={styles.textButton}>
                  Edit
                </button>
                <button onClick={handleClearSettings} className={styles.textButton}>
                  Clear
                </button>
              </div>
            )}
          </div>

          {isEditing ? (
            <div className={styles.editMode}>
              <textarea
                value={editedInstructions}
                onChange={(e) => setEditedInstructions(e.target.value)}
                className={styles.textarea}
                placeholder="Enter voice instructions (e.g., speak slowly, emphasize emotions, use a British accent...)"
                rows={6}
              />
              <div className={styles.editActions}>
                <button onClick={() => setIsEditing(false)} className={styles.cancelButton}>
                  Cancel
                </button>
                <button onClick={handleSaveInstructions} className={styles.saveButton}>
                  Save
                </button>
              </div>
            </div>
          ) : hasInstructions ? (
            <div className={styles.instructionsDisplay}>
              <p className={styles.instructionsText}>
                {isExpanded || !shouldShowExpand
                  ? project.voice_instructions
                  : `${instructionsPreview}...`}
              </p>
              {shouldShowExpand && (
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className={styles.expandButton}
                >
                  {isExpanded ? 'Show less' : 'Show more'}
                </button>
              )}
            </div>
          ) : (
            <div className={styles.emptyState}>
              <p className={styles.emptyText}>
                No voice instructions set. Click "Refine Voice" to add custom instructions for TTS generation.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
