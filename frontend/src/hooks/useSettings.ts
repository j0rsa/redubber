import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../api/client';
import type { SettingsData } from '../types/settings';
import { DEFAULT_SETTINGS } from '../types/settings';

interface UseSettingsResult {
  settings: SettingsData;
  isSaving: boolean;
  error: string | null;
  successMessage: string | null;
  saveSettings: (update: Partial<SettingsData>) => Promise<void>;
}

/**
 * Manages settings state: fetches from GET /api/settings on mount,
 * persists via PUT /api/settings on save.
 */
export const useSettings = (): UseSettingsResult => {
  const [settings, setSettings] = useState<SettingsData>(DEFAULT_SETTINGS);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetchSettings = async (): Promise<void> => {
      try {
        const { data } = await apiClient.get<SettingsData>('/settings');
        if (!cancelled) {
          setSettings(data);
        }
      } catch {
        if (!cancelled) {
          setError('Failed to load settings');
        }
      }
    };

    void fetchSettings();
    return () => {
      cancelled = true;
    };
  }, []);

  const saveSettings = useCallback(async (update: Partial<SettingsData>): Promise<void> => {
    setIsSaving(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const merged: SettingsData = { ...settings, ...update };
      // Never send a masked key back — the backend already has the real one.
      // A masked key looks like "sk-...xxxx" and would overwrite the real key with garbage.
      if (merged.openai_api_key?.startsWith('sk-...') || merged.openai_api_key === settings.openai_api_key) {
        delete (merged as Partial<SettingsData>).openai_api_key;
      }
      const { data } = await apiClient.put<SettingsData>('/settings', merged);
      setSettings(data);
      setSuccessMessage('Settings saved');
    } catch {
      setError('Failed to save settings');
    } finally {
      setIsSaving(false);
    }
  }, [settings]);

  return { settings, isSaving, error, successMessage, saveSettings };
};
