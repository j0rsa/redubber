import type { VideoFile } from '../types';

const ISO639_1_TO_2: Record<string, string> = {
  en: 'eng',
  es: 'spa',
  fr: 'fra',
  de: 'deu',
  it: 'ita',
  pt: 'por',
  ru: 'rus',
  ja: 'jpn',
  ko: 'kor',
  zh: 'zho',
  ar: 'ara',
  hi: 'hin',
};

const ISO639_2_ALIASES: Record<string, string> = {
  chi: 'zho',
  ger: 'deu',
  fre: 'fra',
};

/** Comparable ISO 639-2 code, or empty string. */
export function normalizeLangCode(code: string | undefined | null): string {
  const raw = (code ?? '').trim().toLowerCase();
  if (!raw) return '';
  const three = ISO639_1_TO_2[raw] ?? raw;
  return ISO639_2_ALIASES[three] ?? three;
}

/** True when a video looks fully redubbed to the project target language.
 *
 * Audio-only check: a dubbed track muxed into the file is the definitive
 * signal. Subtitles are intentionally excluded — an external sidecar in the
 * same language is not evidence that the pipeline ran.
 */
export function isVideoInTargetState(
  audioStreams: { language: string }[],
  _subtitles: { language: string }[],
  targetLanguage: string,
): boolean {
  const target = normalizeLangCode(targetLanguage);
  if (!target || audioStreams.length < 2) return false;
  return audioStreams.some(
    (stream) => normalizeLangCode(stream.language) === target,
  );
}

/** True when a video is finalized, including disk-derived target-language state. */
export function isVideoFinalized(
  video: Pick<VideoFile, 'pipeline_status' | 'audio_streams' | 'subtitles'>,
  targetLanguage: string,
): boolean {
  return Boolean(video.pipeline_status?.replaced)
    || isVideoInTargetState(video.audio_streams, video.subtitles, targetLanguage);
}
