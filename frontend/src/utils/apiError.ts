import axios from 'axios';

/** Extract a user-facing message from a FastAPI / axios error response. */
export function getApiErrorMessage(error: unknown, fallback = 'Request failed'): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) => (typeof item === 'object' && item && 'msg' in item ? String(item.msg) : String(item)))
        .join('; ');
    }
  }
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}
