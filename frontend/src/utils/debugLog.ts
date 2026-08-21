type DebugPayload = {
  hypothesisId: string;
  location: string;
  message: string;
  data: Record<string, unknown>;
};

export const writeDebugLog = (payload: DebugPayload): void => {
  void fetch('/api/debug/frontend-log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...payload, timestamp: Date.now() }),
  }).catch(() => undefined);
};
