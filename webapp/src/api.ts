import type {
  Annotation,
  ResegmentInfo,
  Segment,
  StemsResult,
  TrackListItem,
  TrackMeta,
} from '../shared/types.ts';

/** Where the backend is reached, for connection-refused error messages. Requests use
 *  relative paths, so this is just the page origin (dev: the Vite proxy target). */
const backendOrigin = (): string =>
  typeof window !== 'undefined' && window.location ? window.location.origin : 'the server';

/** fetch wrapper that turns a network-layer failure (connection refused, DNS, offline —
 *  where the browser rejects with an opaque `TypeError: Failed to fetch`) into an
 *  actionable message. HTTP error responses (4xx/5xx) still resolve normally; callers
 *  check `res.ok` and surface the status + any server-provided message (see `json`). */
export async function apiFetch(input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch (err) {
    // fetch only rejects on network/transport failures, never on HTTP status.
    throw new Error(
      `Can't reach the jams backend at ${backendOrigin()} — is the server running?`,
      { cause: err },
    );
  }
}

/** Build the message for a non-OK HTTP response: status line plus any server-provided
 *  error (JSON `{ error }` or a short text body). */
async function httpError(res: Response): Promise<Error> {
  let serverMsg = '';
  const body = await res.text().catch(() => '');
  if (body) {
    try {
      serverMsg = ((JSON.parse(body) as { error?: string }).error ?? '').trim();
    } catch {
      serverMsg = body.slice(0, 200).trim();
    }
  }
  return new Error(serverMsg ? `${res.status} ${res.statusText}: ${serverMsg}` : `${res.status} ${res.statusText}`);
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw await httpError(res);
  return (await res.json()) as T;
}

export const api = {
  listTracks: (): Promise<TrackListItem[]> => apiFetch('/api/tracks').then(json<TrackListItem[]>),

  getTrack: (id: string): Promise<TrackMeta> => apiFetch(`/api/tracks/${id}`).then(json<TrackMeta>),

  getAnnotation: (id: string): Promise<Annotation> =>
    apiFetch(`/api/tracks/${id}/annotation`).then(json<Annotation>),

  getPrediction: async (id: string): Promise<Annotation | null> => {
    const res = await apiFetch(`/api/tracks/${id}/prediction`);
    if (res.status === 204) return null;
    return json<Annotation>(res);
  },

  getStems: async (id: string): Promise<StemsResult | null> => {
    const res = await apiFetch(`/api/tracks/${id}/stems`);
    if (res.status === 204) return null;
    return json<StemsResult>(res);
  },

  midiUrl: (id: string, stem: string): string => `/api/tracks/${id}/midi/${stem}`,

  /** Section-count slider metadata; null when the track has no cached activations. */
  getResegmentInfo: async (id: string): Promise<ResegmentInfo | null> => {
    const res = await apiFetch(`/api/tracks/${id}/resegment`);
    if (res.status === 204) return null;
    return json<ResegmentInfo>(res);
  },

  /** Rethreshold the track's cached activations to `count` sections (instant — no model). */
  resegment: async (id: string, count: number): Promise<{ segments: Segment[]; threshold: number }> => {
    const res = await apiFetch(`/api/tracks/${id}/resegment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ count }),
    });
    if (!res.ok) throw await httpError(res);
    const raw = (await res.json()) as {
      segments: { start: number; end: number; label: string }[];
      threshold: number;
    };
    return {
      segments: raw.segments.map((s) => ({
        start: s.start,
        end: s.end,
        label: s.label as Segment['label'],
      })),
      threshold: raw.threshold,
    };
  },

  saveAnnotation: (id: string, ann: Annotation): Promise<{ ok: boolean }> =>
    apiFetch(`/api/tracks/${id}/annotation`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(ann),
    }).then(json<{ ok: boolean }>),

  /** Upload + analyze an audio file; resolves to the new track id. Slow (full analysis) —
   *  the server replies only once the jams backend has finished. Prefer importStart +
   *  the SSE progress stream for interactive use. */
  importTrack: async (file: File, opts?: { stems?: boolean }): Promise<{ id: string }> => {
    const form = new FormData();
    form.append('file', file);
    if (opts?.stems === false) form.append('stems', 'false');
    const res = await apiFetch('/api/import', { method: 'POST', body: form });
    if (!res.ok) throw await httpError(res);
    return (await res.json()) as { id: string };
  },

  /** Kick off a progress-reporting import; stream stage events from importProgressUrl. */
  importStart: async (file: File, opts?: { stems?: boolean }): Promise<{ importId: string }> => {
    const form = new FormData();
    form.append('file', file);
    if (opts?.stems === false) form.append('stems', 'false');
    const res = await apiFetch('/api/import/start', { method: 'POST', body: form });
    if (!res.ok) throw await httpError(res);
    return (await res.json()) as { importId: string };
  },

  importProgressUrl: (importId: string): string => `/api/import/progress/${importId}`,

  audioUrl: (id: string): string => `/audio/${id}`,
};
