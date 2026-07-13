import type { Annotation, StemsResult, TrackListItem, TrackMeta } from '../shared/types.ts';

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export const api = {
  listTracks: (): Promise<TrackListItem[]> => fetch('/api/tracks').then(json<TrackListItem[]>),

  getTrack: (id: string): Promise<TrackMeta> => fetch(`/api/tracks/${id}`).then(json<TrackMeta>),

  getAnnotation: (id: string): Promise<Annotation> =>
    fetch(`/api/tracks/${id}/annotation`).then(json<Annotation>),

  getPrediction: async (id: string): Promise<Annotation | null> => {
    const res = await fetch(`/api/tracks/${id}/prediction`);
    if (res.status === 204) return null;
    return json<Annotation>(res);
  },

  getStems: async (id: string): Promise<StemsResult | null> => {
    const res = await fetch(`/api/tracks/${id}/stems`);
    if (res.status === 204) return null;
    return json<StemsResult>(res);
  },

  midiUrl: (id: string, stem: string): string => `/api/tracks/${id}/midi/${stem}`,

  saveAnnotation: (id: string, ann: Annotation): Promise<{ ok: boolean }> =>
    fetch(`/api/tracks/${id}/annotation`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(ann),
    }).then(json<{ ok: boolean }>),

  /** Upload + analyze an audio file; resolves to the new track id. Slow (full analysis) —
   *  the server replies only once the jams backend has finished. Prefer importStart +
   *  the SSE progress stream for interactive use. */
  importTrack: async (file: File): Promise<{ id: string }> => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/api/import', { method: 'POST', body: form });
    if (!res.ok) {
      const body = (await res.json().catch(() => null)) as { error?: string } | null;
      throw new Error(body?.error ?? `${res.status} ${res.statusText}`);
    }
    return (await res.json()) as { id: string };
  },

  /** Kick off a progress-reporting import; stream stage events from importProgressUrl. */
  importStart: async (file: File): Promise<{ importId: string }> => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/api/import/start', { method: 'POST', body: form });
    if (!res.ok) {
      const body = (await res.json().catch(() => null)) as { error?: string } | null;
      throw new Error(body?.error ?? `${res.status} ${res.statusText}`);
    }
    return (await res.json()) as { importId: string };
  },

  importProgressUrl: (importId: string): string => `/api/import/progress/${importId}`,

  audioUrl: (id: string): string => `/audio/${id}`,
};
