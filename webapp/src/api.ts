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

  audioUrl: (id: string): string => `/audio/${id}`,
};
