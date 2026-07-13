import { useEffect, useMemo, useState } from 'react';

import type { TrackListItem } from '../../shared/types.ts';
import { api } from '../api.ts';
import { useEditor } from '../store.ts';

type Sort = 'track' | 'worst' | 'best';

// Map the realistic accuracy range (~0.55–0.95) onto red→green so the worst tracks stand out.
const scoreColor = (s: number): string => {
  const h = Math.max(0, Math.min(1, (s - 0.55) / 0.4)) * 120;
  return `hsl(${Math.round(h)}, 70%, 55%)`;
};

export function TrackList() {
  const [tracks, setTracks] = useState<TrackListItem[]>([]);
  const [query, setQuery] = useState('');
  const [genre, setGenre] = useState('all');
  const [sort, setSort] = useState<Sort>('track');
  const { trackId, loadTrack, dirty, tracksRev } = useEditor();

  useEffect(() => {
    api.listTracks().then(setTracks).catch(console.error);
  }, [tracksRev]);

  const genres = useMemo(
    () => ['all', ...[...new Set(tracks.map((t) => t.genre))].sort()],
    [tracks],
  );

  const visible = useMemo(() => {
    const q = query.toLowerCase();
    let list = tracks.filter(
      (t) =>
        (genre === 'all' || t.genre === genre) &&
        (q === '' || t.title.toLowerCase().includes(q) || t.id.toLowerCase().includes(q)),
    );
    if (sort !== 'track') {
      // Sorting by eval accuracy only makes sense for tracks that have a prediction.
      list = list
        .filter((t) => t.score !== null)
        .sort((a, b) => (sort === 'worst' ? a.score! - b.score! : b.score! - a.score!));
    }
    return list;
  }, [tracks, query, genre, sort]);

  const select = (id: string) => {
    if (dirty && !confirm('Discard unsaved changes?')) return;
    void loadTrack(id);
  };

  return (
    <nav className="tracklist">
      <div className="filters">
        <input placeholder="Search…" value={query} onChange={(e) => setQuery(e.target.value)} />
        <select value={genre} onChange={(e) => setGenre(e.target.value)}>
          {genres.map((g) => (
            <option key={g} value={g}>
              {g}
            </option>
          ))}
        </select>
      </div>
      <div className="filters">
        <select value={sort} onChange={(e) => setSort(e.target.value as Sort)} title="Sort order">
          <option value="track">Sort: track order</option>
          <option value="worst">Sort: largest eval errors</option>
          <option value="best">Sort: best eval match</option>
        </select>
      </div>
      <div className="count dim">
        {visible.length} tracks{sort !== 'track' && ' with eval'}
      </div>
      <ul>
        {visible.map((t) => (
          <li
            key={t.id}
            className={t.id === trackId ? 'active' : ''}
            onClick={() => select(t.id)}
            title={t.id}
          >
            <span className="title">{t.title || t.id}</span>
            <span className="meta">
              {t.score !== null && (
                <span className="score" style={{ color: scoreColor(t.score) }}>
                  {Math.round(t.score * 100)}%
                </span>
              )}
              {t.edited && <span className="badge">edited</span>}
              <span className="dim">{t.genre}</span>
            </span>
          </li>
        ))}
      </ul>
    </nav>
  );
}
