import { useEffect, useMemo, useState } from 'react';

import type { TrackListItem } from '../../shared/types.ts';
import { api } from '../api.ts';
import { useEditor } from '../store.ts';

export function TrackList() {
  const [tracks, setTracks] = useState<TrackListItem[]>([]);
  const [query, setQuery] = useState('');
  const [genre, setGenre] = useState('all');
  const { trackId, loadTrack, dirty } = useEditor();

  useEffect(() => {
    api.listTracks().then(setTracks).catch(console.error);
  }, []);

  const genres = useMemo(
    () => ['all', ...[...new Set(tracks.map((t) => t.genre))].sort()],
    [tracks],
  );

  const filtered = useMemo(() => {
    const q = query.toLowerCase();
    return tracks.filter(
      (t) =>
        (genre === 'all' || t.genre === genre) &&
        (q === '' || t.title.toLowerCase().includes(q) || t.id.toLowerCase().includes(q)),
    );
  }, [tracks, query, genre]);

  const select = (id: string) => {
    if (dirty && !confirm('Discard unsaved changes?')) return;
    void loadTrack(id);
  };

  return (
    <nav className="tracklist">
      <div className="filters">
        <input
          placeholder="Search…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select value={genre} onChange={(e) => setGenre(e.target.value)}>
          {genres.map((g) => (
            <option key={g} value={g}>
              {g}
            </option>
          ))}
        </select>
      </div>
      <div className="count dim">{filtered.length} tracks</div>
      <ul>
        {filtered.map((t) => (
          <li
            key={t.id}
            className={t.id === trackId ? 'active' : ''}
            onClick={() => select(t.id)}
            title={t.id}
          >
            <span className="title">{t.title || t.id}</span>
            <span className="meta">
              {t.edited && <span className="badge">edited</span>}
              <span className="dim">{t.genre}</span>
            </span>
          </li>
        ))}
      </ul>
    </nav>
  );
}
