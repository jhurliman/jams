import { useEffect } from 'react';

import { api } from './api.ts';
import { Inspector } from './components/Inspector.tsx';
import { StemLanes } from './components/StemLanes.tsx';
import { TrackList } from './components/TrackList.tsx';
import { Transport } from './components/Transport.tsx';
import { Waveform } from './components/Waveform.tsx';
import { useAudio } from './hooks/useAudio.ts';
import { usePeaks } from './hooks/usePeaks.ts';
import { useEditor } from './store.ts';

const isTyping = (el: EventTarget | null): boolean =>
  el instanceof HTMLElement && ['INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName);

export function App() {
  const trackId = useEditor((s) => s.trackId);
  const loading = useEditor((s) => s.loading);
  const url = trackId ? api.audioUrl(trackId) : null;
  const audio = useAudio(url);
  const { peaks, error } = usePeaks(url, trackId);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const ed = useEditor.getState();
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === 's') {
        e.preventDefault();
        void ed.save();
        return;
      }
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) ed.redo();
        else ed.undo();
        return;
      }
      if (isTyping(e.target)) return;
      if (e.key === ' ') {
        e.preventDefault();
        audio.toggle();
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (ed.selectedSegment !== null) ed.deleteSegment(ed.selectedSegment);
        else if (ed.selectedBeat !== null) ed.deleteBeat(ed.selectedBeat);
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        const dt = (e.shiftKey ? 5 : 1) * (e.key === 'ArrowRight' ? 1 : -1);
        audio.seek((audio.audioRef.current?.currentTime ?? 0) + dt);
      } else if (e.key === '+' || e.key === '=') {
        ed.zoomAround(1.4, ed.view.viewportWidth / 2);
      } else if (e.key === '-') {
        ed.zoomAround(1 / 1.4, ed.view.viewportWidth / 2);
      } else if (e.key.toLowerCase() === 'e') {
        ed.toggleEval();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [audio]);

  return (
    <div className="app">
      <TrackList />
      <main className="center">
        <Transport audio={audio} />
        {!trackId ? (
          <div className="placeholder">Select a track to begin annotating.</div>
        ) : (
          <>
            {error && <div className="error">Audio decode failed: {error}</div>}
            {loading && <div className="placeholder">Loading…</div>}
            <Waveform peaks={peaks} audio={audio} />
            {!peaks && !error && <div className="dim hint-line">Decoding waveform…</div>}
            <StemLanes />
          </>
        )}
      </main>
      <Inspector audio={audio} />
    </div>
  );
}
