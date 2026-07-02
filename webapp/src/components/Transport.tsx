import { useEffect, useState } from 'react';

import type { AudioControls } from '../hooks/useAudio.ts';
import { useEditor } from '../store.ts';

const fmt = (t: number): string => {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  const cs = Math.floor((t % 1) * 100);
  return `${m}:${s.toString().padStart(2, '0')}.${cs.toString().padStart(2, '0')}`;
};

export function Transport({ audio }: { audio: AudioControls }) {
  const {
    meta,
    dirty,
    saving,
    save,
    undo,
    redo,
    past,
    future,
    zoomAround,
    view,
    setView,
    showEval,
    toggleEval,
    prediction,
  } = useEditor();
  const [now, setNow] = useState(0);
  const [copied, setCopied] = useState(false);

  const copyId = () => {
    if (!meta) return;
    void navigator.clipboard.writeText(meta.id);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  useEffect(() => {
    const id = setInterval(() => setNow(audio.audioRef.current?.currentTime ?? 0), 80);
    return () => clearInterval(id);
  }, [audio]);

  const fit = () => {
    if (!meta) return;
    setView({ pxPerSec: view.viewportWidth / Math.max(meta.durationSec, 1), scrollLeft: 0 });
  };

  return (
    <div className="transport">
      <button className="btn primary" onClick={audio.toggle} title="Play / pause (space)">
        {audio.isPlaying ? '❚❚' : '▶'}
      </button>
      <span className="time">
        {fmt(now)} <span className="dim">/ {meta ? fmt(meta.durationSec) : '0:00'}</span>
      </span>

      <div className="spacer" />

      {meta && (
        <span className="trackinfo">
          <strong>{meta.title}</strong>
          <button
            className="chip id"
            onClick={copyId}
            title="Click to copy track ID"
          >
            {copied ? '✓ copied' : meta.id}
          </button>
          <span className="chip">{meta.genre}</span>
          <span className="chip">{meta.bpm} BPM</span>
        </span>
      )}

      <div className="spacer" />

      <button
        className={`btn ${showEval && prediction ? 'toggled' : ''}`}
        onClick={toggleEval}
        disabled={!prediction}
        title={prediction ? 'Toggle the read-only model prediction (eval) lane' : 'No eval for this track'}
      >
        {showEval && prediction ? '◉ Eval' : '○ Eval'}
      </button>
      <button className="btn" onClick={() => zoomAround(1 / 1.4, view.viewportWidth / 2)} title="Zoom out">
        −
      </button>
      <button className="btn" onClick={() => zoomAround(1.4, view.viewportWidth / 2)} title="Zoom in">
        +
      </button>
      <button className="btn" onClick={fit} title="Fit to window">
        ⤢
      </button>
      <button className="btn" onClick={undo} disabled={!past.length} title="Undo (⌘Z)">
        ↶
      </button>
      <button className="btn" onClick={redo} disabled={!future.length} title="Redo (⌘⇧Z)">
        ↷
      </button>
      <button
        className={`btn ${dirty ? 'primary' : ''}`}
        onClick={() => void save()}
        disabled={!dirty || saving}
        title="Save (⌘S)"
      >
        {saving ? 'Saving…' : dirty ? '● Save' : 'Saved'}
      </button>
    </div>
  );
}
