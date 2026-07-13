import { useEffect, useRef, useState } from 'react';

import { api } from '../api.ts';
import { useEditor } from '../store.ts';

/** Section-count slider: rethresholds the track's cached boundary activations
 *  (POST /api/tracks/:id/resegment — pure math on the jams side, so effectively instant)
 *  and replaces the annotation's segments with the result. Only rendered when the track
 *  has an activations blob (imported tracks short enough to analyze unchunked). Replacing
 *  is undoable (⌘Z) and, when the current segments carry manual edits, gated behind a
 *  one-time confirm per track. */
export function SectionSlider() {
  const trackId = useEditor((s) => s.trackId);
  const info = useEditor((s) => s.resegmentInfo);
  const segCount = useEditor((s) => s.annotation?.segments.length ?? 0);

  /** Slider position while a request is pending; null → track the annotation count. */
  const [pending, setPending] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  /** Track id the user has already OK'd replacing edited segments on. */
  const consent = useRef<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const seq = useRef(0);

  useEffect(() => {
    setPending(null);
    setError(null);
  }, [trackId]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  if (!trackId || !info || info.maxCount < 2) return null;

  const shown = pending ?? Math.min(segCount, info.maxCount);

  /** Debounced: dragging emits one request per pause, not one per step. */
  const request = (count: number) => {
    setPending(count);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      void (async () => {
        const ed = useEditor.getState();
        if (ed.trackId !== trackId) return;
        const edited = ed.dirty || (ed.meta?.edited ?? false);
        if (edited && consent.current !== trackId) {
          const ok = window.confirm(
            'Replace the current segments with a re-thresholded segmentation?\n' +
              'Beats are untouched, and ⌘Z restores the previous segments.',
          );
          if (!ok) {
            setPending(null);
            return;
          }
        }
        consent.current = trackId;
        const mySeq = ++seq.current;
        try {
          const out = await api.resegment(trackId, count);
          if (mySeq !== seq.current || useEditor.getState().trackId !== trackId) return;
          useEditor.getState().applyResegment(out.segments);
          setPending(null);
          setError(null);
        } catch (err) {
          if (mySeq !== seq.current) return;
          setPending(null);
          setError(err instanceof Error ? err.message : String(err));
        }
      })();
    }, 150);
  };

  return (
    <div className="panel section-slider">
      <h3>Sections</h3>
      <label>
        <input
          type="range"
          min={1}
          max={info.maxCount}
          step={1}
          value={shown}
          onChange={(e) => request(Number(e.target.value))}
        />
        <span className="count">{shown}</span>
      </label>
      <div className="dim">
        Re-thresholds the analysis boundaries ({info.maxCount} max) — ⌘Z to undo.
      </div>
      {error && <div className="error">{error}</div>}
    </div>
  );
}
