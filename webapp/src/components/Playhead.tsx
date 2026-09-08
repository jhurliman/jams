import { useEffect, useRef } from 'react';

import type { AudioControls } from '../hooks/useAudio.ts';
import { useEditor } from '../store.ts';

/** Full-height vertical playback-position line spanning the ENTIRE data-viz stack — the
 *  waveform plus every stem lane below it — rather than a per-row line confined to the
 *  waveform. A single absolutely-positioned overlay aligned to the shared time->x transform
 *  (view.pxPerSec / view.scrollLeft, the same transform every row consumes), so it stays in
 *  sync with all rows through zoom, scroll and playback. */
export function Playhead({ audio }: { audio: AudioControls }) {
  const lineRef = useRef<HTMLDivElement | null>(null);
  // Re-run positioning whenever the shared transform changes (zoom / scroll — including
  // while paused, where currentTime alone fires no event).
  const view = useEditor((s) => s.view);
  const isPlaying = audio.isPlaying;

  useEffect(() => {
    const place = (): void => {
      const el = lineRef.current;
      if (!el) return;
      const { pxPerSec, scrollLeft, viewportWidth } = useEditor.getState().view;
      const x = (audio.audioRef.current?.currentTime ?? 0) * pxPerSec - scrollLeft;
      if (x >= 0 && x <= viewportWidth) {
        el.style.transform = `translateX(${x}px)`;
        el.style.visibility = 'visible';
      } else {
        el.style.visibility = 'hidden';
      }
    };

    place(); // immediate reposition (covers zoom/scroll while paused via the `view` dep)

    // Seeks while paused change no React state; reposition on the audio 'seeked' event
    // (covers every seek source: waveform clicks, transport buttons, keyboard).
    const audioEl = audio.audioRef.current;
    audioEl?.addEventListener('seeked', place);

    let raf = 0;
    if (isPlaying) {
      const tick = (): void => {
        place();
        raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    }
    return () => {
      audioEl?.removeEventListener('seeked', place);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [audio, isPlaying, view]);

  return <div className="playhead" ref={lineRef} aria-hidden="true" />;
}
