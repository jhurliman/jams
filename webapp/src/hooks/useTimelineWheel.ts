import { useEffect } from 'react';

import { useEditor } from '../store.ts';

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

/** Non-passive wheel zoom/pan for the whole data-viz stack. Attaching this to the shared
 *  container (rather than only the waveform canvas) means ⌘/Ctrl+wheel zoom and plain-wheel
 *  pan work while hovering ANY row — the waveform or any stem lane — and, because every row
 *  reads the same store view transform, they all move together. */
export function useTimelineWheel(ref: React.RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const ed = useEditor.getState();
      if (e.ctrlKey || e.metaKey) {
        const rect = el.getBoundingClientRect();
        ed.zoomAround(Math.exp(-e.deltaY * 0.006), e.clientX - rect.left);
      } else {
        const { view: v, meta } = ed;
        const max = meta ? Math.max(0, meta.durationSec * v.pxPerSec - v.viewportWidth) : 0;
        const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        ed.setView({ scrollLeft: clamp(v.scrollLeft + delta, 0, max) });
      }
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [ref]);
}
