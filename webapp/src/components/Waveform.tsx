import { useCallback, useEffect, useRef } from 'react';

import type { AudioControls } from '../hooks/useAudio.ts';
import type { Peaks } from '../hooks/usePeaks.ts';
import { labelColor } from '../labels.ts';
import { useEditor } from '../store.ts';

const H = 260;
const RULER_H = 22;
const EVAL_SEG_H = 22;
const EVAL_BEAT_H = 14;
const EVAL_H = EVAL_SEG_H + EVAL_BEAT_H;
const BOUNDARY_HIT = 6;
const BEAT_HIT = 4;

// Downbeats get a distinct accent so the bar grid reads at a glance; eval uses a warm hue.
const GT_DOWNBEAT = '70,202,216'; // cyan
const EVAL_DOWNBEAT = '245,184,74'; // amber

/** Top of the editable (ground-truth) content; the eval lane occupies RULER_H..contentTop. */
const contentTop = (): number => {
  const { showEval, prediction } = useEditor.getState();
  return RULER_H + (showEval && prediction?.segments.length ? EVAL_H : 0);
};

interface DragState {
  kind: 'boundary' | 'beat';
  index: number;
  /** For a boundary drag: also move the start of the next segment to keep them contiguous. */
  neighbor?: number;
}

interface Props {
  peaks: Peaks | null;
  audio: AudioControls;
}

export function Waveform({ peaks, audio }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const drag = useRef<DragState | null>(null);
  const drawRef = useRef<() => void>(() => {});

  const timeToX = useCallback((t: number): number => {
    const { pxPerSec, scrollLeft } = useEditor.getState().view;
    return t * pxPerSec - scrollLeft;
  }, []);
  const xToTime = useCallback((x: number): number => {
    const { pxPerSec, scrollLeft } = useEditor.getState().view;
    return (x + scrollLeft) / pxPerSec;
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const { annotation, view, selectedSegment, selectedBeat, meta, prediction, showEval } =
      useEditor.getState();
    const W = view.viewportWidth;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0e1117';
    ctx.fillRect(0, 0, W, H);

    const { pxPerSec, scrollLeft } = view;
    const t0 = scrollLeft / pxPerSec;
    const t1 = (scrollLeft + W) / pxPerSec;
    const evalActive = showEval && !!prediction?.segments.length;
    const top = RULER_H + (evalActive ? EVAL_H : 0);

    // --- eval lane (read-only model prediction): segment blocks + a beat strip below ---
    if (evalActive && prediction) {
      const segBottom = RULER_H + EVAL_SEG_H;
      ctx.fillStyle = '#0a0d13';
      ctx.fillRect(0, RULER_H, W, EVAL_H);
      // predicted segments
      for (const seg of prediction.segments) {
        const x = seg.start * pxPerSec - scrollLeft;
        const w = (seg.end - seg.start) * pxPerSec;
        if (x + w < 0 || x > W) continue;
        const color = labelColor(seg.label);
        ctx.fillStyle = color + 'cc';
        ctx.fillRect(x + 0.5, RULER_H + 3, Math.max(1, w - 1), EVAL_SEG_H - 5);
        if (w > 26) {
          ctx.fillStyle = '#0b0e14';
          ctx.font = '600 11px ui-sans-serif, system-ui, sans-serif';
          ctx.fillText(seg.label, Math.max(x, 0) + 5, RULER_H + 15);
        }
      }
      // predicted beats (density-gated; downbeats amber, off-beats dim)
      const ebeatPx = meta ? (60 / Math.max(meta.bpm, 1)) * pxPerSec : 12;
      const eShowOff = ebeatPx >= 6;
      ctx.lineWidth = 1;
      for (const b of prediction.beats) {
        if (b.time < t0 - 0.1 || b.time > t1 + 0.1) continue;
        const down = b.bar === 1;
        if (!down && !eShowOff) continue;
        const x = Math.round(b.time * pxPerSec - scrollLeft) + 0.5;
        ctx.strokeStyle = down ? `rgba(${EVAL_DOWNBEAT},0.95)` : 'rgba(150,140,118,0.5)';
        ctx.beginPath();
        ctx.moveTo(x, down ? segBottom + 1 : top - 4);
        ctx.lineTo(x, top - 1);
        ctx.stroke();
      }
      // separators
      ctx.strokeStyle = '#2a3242';
      ctx.beginPath();
      ctx.moveTo(0, segBottom + 0.5);
      ctx.lineTo(W, segBottom + 0.5);
      ctx.moveTo(0, top + 0.5);
      ctx.lineTo(W, top + 0.5);
      ctx.stroke();
    }

    // --- segment bands (translucent fill + boundaries + label) ---
    if (annotation) {
      annotation.segments.forEach((seg, i) => {
        const x = seg.start * pxPerSec - scrollLeft;
        const w = (seg.end - seg.start) * pxPerSec;
        if (x + w < 0 || x > W) return;
        const color = labelColor(seg.label);
        ctx.fillStyle = color + (i === selectedSegment ? '40' : '22');
        ctx.fillRect(x, top, w, H - top);
        ctx.strokeStyle = color;
        ctx.lineWidth = i === selectedSegment ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(x + 0.5, top);
        ctx.lineTo(x + 0.5, H);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.font = '600 12px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(seg.label, Math.max(x, 0) + 6, top + 14);
      });
    }

    // --- waveform (min/max per pixel column) ---
    if (peaks) {
      const mid = (H + top) / 2;
      const amp = (H - top) / 2 - 4;
      ctx.fillStyle = '#6f86b3';
      for (let px = 0; px < W; px++) {
        const ta = (px + scrollLeft) / pxPerSec;
        const tb = (px + 1 + scrollLeft) / pxPerSec;
        let bi = Math.floor(ta / peaks.bucketDur);
        const bEnd = Math.min(Math.ceil(tb / peaks.bucketDur), peaks.max.length);
        let lo = 0;
        let hi = 0;
        for (; bi < bEnd; bi++) {
          if (peaks.min[bi]! < lo) lo = peaks.min[bi]!;
          if (peaks.max[bi]! > hi) hi = peaks.max[bi]!;
        }
        const y1 = mid - hi * amp;
        const y2 = mid - lo * amp;
        ctx.fillRect(px, y1, 1, Math.max(1, y2 - y1));
      }
    }

    // --- beats (density-aware so the waveform stays the hero when zoomed out) ---
    if (annotation && annotation.beats.length) {
      const beatPx = meta ? (60 / Math.max(meta.bpm, 1)) * pxPerSec : 12;
      const showOffbeats = beatPx >= 6;
      // Downbeats fade as the bar grid gets dense; off-beats only appear once zoomed in.
      const downbeatAlpha = showOffbeats ? 0.85 : clampNum((beatPx * 4) / 70, 0.35, 0.7);
      // As downbeats marks only (when off-beats hidden), keep them as short top ticks rather than
      // full-height lines so they read as a grid, not bars across the waveform.
      const downbeatTop = showOffbeats ? top : top + 4;
      const downbeatBottom = showOffbeats ? H : top + 12;
      ctx.lineWidth = 1;
      for (let i = 0; i < annotation.beats.length; i++) {
        const beat = annotation.beats[i]!;
        if (beat.time < t0 - 0.1 || beat.time > t1 + 0.1) continue;
        const downbeat = beat.bar === 1;
        const selected = i === selectedBeat;
        if (!downbeat && !showOffbeats && !selected) continue;
        const x = Math.round(beat.time * pxPerSec - scrollLeft) + 0.5;
        let y0: number;
        let bottom = H;
        if (selected) {
          ctx.strokeStyle = '#ffd24a';
          ctx.lineWidth = 2;
          y0 = top;
        } else if (downbeat) {
          ctx.strokeStyle = `rgba(${GT_DOWNBEAT},${downbeatAlpha})`;
          ctx.lineWidth = 1;
          y0 = downbeatTop;
          bottom = downbeatBottom;
        } else {
          ctx.strokeStyle = 'rgba(124,136,158,0.3)';
          ctx.lineWidth = 1;
          y0 = H - 30;
        }
        ctx.beginPath();
        ctx.moveTo(x, y0);
        ctx.lineTo(x, bottom);
        ctx.stroke();
      }
    }

    // --- ruler ---
    ctx.fillStyle = '#161b22';
    ctx.fillRect(0, 0, W, RULER_H);
    ctx.fillStyle = '#7d8694';
    ctx.font = '10px ui-monospace, monospace';
    const step = niceStep(pxPerSec);
    const first = Math.ceil(t0 / step) * step;
    for (let t = first; t < t1; t += step) {
      const x = t * pxPerSec - scrollLeft;
      ctx.fillRect(x, RULER_H - 5, 1, 5);
      ctx.fillText(fmtTime(t), x + 3, 11);
    }

    // --- playhead ---
    const cur = audio.audioRef.current?.currentTime ?? 0;
    const px = cur * pxPerSec - scrollLeft;
    if (px >= 0 && px <= W) {
      ctx.strokeStyle = '#39d353';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, H);
      ctx.stroke();
    }
    void meta;
  }, [peaks, audio]);

  drawRef.current = draw;

  // Redraw whenever annotation / view / selection / peaks change.
  const annotation = useEditor((s) => s.annotation);
  const view = useEditor((s) => s.view);
  const selectedSegment = useEditor((s) => s.selectedSegment);
  const selectedBeat = useEditor((s) => s.selectedBeat);
  const prediction = useEditor((s) => s.prediction);
  const showEval = useEditor((s) => s.showEval);
  useEffect(() => {
    draw();
  }, [draw, annotation, view, selectedSegment, selectedBeat, prediction, showEval]);

  // Animation loop while playing: advance playhead, auto-scroll to keep it on screen.
  useEffect(() => {
    if (!audio.isPlaying) return;
    let raf = 0;
    const tick = () => {
      const { view: v, meta } = useEditor.getState();
      const cur = audio.audioRef.current?.currentTime ?? 0;
      const px = cur * v.pxPerSec - v.scrollLeft;
      if (meta && (px > v.viewportWidth * 0.85 || px < 0)) {
        const maxScroll = Math.max(0, meta.durationSec * v.pxPerSec - v.viewportWidth);
        const scrollLeft = Math.min(maxScroll, Math.max(0, cur * v.pxPerSec - v.viewportWidth * 0.5));
        useEditor.getState().setView({ scrollLeft });
      }
      drawRef.current();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [audio.isPlaying, audio.audioRef]);

  // Track viewport width.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      useEditor.getState().setView({ viewportWidth: el.clientWidth });
    });
    ro.observe(el);
    useEditor.getState().setView({ viewportWidth: el.clientWidth });
    return () => ro.disconnect();
  }, []);

  // ---- interaction ----
  const hitBoundary = (x: number): DragState | null => {
    const ann = useEditor.getState().annotation;
    if (!ann) return null;
    for (let i = 0; i < ann.segments.length; i++) {
      const seg = ann.segments[i]!;
      if (Math.abs(timeToX(seg.start) - x) < BOUNDARY_HIT) {
        const neighbor = ann.segments.findIndex((s) => Math.abs(s.end - seg.start) < 1e-6);
        return { kind: 'boundary', index: i, ...(neighbor >= 0 ? { neighbor } : {}) };
      }
    }
    return null;
  };
  const hitBeat = (x: number): number | null => {
    const { annotation: ann, view } = useEditor.getState();
    if (!ann) return null;
    let best = -1;
    let bestD = BEAT_HIT;
    for (let i = 0; i < ann.beats.length; i++) {
      const d = Math.abs(timeToX(ann.beats[i]!.time) - x);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best < 0) return null;
    // Only targetable when neighbouring beats are far enough apart on screen; otherwise the
    // grid is too dense (zoomed out) and the click should select the segment band instead.
    const t = ann.beats[best]!.time;
    const prev = ann.beats[best - 1]?.time ?? t - 1;
    const next = ann.beats[best + 1]?.time ?? t + 1;
    const spacingPx = Math.min(t - prev, next - t) * view.pxPerSec;
    return spacingPx >= 10 ? best : null;
  };

  const capturePointer = (id: number) => {
    try {
      canvasRef.current?.setPointerCapture(id);
    } catch {
      /* synthetic / already-released pointer */
    }
  };

  const onPointerDown = (e: React.PointerEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const ed = useEditor.getState();
    // Clicks in the ruler or the read-only eval lane just seek — only ground truth is editable.
    if (y < contentTop()) {
      audio.seek(xToTime(x));
      return;
    }
    const b = hitBoundary(x);
    if (b) {
      drag.current = b;
      capturePointer(e.pointerId);
      return;
    }
    const beat = hitBeat(x);
    if (beat !== null) {
      if (e.altKey) {
        ed.cycleBeatBar(beat);
        return;
      }
      ed.selectBeat(beat);
      drag.current = { kind: 'beat', index: beat };
      capturePointer(e.pointerId);
      return;
    }
    const ann = ed.annotation;
    const t = xToTime(x);
    const segIdx = ann?.segments.findIndex((s) => t >= s.start && t < s.end) ?? -1;
    if (segIdx >= 0) ed.selectSegment(segIdx);
    audio.seek(t);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const t = Math.max(0, xToTime(e.clientX - rect.left));
    const ed = useEditor.getState();
    if (drag.current.kind === 'boundary') {
      ed.updateSegment(drag.current.index, { start: t });
      if (drag.current.neighbor !== undefined) ed.updateSegment(drag.current.neighbor, { end: t });
    } else {
      ed.moveBeat(drag.current.index, t);
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (drag.current) canvasRef.current!.releasePointerCapture(e.pointerId);
    drag.current = null;
  };

  const onDoubleClick = (e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ed = useEditor.getState();
    const beat = hitBeat(x);
    if (beat !== null) ed.deleteBeat(beat);
    else ed.addBeat(xToTime(x));
  };

  // Wheel/pinch must use a NON-passive native listener: React attaches `wheel` passively, so a
  // synthetic onWheel can't preventDefault() — the browser would then also pinch-zoom the page.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const ed = useEditor.getState();
      if (e.ctrlKey || e.metaKey) {
        // ctrlKey wheel = trackpad pinch (or ctrl+scroll). Scale continuously by magnitude.
        const rect = canvas.getBoundingClientRect();
        ed.zoomAround(Math.exp(-e.deltaY * 0.006), e.clientX - rect.left);
      } else {
        const { view: v, meta } = ed;
        const max = meta ? Math.max(0, meta.durationSec * v.pxPerSec - v.viewportWidth) : 0;
        const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        ed.setView({ scrollLeft: clampNum(v.scrollLeft + delta, 0, max) });
      }
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, []);

  return (
    <div ref={containerRef} className="waveform">
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: H, display: 'block', touchAction: 'none' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onDoubleClick={onDoubleClick}
      />
    </div>
  );
}

const clampNum = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

function niceStep(pxPerSec: number): number {
  const target = 80 / pxPerSec;
  const steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120];
  return steps.find((s) => s >= target) ?? 240;
}

function fmtTime(t: number): string {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
