import { useCallback, useEffect, useRef } from 'react';

import type { AudioControls } from '../hooks/useAudio.ts';
import type { BandSet, Peaks } from '../hooks/usePeaks.ts';
import { labelColor } from '../labels.ts';
import { useEditor } from '../store.ts';

const H = 260;
const RULER_H = 22;
const EVAL_SEG_H = 22;
const EVAL_BEAT_H = 14;
const EVAL_H = EVAL_SEG_H + EVAL_BEAT_H;
const BOUNDARY_HIT = 6;
const BEAT_HIT = 4;

const GT_DOWNBEAT = '70,202,216'; // cyan
const EVAL_DOWNBEAT = '245,184,74'; // amber

// Waveform palette: peak = faint transient halo, rms = bright sustained core; coloured by band.
const PEAK_COLORS = { high: '#99a4cc', mid: '#b07724', low: '#27538f' };
const RMS_COLORS = { high: '#eef2ff', mid: '#ffab3c', low: '#3f86ff' };

const contentTop = (): number => {
  const { showEval, prediction } = useEditor.getState();
  return RULER_H + (showEval && prediction?.segments.length ? EVAL_H : 0);
};

interface DragState {
  kind: 'boundary' | 'beat';
  index: number;
  neighbor?: number;
}

interface Props {
  peaks: Peaks | null;
  audio: AudioControls;
}

export function Waveform({ peaks, audio }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const staticKeyRef = useRef<string>('');
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

  // --- static layer: everything except the playhead, rendered to the offscreen canvas ---
  const renderStatic = useCallback(
    (ctx: CanvasRenderingContext2D, W: number) => {
      const { annotation, view, selectedSegment, selectedBeat, meta, prediction, showEval } =
        useEditor.getState();
      const { pxPerSec, scrollLeft } = view;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#0e1117';
      ctx.fillRect(0, 0, W, H);

      const t0 = scrollLeft / pxPerSec;
      const t1 = (scrollLeft + W) / pxPerSec;
      const evalActive = showEval && !!prediction?.segments.length;
      const top = RULER_H + (evalActive ? EVAL_H : 0);

      // eval lane (read-only model prediction): segment blocks + a beat strip below
      if (evalActive && prediction) {
        const segBottom = RULER_H + EVAL_SEG_H;
        ctx.fillStyle = '#0a0d13';
        ctx.fillRect(0, RULER_H, W, EVAL_H);
        for (const seg of prediction.segments) {
          const x = seg.start * pxPerSec - scrollLeft;
          const w = (seg.end - seg.start) * pxPerSec;
          if (x + w < 0 || x > W) continue;
          ctx.fillStyle = labelColor(seg.label) + 'cc';
          ctx.fillRect(x + 0.5, RULER_H + 3, Math.max(1, w - 1), EVAL_SEG_H - 5);
          if (w > 26) {
            ctx.fillStyle = '#0b0e14';
            ctx.font = '600 11px ui-sans-serif, system-ui, sans-serif';
            ctx.fillText(seg.label, Math.max(x, 0) + 5, RULER_H + 15);
          }
        }
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
        ctx.strokeStyle = '#2a3242';
        ctx.beginPath();
        ctx.moveTo(0, segBottom + 0.5);
        ctx.lineTo(W, segBottom + 0.5);
        ctx.moveTo(0, top + 0.5);
        ctx.lineTo(W, top + 0.5);
        ctx.stroke();
      }

      // ground-truth segment bands
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
        });
      }

      if (peaks) drawWaveform(ctx, peaks, W, top, scrollLeft, pxPerSec);

      // ground-truth beats (cyan downbeats; density-aware)
      if (annotation?.beats.length) {
        const beatPx = meta ? (60 / Math.max(meta.bpm, 1)) * pxPerSec : 12;
        const showOffbeats = beatPx >= 6;
        const downbeatAlpha = showOffbeats ? 0.85 : clampNum((beatPx * 4) / 70, 0.35, 0.7);
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

      // ground-truth segment labels — drawn on top of the waveform with a dark backdrop chip so
      // they stay legible over the (light) waveform and coloured band tint.
      if (annotation) {
        ctx.font = '600 12px ui-sans-serif, system-ui, sans-serif';
        ctx.textBaseline = 'middle';
        for (const seg of annotation.segments) {
          const x = seg.start * pxPerSec - scrollLeft;
          const w = (seg.end - seg.start) * pxPerSec;
          if (x + w < 0 || x > W) continue;
          const lx = Math.max(x, 0) + 6;
          const ly = top + 9;
          const tw = ctx.measureText(seg.label).width;
          ctx.fillStyle = 'rgba(8,10,15,0.78)';
          ctx.fillRect(lx - 4, ly - 9, tw + 8, 18);
          ctx.fillStyle = labelColor(seg.label);
          ctx.fillText(seg.label, lx, ly + 1);
        }
        ctx.textBaseline = 'alphabetic';
      }

      // ruler
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
    },
    [peaks],
  );

  // --- composite: blit the cached static layer. The moving playback line is a separate
  //     full-height DOM overlay (see Playhead) that spans the waveform AND the stem lanes. ---
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const { view } = useEditor.getState();
    const W = view.viewportWidth;
    const dpr = window.devicePixelRatio || 1;
    const dw = Math.round(W * dpr);
    const dh = Math.round(H * dpr);
    if (canvas.width !== dw || canvas.height !== dh) {
      canvas.width = dw;
      canvas.height = dh;
    }
    let off = offscreenRef.current;
    if (!off) {
      off = document.createElement('canvas');
      offscreenRef.current = off;
    }
    if (off.width !== dw || off.height !== dh) {
      off.width = dw;
      off.height = dh;
      staticKeyRef.current = '';
    }

    const st = useEditor.getState();
    const key = [
      st.view.pxPerSec,
      st.view.scrollLeft,
      W,
      dpr,
      st.showEval,
      st.prediction ? 1 : 0,
      st.selectedSegment,
      st.selectedBeat,
      st.rev,
      peaks ? peaks.duration : -1,
    ].join('|');
    if (key !== staticKeyRef.current) {
      const octx = off.getContext('2d')!;
      octx.setTransform(dpr, 0, 0, dpr, 0, 0);
      renderStatic(octx, W);
      staticKeyRef.current = key;
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, dw, dh);
    ctx.drawImage(off, 0, 0);
  }, [peaks, renderStatic]);

  drawRef.current = draw;

  const annotation = useEditor((s) => s.annotation);
  const view = useEditor((s) => s.view);
  const selectedSegment = useEditor((s) => s.selectedSegment);
  const selectedBeat = useEditor((s) => s.selectedBeat);
  const prediction = useEditor((s) => s.prediction);
  const showEval = useEditor((s) => s.showEval);
  const rev = useEditor((s) => s.rev);
  useEffect(() => {
    draw();
  }, [draw, annotation, view, selectedSegment, selectedBeat, prediction, showEval, rev]);

  // Playback loop: auto-scroll to keep the (overlay) playhead on screen. Every scrollLeft
  // change flows through the shared view transform, moving all rows + the playhead together.
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
    const { annotation: ann, view: v } = useEditor.getState();
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
    const t = ann.beats[best]!.time;
    const prev = ann.beats[best - 1]?.time ?? t - 1;
    const next = ann.beats[best + 1]?.time ?? t + 1;
    return Math.min(t - prev, next - t) * v.pxPerSec >= 10 ? best : null;
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
    const t = xToTime(x);
    const segIdx = ed.annotation?.segments.findIndex((s) => t >= s.start && t < s.end) ?? -1;
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

  // Wheel zoom/pan is handled once for the whole data-viz stack (see useTimelineWheel),
  // so it acts over the waveform and every stem lane alike.

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

/** Two-layer 3-band waveform: faint peak halo + bright RMS core, coloured center-out by band. */
function drawWaveform(
  ctx: CanvasRenderingContext2D,
  peaks: Peaks,
  W: number,
  top: number,
  scrollLeft: number,
  pxPerSec: number,
) {
  const midY = (H + top) / 2;
  const amp = (H - top) / 2 - 3;
  const bucketsPerPx = 1 / (peaks.bucketDur * pxPerSec);
  const half = bucketsPerPx / 2;

  const sample = (arr: Float32Array, center: number): number => {
    if (half >= 0.5) {
      let lo = Math.max(0, Math.floor(center - half));
      const hi = Math.min(arr.length, Math.ceil(center + half));
      let m = 0;
      for (; lo < hi; lo++) if (arr[lo]! > m) m = arr[lo]!;
      return m;
    }
    const i = Math.floor(center);
    const f = center - i;
    const a = arr[i] ?? 0;
    return a * (1 - f) + (arr[i + 1] ?? a) * f;
  };

  const heights = (bands: BandSet) => {
    const env = new Float32Array(W);
    const amber = new Float32Array(W);
    const blue = new Float32Array(W);
    for (let px = 0; px < W; px++) {
      const center = (px + 0.5 + scrollLeft) * bucketsPerPx;
      const l = sample(bands.low, center);
      const m = sample(bands.mid, center);
      const h = sample(bands.high, center);
      const s = l + m + h;
      if (s <= 0.002) continue;
      const e = Math.min(amp, Math.sqrt(s) * amp * 0.72);
      env[px] = e;
      amber[px] = (e * (l + m)) / s;
      blue[px] = (e * l) / s;
    }
    return { env, amber, blue };
  };

  const fill = (hgt: Float32Array, color: string) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(0, midY - hgt[0]!);
    for (let px = 1; px < W; px++) ctx.lineTo(px + 0.5, midY - hgt[px]!);
    for (let px = W - 1; px >= 0; px--) ctx.lineTo(px + 0.5, midY + hgt[px]!);
    ctx.closePath();
    ctx.fill();
  };

  const pk = heights(peaks.peak);
  fill(pk.env, PEAK_COLORS.high);
  fill(pk.amber, PEAK_COLORS.mid);
  fill(pk.blue, PEAK_COLORS.low);
  const rm = heights(peaks.rms);
  fill(rm.env, RMS_COLORS.high);
  fill(rm.amber, RMS_COLORS.mid);
  fill(rm.blue, RMS_COLORS.low);
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
