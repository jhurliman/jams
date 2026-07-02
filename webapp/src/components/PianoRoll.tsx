import { useEffect, useRef } from 'react';

import type { StemTranscription } from '../../shared/types.ts';
import { useEditor } from '../store.ts';

const H = 84;

/** Fixed GM percussion lanes (top -> bottom), each matching a set of GM note numbers. */
const DRUM_LANES: { name: string; notes: number[]; color: string }[] = [
  { name: 'cymbals', notes: [49, 51, 52, 53, 55, 57, 59], color: '#c9a227' },
  { name: 'hats', notes: [42, 44, 46], color: '#6aa0ff' },
  { name: 'snare', notes: [37, 38, 39, 40], color: '#e8467c' },
  { name: 'toms', notes: [41, 43, 45, 47, 48, 50], color: '#9b59ff' },
  { name: 'kick', notes: [35, 36], color: '#3fb950' },
];

/** Map a GM percussion note to a lane index; unknown notes fall in the last (kick) lane. */
function drumLane(pitch: number): number {
  const idx = DRUM_LANES.findIndex((l) => l.notes.includes(pitch));
  return idx >= 0 ? idx : DRUM_LANES.length - 1;
}

interface Props {
  transcription: StemTranscription;
}

/** Read-only piano-roll for a single stem. X reuses the waveform's time transform
 *  (t * pxPerSec - scrollLeft) so it stays zoom/scroll-synced with the Waveform. */
export function PianoRoll({ transcription }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawRef = useRef<() => void>(() => {});

  const view = useEditor((s) => s.view);

  const draw = (): void => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const { pxPerSec, scrollLeft, viewportWidth: W } = useEditor.getState().view;
    const dpr = window.devicePixelRatio || 1;
    const dw = Math.round(W * dpr);
    const dh = Math.round(H * dpr);
    if (canvas.width !== dw || canvas.height !== dh) {
      canvas.width = dw;
      canvas.height = dh;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0d13';
    ctx.fillRect(0, 0, W, H);

    const { notes, isDrums } = transcription;

    if (isDrums) {
      const laneH = H / DRUM_LANES.length;
      // lane guides
      ctx.strokeStyle = '#161b22';
      ctx.lineWidth = 1;
      for (let i = 1; i < DRUM_LANES.length; i++) {
        const y = Math.round(i * laneH) + 0.5;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }
      for (const n of notes) {
        const x = n.onset * pxPerSec - scrollLeft;
        const w = Math.max(2, (n.offset - n.onset) * pxPerSec);
        if (x + w < 0 || x > W) continue;
        const lane = drumLane(n.pitch);
        const y = lane * laneH;
        ctx.fillStyle = DRUM_LANES[lane]!.color;
        ctx.globalAlpha = 0.25 + 0.75 * (n.velocity / 127);
        ctx.fillRect(x, y + 2, w, laneH - 4);
      }
      ctx.globalAlpha = 1;
      // lane labels
      ctx.fillStyle = '#7d8694';
      ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
      ctx.textBaseline = 'middle';
      DRUM_LANES.forEach((l, i) => {
        ctx.fillText(l.name, 4, i * laneH + laneH / 2);
      });
      ctx.textBaseline = 'alphabetic';
      return;
    }

    // pitched stem: map MIDI pitch to vertical position across the note range present.
    let lo = 127;
    let hi = 0;
    for (const n of notes) {
      if (n.pitch < lo) lo = n.pitch;
      if (n.pitch > hi) hi = n.pitch;
    }
    if (hi < lo) {
      lo = 48;
      hi = 72;
    }
    const pad = 2;
    const span = Math.max(1, hi - lo);
    const noteH = Math.max(2, Math.min(10, (H - 2 * pad) / (span + 1)));
    const pitchToY = (p: number): number => H - pad - ((p - lo) / span) * (H - 2 * pad - noteH);

    ctx.fillStyle = '#4f8cff';
    for (const n of notes) {
      const x = n.onset * pxPerSec - scrollLeft;
      const w = Math.max(2, (n.offset - n.onset) * pxPerSec);
      if (x + w < 0 || x > W) continue;
      ctx.globalAlpha = 0.25 + 0.75 * (n.velocity / 127);
      ctx.fillRect(x, pitchToY(n.pitch), w, noteH);
    }
    ctx.globalAlpha = 1;
  };
  drawRef.current = draw;

  useEffect(() => {
    drawRef.current();
  }, [view, transcription]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: H, display: 'block' }}
      aria-label={`${transcription.stemType} piano roll`}
    />
  );
}
