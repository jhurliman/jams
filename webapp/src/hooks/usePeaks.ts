import { useEffect, useState } from 'react';

import { loadCachedPeaks, storeCachedPeaks } from '../peakCache.ts';

export interface BandSet {
  low: Float32Array;
  mid: Float32Array;
  high: Float32Array;
}

/** Per-bucket band amplitudes for a Rekordbox-style colored waveform.
 *  `peak` = absolute peak (transient halo), `rms` = energy (bright sustained core). */
export interface Peaks {
  peak: BandSet;
  rms: BandSet;
  /** Seconds covered by each bucket. */
  bucketDur: number;
  duration: number;
}

// Render at a reduced rate (the high band only needs ~2 kHz) to cut memory; fine buckets so
// the renderer can interpolate smoothly when zoomed in.
export const ANALYSIS_SR = 22050;
export const SAMPLES_PER_BUCKET = 48; // ~2.2 ms/bucket
const LOW_HZ = 200;
const HIGH_HZ = 2000;

export function usePeaks(
  url: string | null,
  trackId: string | null,
): { peaks: Peaks | null; error: string | null } {
  const [peaks, setPeaks] = useState<Peaks | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!url) return;
    let cancelled = false;
    setPeaks(null);
    setError(null);

    (async () => {
      try {
        if (trackId) {
          const cached = await loadCachedPeaks(trackId);
          if (cancelled) return;
          if (cached) {
            setPeaks(cached);
            return;
          }
        }
        const computed = await analyze(url);
        if (cancelled) return;
        setPeaks(computed);
        if (trackId) void storeCachedPeaks(trackId, computed);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [url, trackId]);

  return { peaks, error };
}

async function analyze(url: string): Promise<Peaks> {
  const bytes = await (await fetch(url)).arrayBuffer();
  const decodeCtx = new OfflineAudioContext(1, 1, ANALYSIS_SR);
  const buffer = await decodeCtx.decodeAudioData(bytes);

  const [low, mid, high] = await Promise.all([
    bandReduce(buffer, (ctx) => [filter(ctx, 'lowpass', LOW_HZ)]),
    bandReduce(buffer, (ctx) => [filter(ctx, 'highpass', LOW_HZ), filter(ctx, 'lowpass', HIGH_HZ)]),
    bandReduce(buffer, (ctx) => [filter(ctx, 'highpass', HIGH_HZ)]),
  ]);

  return {
    peak: { low: low.peak, mid: mid.peak, high: high.peak },
    rms: { low: low.rms, mid: mid.rms, high: high.rms },
    bucketDur: SAMPLES_PER_BUCKET / ANALYSIS_SR,
    duration: buffer.duration,
  };
}

function filter(ctx: OfflineAudioContext, type: BiquadFilterType, hz: number): BiquadFilterNode {
  const f = ctx.createBiquadFilter();
  f.type = type;
  f.frequency.value = hz;
  return f;
}

/** Render the buffer through a filter chain and reduce to per-bucket peak + RMS. */
async function bandReduce(
  buffer: AudioBuffer,
  makeFilters: (ctx: OfflineAudioContext) => BiquadFilterNode[],
): Promise<{ peak: Float32Array; rms: Float32Array }> {
  const length = Math.ceil(buffer.duration * ANALYSIS_SR);
  const ctx = new OfflineAudioContext(1, length, ANALYSIS_SR);
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  let node: AudioNode = src;
  for (const f of makeFilters(ctx)) {
    node.connect(f);
    node = f;
  }
  node.connect(ctx.destination);
  src.start();
  const data = (await ctx.startRendering()).getChannelData(0);

  const buckets = Math.ceil(data.length / SAMPLES_PER_BUCKET);
  const peak = new Float32Array(buckets);
  const rms = new Float32Array(buckets);
  for (let b = 0; b < buckets; b++) {
    const start = b * SAMPLES_PER_BUCKET;
    const end = Math.min(start + SAMPLES_PER_BUCKET, data.length);
    let m = 0;
    let sumsq = 0;
    for (let i = start; i < end; i++) {
      const v = data[i]!;
      const a = Math.abs(v);
      if (a > m) m = a;
      sumsq += v * v;
    }
    peak[b] = m;
    rms[b] = Math.sqrt(sumsq / Math.max(1, end - start));
  }
  return { peak, rms };
}
