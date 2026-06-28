import { useEffect, useState } from 'react';

/** Per-bucket absolute-peak amplitude for three frequency bands (Rekordbox-style coloring). */
export interface Peaks {
  low: Float32Array;
  mid: Float32Array;
  high: Float32Array;
  /** Seconds covered by each peak bucket. */
  bucketDur: number;
  duration: number;
}

// Render at a reduced rate (plenty for display; the high band only needs ~2 kHz) to cut memory.
const ANALYSIS_SR = 22050;
const SAMPLES_PER_BUCKET = 128; // ~5.8 ms/bucket
const LOW_HZ = 200;
const HIGH_HZ = 2000;

/** Fetch + decode the audio and reduce it to per-bucket low/mid/high peaks for a colored waveform.
 *  Three short offline filter passes; runs once per track on load. */
export function usePeaks(url: string | null): { peaks: Peaks | null; error: string | null } {
  const [peaks, setPeaks] = useState<Peaks | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!url) return;
    let cancelled = false;
    setPeaks(null);
    setError(null);

    (async () => {
      try {
        const bytes = await (await fetch(url)).arrayBuffer();
        const decodeCtx = new OfflineAudioContext(1, 1, ANALYSIS_SR);
        const buffer = await decodeCtx.decodeAudioData(bytes);
        if (cancelled) return;

        const [low, mid, high] = await Promise.all([
          bandPeaks(buffer, (ctx) => [filter(ctx, 'lowpass', LOW_HZ)]),
          bandPeaks(buffer, (ctx) => [
            filter(ctx, 'highpass', LOW_HZ),
            filter(ctx, 'lowpass', HIGH_HZ),
          ]),
          bandPeaks(buffer, (ctx) => [filter(ctx, 'highpass', HIGH_HZ)]),
        ]);
        if (cancelled) return;

        setPeaks({
          low,
          mid,
          high,
          bucketDur: SAMPLES_PER_BUCKET / ANALYSIS_SR,
          duration: buffer.duration,
        });
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [url]);

  return { peaks, error };
}

function filter(ctx: OfflineAudioContext, type: BiquadFilterType, hz: number): BiquadFilterNode {
  const f = ctx.createBiquadFilter();
  f.type = type;
  f.frequency.value = hz;
  return f;
}

/** Render the buffer through a filter chain at ANALYSIS_SR and reduce to per-bucket abs peaks. */
async function bandPeaks(
  buffer: AudioBuffer,
  makeFilters: (ctx: OfflineAudioContext) => BiquadFilterNode[],
): Promise<Float32Array> {
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
  const rendered = await ctx.startRendering();
  const data = rendered.getChannelData(0);

  const buckets = Math.ceil(data.length / SAMPLES_PER_BUCKET);
  const peaks = new Float32Array(buckets);
  for (let b = 0; b < buckets; b++) {
    const start = b * SAMPLES_PER_BUCKET;
    const end = Math.min(start + SAMPLES_PER_BUCKET, data.length);
    let m = 0;
    for (let i = start; i < end; i++) {
      const v = Math.abs(data[i]!);
      if (v > m) m = v;
    }
    peaks[b] = m;
  }
  return peaks;
}
