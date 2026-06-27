import { useEffect, useState } from 'react';

export interface Peaks {
  min: Float32Array;
  max: Float32Array;
  /** Seconds covered by each peak bucket. */
  bucketDur: number;
  duration: number;
}

const SAMPLES_PER_BUCKET = 256;

/** Fetch + decode the audio and reduce it to per-bucket min/max amplitude for waveform drawing.
 *  One pass over the samples (~tens of ms for a few-minute track); good enough on the main thread. */
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
        const buf = await (await fetch(url)).arrayBuffer();
        const ctx = new OfflineAudioContext(1, 1, 44100);
        const audio = await ctx.decodeAudioData(buf);
        if (cancelled) return;

        const channels = Array.from({ length: audio.numberOfChannels }, (_, c) =>
          audio.getChannelData(c),
        );
        const n = audio.length;
        const buckets = Math.ceil(n / SAMPLES_PER_BUCKET);
        const min = new Float32Array(buckets);
        const max = new Float32Array(buckets);

        for (let b = 0; b < buckets; b++) {
          const start = b * SAMPLES_PER_BUCKET;
          const end = Math.min(start + SAMPLES_PER_BUCKET, n);
          let lo = 0;
          let hi = 0;
          for (let i = start; i < end; i++) {
            let s = 0;
            for (const ch of channels) s += ch[i]!;
            s /= channels.length;
            if (s < lo) lo = s;
            if (s > hi) hi = s;
          }
          min[b] = lo;
          max[b] = hi;
        }

        if (!cancelled) {
          setPeaks({
            min,
            max,
            bucketDur: SAMPLES_PER_BUCKET / audio.sampleRate,
            duration: audio.duration,
          });
        }
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
