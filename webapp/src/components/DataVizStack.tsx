import { useRef } from 'react';

import type { AudioControls } from '../hooks/useAudio.ts';
import type { Peaks } from '../hooks/usePeaks.ts';
import { useTimelineWheel } from '../hooks/useTimelineWheel.ts';
import { Playhead } from './Playhead.tsx';
import { StemLanes } from './StemLanes.tsx';
import { Waveform } from './Waveform.tsx';

interface Props {
  peaks: Peaks | null;
  error: string | null;
  audio: AudioControls;
}

/** Vertical stack of time-aligned data-viz rows: the waveform (with beat grid / eval /
 *  segment bands) plus the per-stem piano-roll lanes. Every row consumes the store's shared
 *  view transform, so zoom/scroll/pan act on all of them at once; a single full-height
 *  playhead overlay and a shared wheel handler span the whole stack. */
export function DataVizStack({ peaks, error, audio }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  useTimelineWheel(ref);

  return (
    <div className="vizstack" ref={ref}>
      <Waveform peaks={peaks} audio={audio} />
      {!peaks && !error && <div className="dim hint-line">Decoding waveform…</div>}
      <StemLanes />
      <Playhead audio={audio} />
    </div>
  );
}
