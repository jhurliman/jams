/** Shared annotation model — the single source of truth for client and server. */

/** The 11-class Raveform functional vocabulary, in the trained classifier's index order.
 *  Must stay in sync with `src/jams/data/structure_worker.py:_RAVEFORM_LABELS`. */
export const SECTION_LABELS = [
  'intro',
  'altintro',
  'buildup',
  'drop',
  'breakdown',
  'bridge',
  'cooldown',
  'outro',
  'altoutro',
  'start',
  'end',
] as const;

export type SectionLabel = (typeof SECTION_LABELS)[number];

/** A single beat. `bar`=1 marks a downbeat (bar start); 2..N are off-beats within the bar. */
export interface Beat {
  time: number;
  bar: number;
}

/** A functional structure segment. `start`/`end` in seconds. */
export interface Segment {
  start: number;
  end: number;
  label: SectionLabel;
}

/** Everything needed to render and edit one track. */
export interface Annotation {
  trackId: string;
  beats: Beat[];
  segments: Segment[];
}

export interface TrackMeta {
  id: string;
  title: string;
  genre: string;
  bpm: number;
  durationSec: number;
  /** Whether an edited annotation has been saved (vs. only the source reference exists). */
  edited: boolean;
}

export interface TrackListItem {
  id: string;
  title: string;
  genre: string;
  bpm: number;
  edited: boolean;
}
