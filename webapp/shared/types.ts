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
  /** Whether a read-only model prediction (eval layer) is available for this track. */
  hasPrediction: boolean;
}

/** One transcribed note from a stem's MIDI. Times in seconds. */
export interface StemNote {
  onset: number;
  offset: number;
  pitch: number;
  velocity: number;
  /** GM program (0-indexed) YourMT3+ assigned this note to; absent for transcribers
   *  without instrument labels (basic-pitch, drums). */
  program?: number;
}

/** Per-instrument grouping of a transcription's notes (YourMT3+ GM programs). */
export interface StemInstrument {
  /** General MIDI program (0-indexed). */
  program: number;
  /** GM instrument name, e.g. 'Acoustic Grand Piano'. */
  name: string;
  /** Notes in this transcription with this program. */
  nNotes: number;
}

/** Per-stem MIDI transcription. `stemType` is one of 'drums' | 'bass' | 'other' | 'vocals'. */
export interface StemTranscription {
  stemType: string;
  gmProgram: number;
  isDrums: boolean;
  notes: StemNote[];
  method: string;
  /** Per-GM-program summary of `notes` (YourMT3+ only; absent = no instrument labels). */
  instruments?: StemInstrument[];
}

/** Read-only per-stem stem-separation + MIDI transcription result for a track. */
export interface StemsResult {
  stems: { stemType: string; audioPath: string }[];
  transcriptions: StemTranscription[];
  /** Per-stem + 'combined' MIDI file paths (server-side). */
  midiPaths: Record<string, string>;
  method: string;
  durationSec: number | null;
}

/** Section-count slider metadata for a track with cached structure activations. */
export interface ResegmentInfo {
  /** Section count the import-time boundary threshold produced. */
  initialCount: number;
  /** Section count with every candidate boundary enabled (slider max). */
  maxCount: number;
}

export interface TrackListItem {
  id: string;
  title: string;
  genre: string;
  bpm: number;
  edited: boolean;
  /** Eval accuracy (0–1) of the model prediction vs ground truth; null when no prediction. */
  score: number | null;
}
