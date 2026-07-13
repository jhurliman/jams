import { existsSync, readFileSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';

import {
  type Annotation,
  type Beat,
  type Segment,
  SECTION_LABELS,
  type SectionLabel,
  type StemNote,
  type StemsResult,
  type StemTranscription,
  type TrackListItem,
} from '../shared/types.ts';
import { loadImports } from './imports.ts';
import {
  ANNOTATIONS_DIR,
  beatCsvPath,
  editedPath,
  hasAudio,
  hasPrediction,
  hasStems,
  predictionPath,
  SEGMENTS_JSON,
  stemsPath,
} from './paths.ts';

interface RawSection {
  name: string;
  start: number;
  end: number;
}
interface RawTrack {
  key: string;
  title: string;
  genre: string;
  average_bpm: number;
  duration: number;
  sections: RawSection[];
}

const LABEL_SET = new Set<string>(SECTION_LABELS);
const coerceLabel = (name: string): SectionLabel =>
  (LABEL_SET.has(name) ? name : 'drop') as SectionLabel;

let cache: Map<string, RawTrack> | null = null;

/** Load + cache segments.json keyed by track id. */
function rawTracks(): Map<string, RawTrack> {
  if (cache) return cache;
  const arr = JSON.parse(readFileSync(SEGMENTS_JSON, 'utf8')) as RawTrack[];
  cache = new Map(arr.map((t) => [t.key, t]));
  return cache;
}

/** Dataset tracks + drag-and-drop imports. Imports are re-read every call (the file is
 *  small and grows while the server runs); the dataset stays cached. */
function allTracks(): Map<string, RawTrack> {
  const imports = loadImports();
  if (imports.length === 0) return rawTracks();
  const merged = new Map(rawTracks());
  for (const t of imports) merged.set(t.key, t);
  return merged;
}

export function listTracks(): TrackListItem[] {
  return [...allTracks().values()]
    .filter((t) => hasAudio(t.key))
    .map((t) => ({
      id: t.key,
      title: t.title,
      genre: t.genre,
      bpm: Math.round(t.average_bpm),
      edited: existsSync(editedPath(t.key)),
      score: scoreTrack(t),
    }));
}

const scoreCache = new Map<string, number | null>();

/** Eval accuracy for a track: the fraction of the timeline where the model's predicted label
 *  matches the canonical ground-truth label. null when there's no prediction. Lower = larger
 *  error, so the UI can sort worst-first. */
function scoreTrack(t: RawTrack): number | null {
  if (scoreCache.has(t.key)) return scoreCache.get(t.key)!;
  let score: number | null = null;
  if (hasPrediction(t.key) && t.duration > 0) {
    const pred = loadPrediction(t.key);
    if (pred) {
      let correct = 0;
      for (const g of t.sections) {
        for (const p of pred.segments) {
          if (p.label === g.name) {
            const ov = Math.min(g.end, p.end) - Math.max(g.start, p.start);
            if (ov > 0) correct += ov;
          }
        }
      }
      score = Math.min(1, correct / t.duration);
    }
  }
  scoreCache.set(t.key, score);
  return score;
}

export function trackMeta(id: string) {
  const t = allTracks().get(id);
  if (!t) return null;
  return {
    id,
    title: t.title,
    genre: t.genre,
    bpm: Math.round(t.average_bpm),
    durationSec: t.duration,
    edited: existsSync(editedPath(id)),
    hasPrediction: hasPrediction(id),
  };
}

/** Read-only model prediction (eval layer), if one exists for this track. */
export function loadPrediction(id: string): Annotation | null {
  if (!hasPrediction(id)) return null;
  return JSON.parse(readFileSync(predictionPath(id), 'utf8')) as Annotation;
}

/** Shape of the on-disk (Python-dumped) StemsResult: snake_case. */
interface RawStemNote {
  onset: number;
  offset: number;
  pitch: number;
  velocity: number;
}
interface RawStemTranscription {
  stem_type: string;
  gm_program: number;
  is_drums: boolean;
  notes: RawStemNote[];
  method: string;
}
interface RawStemAudio {
  stem_type: string;
  audio_path: string;
}
interface RawStemsResult {
  stems: RawStemAudio[];
  transcriptions: RawStemTranscription[];
  midi_paths: Record<string, string>;
  method: string;
  duration_sec: number | null;
}

/** Read-only per-stem MIDI transcription result, if one exists. Maps snake_case -> camelCase so
 *  the client stays on the app's camelCase convention (like trackMeta vs the python model). */
export function loadStems(id: string): StemsResult | null {
  if (!hasStems(id)) return null;
  const raw = JSON.parse(readFileSync(stemsPath(id), 'utf8')) as RawStemsResult;
  const notes = (ns: RawStemNote[]): StemNote[] =>
    ns.map((n) => ({ onset: n.onset, offset: n.offset, pitch: n.pitch, velocity: n.velocity }));
  const transcriptions: StemTranscription[] = raw.transcriptions.map((t) => ({
    stemType: t.stem_type,
    gmProgram: t.gm_program,
    isDrums: t.is_drums,
    notes: notes(t.notes),
    method: t.method,
  }));
  return {
    stems: raw.stems.map((s) => ({ stemType: s.stem_type, audioPath: s.audio_path })),
    transcriptions,
    midiPaths: raw.midi_paths,
    method: raw.method,
    durationSec: raw.duration_sec,
  };
}

/** Beats from the source beat CSV (`time,downbeat,section`; `downbeat`=bar position, 1=downbeat). */
function beatsFromCsv(id: string): Beat[] {
  const path = beatCsvPath(id);
  if (!existsSync(path)) return [];
  const beats: Beat[] = [];
  const lines = readFileSync(path, 'utf8').split('\n');
  for (const line of lines) {
    const [time, bar] = line.split(',');
    if (!time || time === 'time') continue;
    const t = Number(time);
    if (Number.isFinite(t)) beats.push({ time: t, bar: Number(bar) || 1 });
  }
  return beats;
}

function segmentsFromRaw(t: RawTrack): Segment[] {
  return t.sections.map((s) => ({ start: s.start, end: s.end, label: coerceLabel(s.name) }));
}

/** Edited annotation if it exists, otherwise the source reference: beat CSV +
 *  segments.json for dataset tracks, the imports.json row for imported tracks. */
export function loadAnnotation(id: string): Annotation | null {
  if (existsSync(editedPath(id))) {
    return JSON.parse(readFileSync(editedPath(id), 'utf8')) as Annotation;
  }
  const imported = loadImports().find((t) => t.key === id);
  if (imported) {
    return { trackId: id, beats: imported.beats, segments: segmentsFromRaw(imported) };
  }
  const t = rawTracks().get(id);
  if (!t) return null;
  return { trackId: id, beats: beatsFromCsv(id), segments: segmentsFromRaw(t) };
}

export async function saveAnnotation(id: string, ann: Annotation): Promise<void> {
  await mkdir(ANNOTATIONS_DIR, { recursive: true });
  const sorted: Annotation = {
    trackId: id,
    beats: [...ann.beats].sort((a, b) => a.time - b.time),
    segments: [...ann.segments].sort((a, b) => a.start - b.start),
  };
  await writeFile(editedPath(id), JSON.stringify(sorted, null, 2));
}
