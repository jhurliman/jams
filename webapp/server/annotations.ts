import { existsSync, readFileSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';

import {
  type Annotation,
  type Beat,
  type Segment,
  SECTION_LABELS,
  type SectionLabel,
  type TrackListItem,
} from '../shared/types.ts';
import { ANNOTATIONS_DIR, beatCsvPath, editedPath, hasAudio, SEGMENTS_JSON } from './paths.ts';

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

export function listTracks(): TrackListItem[] {
  return [...rawTracks().values()]
    .filter((t) => hasAudio(t.key))
    .map((t) => ({
      id: t.key,
      title: t.title,
      genre: t.genre,
      bpm: Math.round(t.average_bpm),
      edited: existsSync(editedPath(t.key)),
    }));
}

export function trackMeta(id: string) {
  const t = rawTracks().get(id);
  if (!t) return null;
  return {
    id,
    title: t.title,
    genre: t.genre,
    bpm: Math.round(t.average_bpm),
    durationSec: t.duration,
    edited: existsSync(editedPath(id)),
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

/** Edited annotation if it exists, otherwise the source reference (beat CSV + segments.json). */
export function loadAnnotation(id: string): Annotation | null {
  if (existsSync(editedPath(id))) {
    return JSON.parse(readFileSync(editedPath(id), 'utf8')) as Annotation;
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
