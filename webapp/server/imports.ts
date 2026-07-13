/** Drag-and-drop track import: run an uploaded file through the jams analysis API and
 *  register it as an annotator track (audio + beats + segments) under `import.<slug>`.
 *
 *  Imported tracks live outside the Raveform ground-truth dataset: their audio goes to
 *  AUDIO_DIR like any track, but their metadata + beat grid are appended to
 *  `imports.json` (IMPORTS_JSON) rather than the dataset's segments.json.
 */
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { extname, resolve } from 'node:path';

import type { Beat, Segment } from '../shared/types.ts';
import { AUDIO_DIR, IMPORTS_JSON } from './paths.ts';

/** Where the jams analysis API lives; scripts/dev.sh starts it on :8000. */
const JAMS_API_URL = process.env.JAMS_API_URL ?? 'http://localhost:8000';

/** Mirrors SUPPORTED_FORMATS in src/jams/analysis/audio.py. */
const IMPORT_EXTS = new Set(['.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a', '.aac']);

/** Shape of one imports.json row — RawTrack (segments.json) plus the beat grid, which
 *  dataset tracks keep in per-track CSVs but imports carry inline. */
export interface ImportedTrack {
  key: string;
  title: string;
  genre: string;
  average_bpm: number;
  duration: number;
  sections: { name: string; start: number; end: number }[];
  beats: Beat[];
}

interface JamsStructure {
  bpm: number | null;
  beats: number[];
  downbeats: number[];
  segments: { start: number; end: number; label: string }[];
}
interface JamsResponse {
  duration_sec: number | null;
  key: { key: string } | null;
  tempo: { bpm: number } | null;
  structure: JamsStructure | null;
}

export class ImportError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export function loadImports(): ImportedTrack[] {
  if (!existsSync(IMPORTS_JSON)) return [];
  return JSON.parse(readFileSync(IMPORTS_JSON, 'utf8')) as ImportedTrack[];
}

const slugify = (name: string): string =>
  name
    .toLowerCase()
    .replace(/\.[^.]+$/, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48) || 'track';

/** Assign 1-based bar positions to beat times given the downbeat times.
 *  Beats between downbeats count up from 1 at each downbeat; leading beats before the
 *  first downbeat count backwards from it assuming the dominant bar length. */
function toBeats(times: number[], downbeats: number[]): Beat[] {
  const sorted = [...times].sort((a, b) => a - b);
  const isDown = sorted.map((t) => downbeats.some((d) => Math.abs(d - t) < 0.03));
  const downIdx = isDown.flatMap((d, i) => (d ? [i] : []));
  // Dominant beats-per-bar from downbeat spacing (falls back to 4/4).
  const gaps = downIdx.slice(1).map((v, i) => v - downIdx[i]!);
  const barLen = gaps.length
    ? (gaps.sort((a, b) => a - b)[Math.floor(gaps.length / 2)] ?? 4)
    : 4;
  const first = downIdx[0] ?? 0;
  return sorted.map((time, i) => {
    const lastDown = downIdx.filter((d) => d <= i).at(-1);
    const offset = lastDown !== undefined ? i - lastDown : (((i - first) % barLen) + barLen) % barLen;
    return { time, bar: (offset % barLen) + 1 };
  });
}

/** One progress event from the analyze→import pipeline. Stages run CONCURRENTLY on
 *  the jams side (key ∥ tempo→structure), so consumers get independent per-stage
 *  running/done transitions rather than a strict sequence. */
export interface ImportProgress {
  running: string[];
  done: string[];
}

interface JamsJobStatus {
  status: 'running' | 'done' | 'error';
  stages_running: string[];
  stages_done: string[];
  result?: JamsResponse;
  error?: string;
  error_stage?: string | null;
}

const JOB_POLL_MS = 500;

const notReachable = () =>
  new ImportError(
    503,
    `jams analysis server is not reachable at ${JAMS_API_URL} — ` +
      `start the full stack with scripts/dev.sh (or just the API with \`uv run jams\`).`,
  );

/** Start an async jams analysis job and poll it to completion, forwarding stage
 *  transitions to `onProgress`. */
async function analyzeWithProgress(
  fileName: string,
  bytes: Uint8Array,
  onProgress?: (p: ImportProgress) => void,
): Promise<JamsResponse> {
  const form = new FormData();
  form.append('file', new Blob([bytes]), fileName);
  form.append('key', 'true');
  form.append('tempo', 'true');
  form.append('structure', 'true');
  form.append('async', 'true');

  let res: Response;
  try {
    res = await fetch(`${JAMS_API_URL}/v1/analyze`, { method: 'POST', body: form });
  } catch {
    throw notReachable();
  }
  if (res.status !== 202) {
    const detail = await res
      .json()
      .then((b) => (b as { detail?: string }).detail)
      .catch(() => null);
    throw new ImportError(res.status, `analysis failed (${res.status}): ${detail ?? res.statusText}`);
  }
  const { job_id } = (await res.json()) as { job_id: string };

  let last = '';
  for (;;) {
    await new Promise((r) => setTimeout(r, JOB_POLL_MS));
    let poll: Response;
    try {
      poll = await fetch(`${JAMS_API_URL}/v1/jobs/${job_id}`);
    } catch {
      throw notReachable();
    }
    if (!poll.ok) throw new ImportError(502, `analysis job lookup failed (${poll.status})`);
    const job = (await poll.json()) as JamsJobStatus;
    const sig = JSON.stringify([job.stages_running, job.stages_done]);
    if (sig !== last) {
      last = sig;
      onProgress?.({ running: job.stages_running, done: job.stages_done });
    }
    if (job.status === 'error') {
      const where = job.error_stage ? ` during ${job.error_stage}` : '';
      throw new ImportError(502, `analysis failed${where}: ${job.error ?? 'unknown error'}`);
    }
    if (job.status === 'done') {
      if (!job.result) throw new ImportError(502, 'analysis job finished without a result');
      return job.result;
    }
  }
}

/** Run one uploaded file through jams and register it. Returns the new track id.
 *  `onProgress` (optional) receives concurrent stage transitions plus the final
 *  'importing' step; omitting it keeps the original blocking behavior. */
export async function importTrack(
  fileName: string,
  bytes: Uint8Array,
  onProgress?: (p: ImportProgress) => void,
): Promise<string> {
  const ext = extname(fileName).toLowerCase();
  if (!IMPORT_EXTS.has(ext)) {
    throw new ImportError(422, `Unsupported audio format '${ext || fileName}'`);
  }

  const analysis = await analyzeWithProgress(fileName, bytes, onProgress);
  onProgress?.({ running: ['importing'], done: [] });
  const structure = analysis.structure;
  if (!structure || structure.beats.length === 0) {
    throw new ImportError(502, 'analysis returned no beat grid — is the structure backend configured?');
  }

  // Unique id under the import namespace.
  const existing = new Set(loadImports().map((t) => t.key));
  const base = `import.${slugify(fileName)}`;
  let id = base;
  for (let n = 2; existing.has(id); n++) id = `${base}-${n}`;

  const beats = toBeats(structure.beats, structure.downbeats);
  const duration = analysis.duration_sec ?? beats.at(-1)?.time ?? 0;
  const segments: Segment[] = structure.segments.map((s) => ({
    start: s.start,
    end: s.end,
    label: s.label as Segment['label'],
  }));
  const keyName = analysis.key?.key;

  await mkdir(AUDIO_DIR, { recursive: true });
  await writeFile(resolve(AUDIO_DIR, `${id}${ext}`), bytes);

  const entry: ImportedTrack = {
    key: id,
    title: keyName
      ? `${fileName.replace(/\.[^.]+$/, '')} [${keyName}]`
      : fileName.replace(/\.[^.]+$/, ''),
    genre: 'import',
    average_bpm: analysis.tempo?.bpm ?? structure.bpm ?? 0,
    duration,
    sections: segments.map((s) => ({ name: s.label, start: s.start, end: s.end })),
    beats,
  };
  const all = [...loadImports(), entry];
  await writeFile(IMPORTS_JSON, JSON.stringify(all, null, 2));
  return id;
}
