import { useCallback, useEffect, useRef, useState } from 'react';

import { api } from '../api.ts';
import { STRUCTURE_ANALYSIS_REALTIME_FACTOR } from '../config.ts';
import { importStemsEnabled } from '../importPrefs.ts';
import { useEditor } from '../store.ts';

/** Stages shown in the checklist, in display order. Analysis stages run CONCURRENTLY
 *  on the jams side (key ∥ tempo→structure; stems last) — each flips running/done
 *  independently. The stems row only shows when transcription is enabled. */
const STAGES = ['upload', 'key', 'tempo', 'structure', 'stems', 'importing'] as const;
type Stage = (typeof STAGES)[number];
type StageState = 'pending' | 'running' | 'done';

/** Share of the overall bar each stage owns; structure dominates wall time when stems
 *  are off, stems (separation + MIDI transcription) dominate when on. */
const STAGE_WEIGHT: Record<Stage, number> = {
  upload: 0.05,
  key: 0.1,
  tempo: 0.1,
  structure: 0.7,
  stems: 0,
  importing: 0.05,
};
const STAGE_WEIGHT_STEMS: Record<Stage, number> = {
  upload: 0.05,
  key: 0.05,
  tempo: 0.05,
  structure: 0.3,
  stems: 0.5,
  importing: 0.05,
};

const STAGE_LABEL: Record<Stage, string> = {
  upload: 'Upload',
  key: 'Key (CNN)',
  tempo: 'Tempo',
  structure: 'Beats & structure',
  stems: 'Stems & MIDI',
  importing: 'Importing track',
};

type Phase =
  | { kind: 'idle' }
  | { kind: 'hover' }
  | { kind: 'analyzing'; name: string }
  | { kind: 'error'; message: string };

const AUDIO_EXT = /\.(wav|mp3|flac|aiff|ogg|m4a|aac)$/i;

/** Best-effort local audio duration (drives the structure-stage bar estimate). */
function fileDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const el = new Audio();
    const done = (v: number | null) => {
      URL.revokeObjectURL(url);
      resolve(v);
    };
    el.onloadedmetadata = () => done(Number.isFinite(el.duration) ? el.duration : null);
    el.onerror = () => done(null);
    setTimeout(() => done(null), 3000);
    el.src = url;
  });
}

const initialStages = (): Record<Stage, StageState> => ({
  upload: 'running',
  key: 'pending',
  tempo: 'pending',
  structure: 'pending',
  stems: 'pending',
  importing: 'pending',
});

/** Full-window drag-and-drop import: drop an audio file anywhere to run it through the
 *  jams analysis backend and open it as a new track. Renders nothing while idle. */
export function ImportDropZone() {
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' });
  const [stages, setStages] = useState<Record<Stage, StageState>>(initialStages);
  const [elapsed, setElapsed] = useState(0);
  const [withStems, setWithStems] = useState(true);
  // dragenter/dragleave fire for every child element; track depth to know when the
  // pointer actually left the window.
  const depth = useRef(0);
  const es = useRef<EventSource | null>(null);
  const timing = useRef({ startedAt: 0, structureStartedAt: 0, durationSec: null as number | null });

  useEffect(() => () => es.current?.close(), []);

  // 4 Hz tick drives the elapsed readout + intra-structure bar animation.
  useEffect(() => {
    if (phase.kind !== 'analyzing') return;
    const t = setInterval(() => setElapsed((Date.now() - timing.current.startedAt) / 1000), 250);
    return () => clearInterval(t);
  }, [phase.kind]);

  const progress = useCallback((): number => {
    const weights = withStems ? STAGE_WEIGHT_STEMS : STAGE_WEIGHT;
    let total = 0;
    for (const s of STAGES) {
      if (stages[s] === 'done') total += weights[s];
      else if (stages[s] === 'running' && s === 'structure') {
        const { structureStartedAt, durationSec } = timing.current;
        if (structureStartedAt && durationSec) {
          const expected = durationSec * STRUCTURE_ANALYSIS_REALTIME_FACTOR;
          const frac = Math.min((Date.now() - structureStartedAt) / 1000 / expected, 0.95);
          total += weights[s] * frac;
        }
      } else if (stages[s] === 'running') {
        total += weights[s] * 0.5;
      }
    }
    return Math.min(total, 0.99);
  }, [stages, withStems]);

  const doImport = useCallback(async (file: File) => {
    if (!AUDIO_EXT.test(file.name)) {
      setPhase({ kind: 'error', message: `'${file.name}' is not a supported audio file` });
      return;
    }
    const stems = importStemsEnabled();
    setWithStems(stems);
    timing.current = { startedAt: Date.now(), structureStartedAt: 0, durationSec: null };
    setStages(initialStages());
    setElapsed(0);
    setPhase({ kind: 'analyzing', name: file.name });
    void fileDuration(file).then((d) => {
      timing.current.durationSec = d;
    });
    try {
      const { importId } = await api.importStart(file, { stems });
      setStages((s) => ({ ...s, upload: 'done' }));
      const stream = new EventSource(api.importProgressUrl(importId));
      es.current = stream;
      stream.onmessage = (msg) => {
        const ev = JSON.parse(msg.data as string) as
          | { type: 'progress'; running: string[]; done: string[] }
          | { type: 'done'; id: string }
          | { type: 'error'; message: string };
        if (ev.type === 'progress') {
          setStages((s) => {
            const next = { ...s, upload: 'done' as StageState };
            for (const st of ev.running) {
              if (st in next) next[st as Stage] = 'running';
              if (st === 'structure' && !timing.current.structureStartedAt) {
                timing.current.structureStartedAt = Date.now();
              }
              // once analysis stages hand off to importing, they're all complete
              if (st === 'importing') {
                next.key = 'done';
                next.tempo = 'done';
                next.structure = 'done';
                next.stems = 'done';
              }
            }
            for (const st of ev.done) if (st in next) next[st as Stage] = 'done';
            return next;
          });
        } else if (ev.type === 'done') {
          stream.close();
          es.current = null;
          const ed = useEditor.getState();
          ed.refreshTracks();
          void ed.loadTrack(ev.id).then(() => setPhase({ kind: 'idle' }));
        } else {
          stream.close();
          es.current = null;
          setPhase({ kind: 'error', message: ev.message });
        }
      };
      stream.onerror = () => {
        if (es.current === stream) {
          stream.close();
          es.current = null;
          setPhase({ kind: 'error', message: 'lost connection to the import progress stream' });
        }
      };
    } catch (err) {
      setPhase({ kind: 'error', message: err instanceof Error ? err.message : String(err) });
    }
  }, []);

  useEffect(() => {
    const hasFiles = (e: DragEvent) => [...(e.dataTransfer?.types ?? [])].includes('Files');
    const onDragEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth.current += 1;
      setPhase((p) => (p.kind === 'idle' || p.kind === 'error' ? { kind: 'hover' } : p));
    };
    const onDragOver = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault();
    };
    const onDragLeave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth.current = Math.max(0, depth.current - 1);
      if (depth.current === 0) setPhase((p) => (p.kind === 'hover' ? { kind: 'idle' } : p));
    };
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth.current = 0;
      const file = e.dataTransfer?.files[0];
      if (file) void doImport(file);
      else setPhase({ kind: 'idle' });
    };
    window.addEventListener('dragenter', onDragEnter);
    window.addEventListener('dragover', onDragOver);
    window.addEventListener('dragleave', onDragLeave);
    window.addEventListener('drop', onDrop);
    return () => {
      window.removeEventListener('dragenter', onDragEnter);
      window.removeEventListener('dragover', onDragOver);
      window.removeEventListener('dragleave', onDragLeave);
      window.removeEventListener('drop', onDrop);
    };
  }, [doImport]);

  if (phase.kind === 'idle') return null;
  return (
    <div className={`import-overlay ${phase.kind}`}>
      {phase.kind === 'hover' && <div className="import-card">Drop to analyze &amp; import</div>}
      {phase.kind === 'analyzing' && (
        <div className="import-card">
          <div>
            Analyzing <strong>{phase.name}</strong>
          </div>
          <div className="import-bar">
            <div className="import-bar-fill" style={{ width: `${progress() * 100}%` }} />
          </div>
          <ul className="import-stages">
            {STAGES.filter((s) => s !== 'stems' || withStems).map((s) => (
              <li key={s} className={stages[s]}>
                <span className="mark">
                  {stages[s] === 'done' ? '✓' : stages[s] === 'running' ? <span className="spinner sm" /> : '·'}
                </span>
                {STAGE_LABEL[s]}
              </li>
            ))}
          </ul>
          <div className="dim">
            {Math.floor(elapsed)}s elapsed —{' '}
            {stages.stems === 'running'
              ? 'stem separation & MIDI transcription can take a few minutes'
              : 'key & tempo run alongside structure'}
          </div>
        </div>
      )}
      {phase.kind === 'error' && (
        <div className="import-card error" onClick={() => setPhase({ kind: 'idle' })}>
          <div>Import failed</div>
          <div className="dim">{phase.message}</div>
          <div className="dim">(click to dismiss)</div>
        </div>
      )}
    </div>
  );
}
