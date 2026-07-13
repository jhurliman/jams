import { createReadStream, existsSync, statSync } from 'node:fs';
import { Readable } from 'node:stream';
import type { ReadableStream as NodeWebReadableStream } from 'node:stream/web';

import { randomUUID } from 'node:crypto';

import { serve } from '@hono/node-server';
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { streamSSE } from 'hono/streaming';

import type { Annotation } from '../shared/types.ts';
import {
  listTracks,
  loadAnnotation,
  loadPrediction,
  loadStems,
  saveAnnotation,
  trackMeta,
} from './annotations.ts';
import {
  ImportError,
  type ImportProgress,
  importTrack,
  resegmentInfo,
  resegmentTrack,
} from './imports.ts';
import { audioPath } from './paths.ts';

const app = new Hono();
app.use('/api/*', cors());

app.get('/api/tracks', (c) => c.json(listTracks()));

/** Drag-and-drop import: analyze an uploaded audio file via the jams API and register
 *  it as a new track. Slow by design (full analysis) — the client shows progress. */
app.post('/api/import', async (c) => {
  const body = await c.req.parseBody();
  const file = body.file;
  if (!(file instanceof File)) {
    return c.json({ error: "multipart form must include a 'file' field" }, 400);
  }
  try {
    const id = await importTrack(file.name, new Uint8Array(await file.arrayBuffer()));
    return c.json({ id });
  } catch (err) {
    if (err instanceof ImportError) return c.json({ error: err.message }, err.status as 400);
    return c.json({ error: err instanceof Error ? err.message : String(err) }, 500);
  }
});

/** Progress-reporting variant of /api/import: POST the file to /start (returns an
 *  import id immediately), then stream stage events from /progress/:id as SSE.
 *  Events: {type:'progress', running, done} | {type:'done', id} | {type:'error', ...}. */
interface PendingImport {
  events: ImportProgress[];
  done?: { id: string };
  error?: { status: number; message: string };
  finishedAt?: number;
}
const pendingImports = new Map<string, PendingImport>();
const PENDING_TTL_MS = 10 * 60 * 1000;

function purgePending(): void {
  const now = Date.now();
  for (const [id, p] of pendingImports) {
    if (p.finishedAt && now - p.finishedAt > PENDING_TTL_MS) pendingImports.delete(id);
  }
}

app.post('/api/import/start', async (c) => {
  purgePending();
  const body = await c.req.parseBody();
  const file = body.file;
  if (!(file instanceof File)) {
    return c.json({ error: "multipart form must include a 'file' field" }, 400);
  }
  const importId = randomUUID();
  const state: PendingImport = { events: [] };
  pendingImports.set(importId, state);
  const bytes = new Uint8Array(await file.arrayBuffer());
  void importTrack(file.name, bytes, (p) => state.events.push(p))
    .then((id) => {
      state.done = { id };
      state.finishedAt = Date.now();
    })
    .catch((err: unknown) => {
      state.error =
        err instanceof ImportError
          ? { status: err.status, message: err.message }
          : { status: 500, message: err instanceof Error ? err.message : String(err) };
      state.finishedAt = Date.now();
    });
  return c.json({ importId });
});

app.get('/api/import/progress/:importId', (c) => {
  const state = pendingImports.get(c.req.param('importId'));
  if (!state) return c.notFound();
  return streamSSE(c, async (stream) => {
    let sent = 0;
    for (;;) {
      while (sent < state.events.length) {
        const ev = state.events[sent++]!;
        await stream.writeSSE({ data: JSON.stringify({ type: 'progress', ...ev }) });
      }
      if (state.done) {
        await stream.writeSSE({ data: JSON.stringify({ type: 'done', ...state.done }) });
        return;
      }
      if (state.error) {
        await stream.writeSSE({ data: JSON.stringify({ type: 'error', ...state.error }) });
        return;
      }
      await new Promise((r) => setTimeout(r, 200));
    }
  });
});

app.get('/api/tracks/:id', (c) => {
  const meta = trackMeta(c.req.param('id'));
  return meta ? c.json(meta) : c.notFound();
});

app.get('/api/tracks/:id/annotation', (c) => {
  const ann = loadAnnotation(c.req.param('id'));
  return ann ? c.json(ann) : c.notFound();
});

app.get('/api/tracks/:id/prediction', (c) => {
  const pred = loadPrediction(c.req.param('id'));
  return pred ? c.json(pred) : c.body(null, 204);
});

app.get('/api/tracks/:id/stems', (c) => {
  const stems = loadStems(c.req.param('id'));
  return stems ? c.json(stems) : c.body(null, 204);
});

/** Section-count slider metadata (204 when the track has no cached activations). */
app.get('/api/tracks/:id/resegment', (c) => {
  const info = resegmentInfo(c.req.param('id'));
  return info ? c.json(info) : c.body(null, 204);
});

/** Rethreshold the track's cached structure activations to `count` sections. Proxies to
 *  jams' /v1/resegment (pure numpy — effectively instant); the client replaces its
 *  in-memory segments with the result, so nothing is written server-side here. */
app.post('/api/tracks/:id/resegment', async (c) => {
  const { count } = (await c.req.json()) as { count?: number };
  if (!Number.isInteger(count) || count! < 1) {
    return c.json({ error: 'count must be a positive integer' }, 400);
  }
  try {
    return c.json(await resegmentTrack(c.req.param('id'), count!));
  } catch (err) {
    if (err instanceof ImportError) return c.json({ error: err.message }, err.status as 400);
    return c.json({ error: err instanceof Error ? err.message : String(err) }, 500);
  }
});

/** Stream a per-stem (or 'combined') MIDI file resolved from the stems result's `midiPaths`. */
app.get('/api/tracks/:id/midi/:stem', (c) => {
  const stems = loadStems(c.req.param('id'));
  const path = stems?.midiPaths[c.req.param('stem')];
  if (!path || !existsSync(path)) return c.notFound();
  const stream = createReadStream(path);
  return new Response(Readable.toWeb(stream) as NodeWebReadableStream as ReadableStream, {
    headers: {
      'Content-Type': 'audio/midi',
      'Content-Length': String(statSync(path).size),
      'Content-Disposition': `attachment; filename="${c.req.param('id')}-${c.req.param('stem')}.mid"`,
    },
  });
});

app.put('/api/tracks/:id/annotation', async (c) => {
  const id = c.req.param('id');
  if (!trackMeta(id)) return c.notFound();
  const body = (await c.req.json()) as Annotation;
  await saveAnnotation(id, body);
  return c.json({ ok: true, edited: true });
});

const AUDIO_MIME: Record<string, string> = {
  '.m4a': 'audio/mp4',
  '.aac': 'audio/aac',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
  '.aiff': 'audio/aiff',
};

/** Range-aware audio streaming so the browser can seek without downloading the whole file. */
app.get('/audio/:id', (c) => {
  const path = audioPath(c.req.param('id'));
  if (!existsSync(path)) return c.notFound();
  const size = statSync(path).size;
  const range = c.req.header('range');

  const headers: Record<string, string> = {
    'Content-Type': AUDIO_MIME[path.slice(path.lastIndexOf('.'))] ?? 'audio/mp4',
    'Accept-Ranges': 'bytes',
  };

  if (range) {
    const match = /bytes=(\d*)-(\d*)/.exec(range);
    const start = match?.[1] ? Number(match[1]) : 0;
    const end = match?.[2] ? Number(match[2]) : size - 1;
    const stream = createReadStream(path, { start, end });
    return new Response(Readable.toWeb(stream) as NodeWebReadableStream as ReadableStream, {
      status: 206,
      headers: {
        ...headers,
        'Content-Range': `bytes ${start}-${end}/${size}`,
        'Content-Length': String(end - start + 1),
      },
    });
  }

  const stream = createReadStream(path);
  return new Response(Readable.toWeb(stream) as NodeWebReadableStream as ReadableStream, {
    headers: { ...headers, 'Content-Length': String(size) },
  });
});

const port = Number(process.env.PORT ?? 8787);
serve({ fetch: app.fetch, port }, (info) => {
  console.log(`jams annotator API on http://localhost:${info.port}`);
});
