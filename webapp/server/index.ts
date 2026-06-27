import { createReadStream, existsSync, statSync } from 'node:fs';
import { Readable } from 'node:stream';
import type { ReadableStream as NodeWebReadableStream } from 'node:stream/web';

import { serve } from '@hono/node-server';
import { Hono } from 'hono';
import { cors } from 'hono/cors';

import type { Annotation } from '../shared/types.ts';
import { listTracks, loadAnnotation, saveAnnotation, trackMeta } from './annotations.ts';
import { audioPath } from './paths.ts';

const app = new Hono();
app.use('/api/*', cors());

app.get('/api/tracks', (c) => c.json(listTracks()));

app.get('/api/tracks/:id', (c) => {
  const meta = trackMeta(c.req.param('id'));
  return meta ? c.json(meta) : c.notFound();
});

app.get('/api/tracks/:id/annotation', (c) => {
  const ann = loadAnnotation(c.req.param('id'));
  return ann ? c.json(ann) : c.notFound();
});

app.put('/api/tracks/:id/annotation', async (c) => {
  const id = c.req.param('id');
  if (!trackMeta(id)) return c.notFound();
  const body = (await c.req.json()) as Annotation;
  await saveAnnotation(id, body);
  return c.json({ ok: true, edited: true });
});

/** Range-aware audio streaming so the browser can seek without downloading the whole file. */
app.get('/audio/:id', (c) => {
  const path = audioPath(c.req.param('id'));
  if (!existsSync(path)) return c.notFound();
  const size = statSync(path).size;
  const range = c.req.header('range');

  const headers: Record<string, string> = {
    'Content-Type': 'audio/mp4',
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
