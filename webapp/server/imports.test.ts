/** Import-flow tests focused on stems persistence: the jams API is mocked with a tiny
 *  HTTP server, RAVEFORM_DIR points at a temp data tree, and no real models run.
 *  Run: npm test (tsx --test server/imports.test.ts). */
import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { after, before, test } from 'node:test';

// Both modules read env at import time — configure before the dynamic imports below.
const dataDir = mkdtempSync(join(tmpdir(), 'annotator-data-'));
writeFileSync(join(dataDir, 'segments.json'), '[]');
process.env.RAVEFORM_DIR = dataDir;

/** What the mock jams API serves; tests overwrite `result` per scenario. */
const mock = {
  result: {} as Record<string, unknown>,
  lastAnalyzeBody: '',
};

let server: Server;
let jamsDir: string; // stand-in for jams' temp stems work dir

before(async () => {
  server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => {
      if (req.method === 'POST' && req.url === '/v1/analyze') {
        mock.lastAnalyzeBody = Buffer.concat(chunks).toString('latin1');
        res.writeHead(202, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ job_id: 'j1' }));
      } else if (req.url === '/v1/jobs/j1') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(
          JSON.stringify({ status: 'done', stages_running: [], stages_done: [], result: mock.result }),
        );
      } else {
        res.writeHead(404).end();
      }
    });
  });
  await new Promise<void>((r) => server.listen(0, r));
  const addr = server.address();
  if (addr === null || typeof addr === 'string') throw new Error('no port');
  process.env.JAMS_API_URL = `http://localhost:${addr.port}`;

  jamsDir = mkdtempSync(join(tmpdir(), 'jams-stems-'));
});

after(() => {
  server.close();
  rmSync(dataDir, { recursive: true, force: true });
  rmSync(jamsDir, { recursive: true, force: true });
});

const structure = {
  bpm: 128,
  beats: [0.5, 1.0, 1.5, 2.0],
  downbeats: [0.5],
  segments: [{ start: 0, end: 4, label: 'intro' }],
  activations: null,
};
const base = { duration_sec: 4, key: { key: 'A minor' }, tempo: { bpm: 128 }, structure };

function fakeStemsResult() {
  mkdirSync(jamsDir, { recursive: true });
  const wav = (name: string) => {
    const p = join(jamsDir, name);
    writeFileSync(p, `fake-${name}`);
    return p;
  };
  return {
    stems: [
      { stem_type: 'drums', audio_path: wav('drums.wav') },
      { stem_type: 'bass', audio_path: wav('bass.wav') },
    ],
    transcriptions: [
      {
        stem_type: 'drums',
        gm_program: 0,
        is_drums: true,
        notes: [{ onset: 0.5, offset: 0.6, pitch: 36, velocity: 100 }],
        method: 'adtof',
      },
    ],
    midi_paths: { drums: wav('drums.mid'), combined: wav('combined.mid') },
    method: 'scnet_xl_ihf+yourmt3+adtof',
    duration_sec: 4,
  };
}

test('import persists stems: files copied, paths rewritten, JSON written', async () => {
  const { importTrack } = await import('./imports.ts');
  mock.result = { ...base, stems: fakeStemsResult() };

  const id = await importTrack('My Song.wav', new Uint8Array([1, 2, 3]));

  assert.match(mock.lastAnalyzeBody, /name="stems"/, 'analyze request must ask for stems');
  const stemsJson = resolve(dataDir, 'stems', `${id}.json`);
  assert.ok(existsSync(stemsJson), 'stems JSON written');
  const raw = JSON.parse(readFileSync(stemsJson, 'utf8')) as ReturnType<typeof fakeStemsResult>;

  for (const s of raw.stems) {
    assert.ok(s.audio_path.startsWith(resolve(dataDir, 'stems', id)), `copied: ${s.audio_path}`);
    assert.ok(existsSync(s.audio_path), 'stem wav exists');
  }
  for (const p of Object.values(raw.midi_paths)) {
    assert.ok(p.startsWith(resolve(dataDir, 'stems', id)), `copied: ${p}`);
    assert.equal(readFileSync(p, 'utf8'), `fake-${p.split('/').at(-1)}`, 'content preserved');
  }
  assert.equal(raw.transcriptions.length, 1);
  assert.equal(raw.transcriptions[0]!.notes[0]!.pitch, 36);
});

test('stems requested but absent from the result is a loud error', async () => {
  const { importTrack, ImportError } = await import('./imports.ts');
  mock.result = { ...base }; // no stems key
  await assert.rejects(
    importTrack('Other Song.wav', new Uint8Array([1])),
    (err: unknown) =>
      err instanceof ImportError && err.status === 502 && /stems were requested/.test(err.message),
  );
});

test('stems=false skips the request and writes no stems JSON', async () => {
  const { importTrack } = await import('./imports.ts');
  mock.result = { ...base }; // jams returns none, and that's fine
  const id = await importTrack('Quiet Song.wav', new Uint8Array([1]), undefined, { stems: false });
  assert.doesNotMatch(mock.lastAnalyzeBody, /name="stems"/, 'stems must not be requested');
  assert.ok(!existsSync(resolve(dataDir, 'stems', `${id}.json`)), 'no stems JSON');
});
