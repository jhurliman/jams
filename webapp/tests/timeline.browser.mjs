import { chromium } from 'playwright';
import assert from 'node:assert/strict';
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.route('**/api/**', (r) => r.fulfill({ json: [] }));
  const samples = 22050 * 10;
  const wav = Buffer.alloc(44 + samples * 2);
  wav.write('RIFF');
  wav.writeUInt32LE(wav.length - 8, 4);
  wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(22050, 24);
  wav.writeUInt32LE(44100, 28);
  wav.writeUInt16LE(2, 32);
  wav.writeUInt16LE(16, 34);
  wav.write('data', 36);
  wav.writeUInt32LE(samples * 2, 40);
  for (let i = 0; i < samples; i++)
    wav.writeInt16LE(Math.round(5000 * Math.sin((i * 2 * Math.PI * 220) / 22050)), 44 + 2 * i);
  await page.route('**/audio/**', (r) => {
    const range = r.request().headers()['range'];
    const start = range ? Number(range.match(/bytes=(\d+)/)[1]) : 0;
    return r.fulfill({
      status: range ? 206 : 200,
      contentType: 'audio/wav',
      headers: {
        'accept-ranges': 'bytes',
        'content-length': String(wav.length - start),
        ...(range ? { 'content-range': `bytes ${start}-${wav.length - 1}/${wav.length}` } : {}),
      },
      body: wav.subarray(start),
    });
  });
  await page.goto(process.env.TIMELINE_URL ?? 'http://127.0.0.1:5173');
  await page.evaluate(async () => {
    const { useEditor } = await import('/src/store.ts');
    useEditor.setState({
      trackId: 'fixture',
      meta: { id: 'fixture', title: 'Review fixture', durationSec: 120 },
      annotation: { beats: [], segments: [], schemaVersion: 1 },
      stems: {
        midiPaths: {},
        transcriptions: ['drums', 'bass', 'other', 'vocals'].map((stemType) => ({
          stemType,
          gmProgram: 0,
          isDrums: stemType === 'drums',
          method: 'fixture',
          notes: [{ onset: 2, offset: 3, pitch: 60, velocity: 100 }],
        })),
      },
      view: { pxPerSec: 80, scrollLeft: 0, viewportWidth: 700 },
    });
  });
  await page.locator('.stemlanes').waitFor();
  const lanes = page.locator('.stemlanes');
  await lanes.hover();
  await page.mouse.wheel(0, 150);
  await page.waitForTimeout(150);
  assert.ok(
    (await lanes.evaluate((el) => el.scrollTop)) > 0,
    'Vertical wheel reaches lower stem lanes',
  );
  await page.keyboard.down('Shift');
  await page.mouse.wheel(0, 100);
  await page.keyboard.up('Shift');
  await page.waitForTimeout(150);
  const scroll = await page.evaluate(
    async () => (await import('/src/store.ts')).useEditor.getState().view.scrollLeft,
  );
  assert.ok(scroll > 0, 'Shift+wheel pans timeline');
  const zoomBefore = await page.evaluate(
    async () => (await import('/src/store.ts')).useEditor.getState().view.pxPerSec,
  );
  await page.keyboard.down('Control');
  await page.mouse.wheel(0, -100);
  await page.keyboard.up('Control');
  await page.waitForTimeout(150);
  assert.ok(
    (await page.evaluate(
      async () => (await import('/src/store.ts')).useEditor.getState().view.pxPerSec,
    )) > zoomBefore,
    'Ctrl+wheel zooms over lanes',
  );
  // Force a permanent gutter to exercise scrollbar geometry on macOS too.
  await page.addStyleTag({ content: '.stemlanes { scrollbar-gutter: stable; }' });
  await page.evaluate(async () => {
    const s = (await import('/src/store.ts')).useEditor;
    s.getState().setView({ scrollLeft: 0, pxPerSec: 80 });
  });
  await page.waitForTimeout(150);
  for (const canvas of await page.locator('.stemlane canvas').all()) {
    const m = await canvas.evaluate((el) => ({
      width: el.width,
      css: el.getBoundingClientRect().width,
      dpr: devicePixelRatio,
    }));
    assert.ok(Math.abs(m.width - m.css * m.dpr) <= 1, 'Piano-roll pixels match CSS width');
  }
  const rect = await page.locator('.vizstack').boundingBox();
  const play = await page.locator('.playhead').boundingBox();
  assert.ok(Math.abs(rect.height - play.height) < 1, 'Playhead spans stack');
  assert.deepEqual(errors, []);
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(200);
  const tx = await page
    .locator('.playhead')
    .evaluate((el) => new DOMMatrix(getComputedStyle(el).transform).m41);
  assert.ok(Math.abs(tx - 80) < 2, `Paused seek positions playhead at one second: ${tx}`);
  await page.keyboard.press('Space');
  await page.waitForTimeout(300);
  const playingX = await page
    .locator('.playhead')
    .evaluate((el) => new DOMMatrix(getComputedStyle(el).transform).m41);
  assert.ok(playingX > tx, 'Playback moves playhead');
  await page.keyboard.press('Space');
  if (process.env.TIMELINE_SCREENSHOT)
    await page.screenshot({ path: process.env.TIMELINE_SCREENSHOT });
  console.log(
    'PASS: vertical scroll, timeline pan, lane zoom, scrollbar geometry, full-height playhead, paused seek and playback; no page errors',
  );
} finally {
  await browser.close();
}
