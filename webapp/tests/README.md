# Timeline browser regression

Run `npm ci` and `npx playwright install chromium`, then start the Vite development
server with `npm run dev:web -- --host 127.0.0.1`. In another terminal run
`npm run test:timeline`. Set `TIMELINE_URL` if the server uses another address.
No backend or music assets are needed: the test serves a generated ten-second WAV
and seeds synthetic stem notes in the editor. It checks vertical lane scrolling,
Shift+wheel pan, Ctrl+wheel zoom, scrollbar pixel geometry, full-height playhead,
paused seeking before first playback, and movement during playback.

Optional `TIMELINE_SCREENSHOT=/absolute/path.png` saves the final browser state.
