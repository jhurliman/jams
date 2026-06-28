# jams annotator

A TypeScript web app for visualizing and editing the jams beat / structure annotations —
waveform with beats and functional segments overlaid, zoom, edit, save, and playback.

![overview](waveform with colored segment bands, beat grid, and a transport bar)

## Run

```sh
cd webapp
npm install
npm run dev     # starts the API (:8787) + Vite (:5173); open http://localhost:5173
```

By default it reads the Raveform data tree at `../eval/data/raveform` (audio, beat CSVs,
`segments.json`). Override with `RAVEFORM_DIR=/path npm run dev:server`. Edited annotations are
saved to `<data>/annotations/<track_id>.json` (the source files are never overwritten).

## What it does

- **Track list** — all tracks with audio, filterable by genre / search; an `edited` badge marks
  tracks with a saved annotation.
- **Waveform** — decoded client-side (min/max peaks); rendered on a canvas with its own
  zoom/scroll so beats and segments share one exact coordinate space.
- **Beats** — downbeats (bar position 1) drawn full-height/light, off-beats short/dim.
- **Segments** — translucent colored bands per functional label, with editable boundaries.
- **Playback** — play/pause, click-to-seek, auto-scroll to follow the playhead.

## Editing

| action | how |
|---|---|
| select a segment | click its band |
| relabel | the Label dropdown in the inspector (11-class EDM vocab) |
| move a boundary | drag it on the waveform |
| split a segment | "Split at playhead" |
| add / delete a beat | double-click empty space / double-click a beat (zoom in first) |
| toggle a beat's bar position | alt-click a beat |
| zoom · pan | ⌘/Ctrl + wheel · wheel |
| undo / redo · save | ⌘Z / ⌘⇧Z · ⌘S |

Keyboard: space = play/pause, ←/→ (shift = ×5) nudge, Delete = remove selection.

## Stack

Vite + React 18 + strict TypeScript · Zustand (state + undo history) · Hono (API, range-aware
audio streaming) · custom canvas renderer (no waveform library, for full control over the
overlays and edit handles). Shared annotation types in `shared/types.ts` are the contract between
client and server and mirror `src/jams/data/structure_worker.py:_RAVEFORM_LABELS`.

`npm run typecheck` · `npm run lint` · `npm run build`.
