# jams

On-demand **music-information-retrieval API** for DJ / electronic music. Point it at a
track and get its **key**, **tempo**, and (optionally) **song structure** — using the
SOTA-on-GiantSteps methods benchmarked in the companion eval harness.

| Analysis | Method | Accuracy (GiantSteps) |
|----------|--------|-----------------------|
| Key | Essentia `edma` tonic + a learned major/minor refinement | MIREX **0.801** / exact 0.743 |
| Tempo | Pretrained **TempoCNN** + genre-aware octave resolution | Acc1 **0.965** (corrected labels) |
| Structure | **All-In-One EDM ensemble on-device** (Apple-Silicon/MPS) | Raveform held-out CV reproduces SOTA (see `eval/`) |
| Stems → MIDI | **Demucs** 4-stem split + per-stem transcription (basic-pitch; ADTOF drums → General MIDI) | Slakh oracle: bass 0.80 / other 0.45 note-F; SDR 10.2 dB drums (see `eval/`) |

Both key and tempo fall back to librosa automatically if Essentia isn't installed. Key
mode (major/minor) is refined by a small chroma classifier — see *Key mode* below.

## Requirements

- **Python 3.14** — pinned in `.python-version`, so `uv` picks it automatically. This is
  required: on macOS arm64 the `essentia-tensorflow` wheel ships **only** for CPython 3.14
  (3.11/3.13 fail to resolve with a "no wheel for this platform" error). If you hit that,
  check `python --version` / `.python-version`.
- `uv` (https://docs.astral.sh/uv). First `uv sync` pulls `essentia-tensorflow` (~95 MB,
  native) and TensorFlow — give it a minute.
- The TempoCNN model is bundled (`src/jams/data/models/deepsquare-k16-3.pb`); no download.
- For the librosa *fallback* path to decode mp3s you need `ffmpeg` on PATH (Essentia decodes
  mp3 natively, so this only matters if Essentia is unavailable).

## Quickstart

```sh
uv sync                       # install (pulls essentia-tensorflow — heavy, native)
uv run jams                   # serve on http://0.0.0.0:8000  (Swagger at /docs)
```

Analyze an upload:

```sh
curl -s -F file=@track.wav -F genre="Drum & Bass" http://localhost:8000/v1/analyze | jq
```

Analyze a file already on the server (e.g. your local library):

```sh
curl -s http://localhost:8000/v1/analyze/path \
  -H 'content-type: application/json' \
  -d '{"path": "/Users/me/Music/track.wav", "genre": "Dubstep"}' | jq
```

Example response:

```json
{
  "filename": "track.wav",
  "duration_sec": 124.0,
  "key": {"key": "F minor", "tonic": "F", "mode": "minor", "confidence": 0.81, "method": "essentia-edma"},
  "tempo": {"bpm": 174.0, "bpm_raw": 87.0, "bpm_alt": 87.0, "octave_resolved": true, "method": "tempocnn-deepsquare"}
}
```

## Tempo octave resolution (the DJ-critical bit)

Tempo trackers get the BPM *value* right but can be an octave off (half/double-time) —
the error concentrates in **Drum & Bass** and **Dubstep**. Pass a `genre` (or explicit
`bpm_min`/`bpm_max`) and the result is folded into that genre's canonical octave. D&B and
jungle resolve to **full tempo (~174)**, not half-time. `bpm_alt` always returns the
other octave so a client can flip it. With no hint, the raw value is returned unchanged
(nothing is silently folded).

## Key mode (major vs minor)

`edma` nails the *tonic* but, like all template methods, over-calls **major** on minor
tracks — the diagnostic note is the **third** (minor 3rd vs major 3rd above the tonic),
which a full-template correlation dilutes. We keep edma's tonic and refine the *mode*
with a small logistic classifier over chroma cues (the third, 6th, 7th, and a
bass-register third), overriding edma only when confident. 5-fold CV: MIREX
**0.759→0.801**, exact **0.688→0.743**, with major-key recall preserved. The model ships
at `src/jams/data/mode_model.json`; retrain with `eval/train_mode_model.py`. Pass
`detect_key(path, refine_mode=False)` to skip it (saves a ~1-2 s chroma pass).

## Song structure (on-device)

Structure (beats / downbeats / **functional segments** — intro/buildup/drop/breakdown/…) comes
from **All-In-One** (Kim & Nam, WASPAA 2023). By default it runs the **EDM-trained `all-all`
8-fold ensemble locally on Apple Silicon** via PyTorch-MPS — no Replicate, no network, no
per-call cost. The EDM weights live on the same HuggingFace repo as the stock model and load via
a state-dict remap (no retraining). Because All-In-One needs torch/natten/demucs (which have no
Python 3.14 wheels and so can't share jams' env), the worker `src/jams/data/structure_worker.py`
is a **self-contained `uv` script** that bootstraps its own environment; jams launches it once
and keeps the models resident. **Requirement:** `uv` on PATH and an Apple-Silicon Mac. Structure
is opt-in per request (`structure=true`).

On Raveform's held-out 8-fold CV this reproduces the paper's SOTA (beat 0.978 / downbeat 0.964 /
boundary HR 0.755 / pairwise 0.825), and the production ensemble is more robust still — see
`eval/README.md`.

`target_bpm` (jams' octave-resolved tempo, fed automatically when you request `tempo` + `structure`)
is a *secondary* octave-correction safety net: it post-hoc rescales the beat grid only on a clean
half/double-time read. The EDM model already tracks D&B/dubstep at the right octave, so it's
usually a no-op — but harmless. (The earlier `±1 BPM` DBN-constraint approach was removed; it
crippled beat-F.)

Prefer the hosted model? Set `JAMS_STRUCTURE_BACKEND=replicate` (+ a Replicate token) to use
the original `jhurliman/allinone-targetbpm` endpoint instead.

## Stems → MIDI (on-device)

Opt-in per request (`stems=true`): split a track into 4 stems (**drums / bass / other /
vocals**) with **Demucs `htdemucs`**, then transcribe each to MIDI —

- **bass / vocals** → basic-pitch, monophonic post-filter (GM programs 34 / 85). Bass is
  shifted +12 to the written-MIDI convention (validated on Slakh: note-F 0.04 → 0.80).
- **other** → basic-pitch, polyphonic (GM piano)
- **drums** → **ADTOF Frame_RNN** (torch port of Zehren et al.'s crowdsourced-data CRNN;
  F 88.5 vs the original's 88.7 on MDBDrums++) → General MIDI percussion on channel 10
  (36 kick, 38 snare, 42 hats, 47 toms, 49 cymbals), quantized to jams' beat grid

Output is one `.mid` per stem plus a combined Type-1 multitrack `.mid`, and inline note arrays.
Like structure, the heavy models run in self-contained `uv` workers (no Python 3.14 wheels for
demucs/basic-pitch/torch), kept resident: `src/jams/data/stems_worker.py` (separation +
pitched) and `drum_worker.py` (drums, isolated so its git-sourced model dependency never
touches jams' own env). The orchestrator (`analysis/stems.py` + `analysis/gm.py`) merges them
and assembles the MIDI.

**Platform:** fully cross-platform — separation auto-selects cuda → mps → cpu, and both
transcribers are torch/ONNX, so the whole pipeline (drums included) runs on Apple-Silicon
Macs, Linux, and CI identically. Config: `JAMS_STEMS_MODEL` (`htdemucs`),
`JAMS_STEMS_QUANTIZE`, `JAMS_STEMS_OUT_DIR`, `JAMS_STEMS_UV`. See `eval/README.md` for the
transcription benchmark.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/analyze` | Multipart upload (`file`, `key`, `tempo`, `structure`, `genre`, `bpm_min`, `bpm_max`) |
| `POST` | `/v1/analyze/path` | JSON body with a server-side `path` + the same options |
| `GET`  | `/health` | Liveness + version |
| `GET`  | `/docs` | OpenAPI / Swagger UI |

Add **`?format=jams`** to either analyze endpoint to get the result as a
[JAMS](https://jams.readthedocs.io) document (the standard MIR annotation format the
Harmonix Set ships in) instead of the native schema: key → `key_mode`, tempo → `tempo`,
structure → `beat` + `segment_open`, each with per-observation `time`/`duration`/`confidence`
and `annotation_metadata` provenance (the producing `method` lands in `annotation_tools`).

```sh
curl -s 'http://localhost:8000/v1/analyze/path?format=jams' \
  -H 'content-type: application/json' \
  -d '{"path": "/Users/me/Music/track.wav", "structure": true, "genre": "Drum & Bass"}' | jq
```

## Configuration

Env vars (prefix `JAMS_`, or a local `.env`): `JAMS_HOST`, `JAMS_PORT`, `JAMS_LOG_LEVEL`,
`JAMS_MAX_UPLOAD_MB`. Structure backend: `JAMS_STRUCTURE_BACKEND` (`local` default | `replicate`),
`JAMS_STRUCTURE_MODEL` (`all-all` EDM ensemble default; `harmonix-all` for pop), `JAMS_STRUCTURE_UV` (path to `uv` if not on
PATH); the `replicate` backend needs `JAMS_REPLICATE_API_TOKEN` (or `REPLICATE_API_TOKEN`) and
`pip install 'jams[structure]'`.

## Develop

```sh
uv sync --all-extras --dev
uv run pytest          # tempo-resolution tests are pure; API tests use real analysis
uv run ruff check src tests
uv run mypy src
```

## Reproduce / push the accuracy

The `eval/` harness benchmarks the production functions against GiantSteps and is how the
numbers above were measured. Run in the project env with the `eval` extra:

```sh
uv run --extra eval eval/acquire_dataset.py   # download GiantSteps Key (~816 MB audio, one time)
uv run --extra eval eval/evaluate.py          # key MIREX + tempo Acc1/Acc2
uv run --extra eval eval/analyze_errors.py    # where the errors are, by genre/mode/octave
```

See `eval/README.md` for the method shoot-outs, the wrong-label story, and the curated
`tempo_corrections.csv`.

## Layout

```
src/jams/
  analysis/   key.py · tempo.py · structure.py · audio.py   (the MIR core)
  api/        app.py · routes.py                            (FastAPI)
  models.py   pydantic schemas
  config.py   settings
  data/models/deepsquare-k16-3.pb                            (bundled TempoCNN)
  data/structure_worker.py                                   (self-contained uv worker: All-In-One)
```
