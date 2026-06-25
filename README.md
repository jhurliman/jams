# jams

On-demand **music-information-retrieval API** for DJ / electronic music. Point it at a
track and get its **key**, **tempo**, and (optionally) **song structure** — using the
SOTA-on-GiantSteps methods benchmarked in the companion eval harness.

| Analysis | Method | Accuracy (GiantSteps) |
|----------|--------|-----------------------|
| Key | Essentia `KeyExtractor` (EDM-tuned `edma` profile) | MIREX **0.759** / exact 0.688 |
| Tempo | Pretrained **TempoCNN** + genre-aware octave resolution | Acc1 **0.965** (corrected labels) |
| Structure | All-In-One via Replicate (beats / downbeats / segments) | — |

Both key and tempo fall back to librosa automatically if Essentia isn't installed.

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

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/analyze` | Multipart upload (`file`, `key`, `tempo`, `structure`, `genre`, `bpm_min`, `bpm_max`) |
| `POST` | `/v1/analyze/path` | JSON body with a server-side `path` + the same options |
| `GET`  | `/health` | Liveness + version |
| `GET`  | `/docs` | OpenAPI / Swagger UI |

## Configuration

Env vars (prefix `JAMS_`, or a local `.env`): `JAMS_HOST`, `JAMS_PORT`, `JAMS_LOG_LEVEL`,
`JAMS_MAX_UPLOAD_MB`, `JAMS_REPLICATE_API_TOKEN` (or `REPLICATE_API_TOKEN`) for the
optional structure endpoint (`pip install 'jams[structure]'`).

## Develop

```sh
uv sync --all-extras --dev
uv run pytest          # tempo-resolution tests are pure; API tests use real analysis
uv run ruff check src tests
uv run mypy src
```

## Layout

```
src/jams/
  analysis/   key.py · tempo.py · structure.py · audio.py   (the MIR core)
  api/        app.py · routes.py                            (FastAPI)
  models.py   pydantic schemas
  config.py   settings
  data/models/deepsquare-k16-3.pb                            (bundled TempoCNN)
```
