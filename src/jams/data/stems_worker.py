#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#   "demucs>=4.0",
#   "basic-pitch[onnx]>=0.4",
#   "soundfile>=0.12",
#   "numpy>=1.23,<2",
#   "librosa>=0.10",
#   "pyyaml>=6.0",
# ]
# ///
"""Stem separation + pitched-stem transcription worker for jams.

Self-contained uv script: ``uv run --script stems_worker.py ...`` resolves and caches its own
environment. It runs in a *separate* interpreter from jams (jams is pinned to Python 3.14;
demucs/basic-pitch have no 3.14 wheels). jams invokes it as a subprocess and never imports it.
Python is capped <3.12 because basic-pitch hard-depends on tensorflow on Linux, whose 2.15
wheels stop at cp311 (inference still runs via ONNX; TF is just an install-time dependency).

Scope: **separation + PITCHED transcription only.** Drums are transcribed by the sibling
``drum_worker.py`` (isolated because its Magenta/TF1 stack can't co-resolve with demucs +
basic-pitch), and MIDI assembly + beat quantization happen in the orchestrator
(``jams.analysis.stems`` + ``jams.analysis.gm``). This keeps this env modern and conflict-free.

Pipeline:
1. Separate the mix into 4 stems (drums/bass/other/vocals). Two backends, chosen by the
   requested model name (device auto-select cuda -> mps -> cpu on both):
   - ``scnet*`` (default ``scnet_xl_ihf``): the vendored **SCNet XL IHF** (see ``scnet/``),
     ZFTurbo's MUSDB18 checkpoint, downloaded to ~/.cache/jams/scnet on first use. Won our
     Slakh-test A/B: SI-SDR drums 14.3 dB (htdemucs 11.6), bass note-F 0.596 -> 0.645.
   - ``htdemucs*``: Demucs via the stable 4.0.x APIs (kept for speed / comparison).
2. Transcribe the pitched stems (bass/other/vocals) with basic-pitch: bass/vocals get a
   monophonic post-filter; ``other`` stays polyphonic. The drums stem wav is written but NOT
   transcribed here (the orchestrator hands it to drum_worker).

Modes:
  single-shot:  stems_worker.py --audio FILE [--out-dir DIR]  -> prints one JSON object
  serve (JSONL): stems_worker.py --serve
     request:  {"audio": "mix.wav", "out_dir": "..."}                 # separate + pitched
           or: {"stems": {"bass": "b.wav", ...}, "out_dir": "..."}    # oracle: transcribe given
     response: {"ok": true, "result": {...}} | {"ok": false, "error": "..."}

Result schema:
  {"stems": [{"stem_type": "drums", "audio_path": "..."}, ...],
   "transcriptions": [{"stem_type": "bass", "gm_program": 33, "is_drums": false,
                       "notes": [{"onset": s, "offset": s, "pitch": midi, "velocity": v}, ...],
                       "method": "basic-pitch"}, ...],       # pitched stems only
   "duration_sec": 123.4}

Heavy imports live inside functions so the module is import-safe in jams' env.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

STEM_ORDER = ("drums", "bass", "other", "vocals")
PITCHED_STEMS = ("bass", "other", "vocals")

# General MIDI programs per pitched stem (kept in sync with jams.analysis.gm.GM_PROGRAM).
GM_PROGRAM = {"bass": 33, "other": 0, "vocals": 85}
MONOPHONIC_STEMS = frozenset({"bass", "vocals"})
FREQ_RANGE = {"bass": (30.0, 400.0), "vocals": (65.0, 2100.0), "other": (None, None)}
# Bass is written an octave above where it sounds (MIDI/notation convention); basic-pitch
# detects the sounding pitch. +12 aligns our bass MIDI with the written convention — validated
# on Slakh GT: note-F 0.04 -> 0.80 across all tracks, no regressions.
BASS_OCTAVE_SHIFT = 12
# basic-pitch (onset, frame) thresholds per stem. The dense polyphonic "other" stem does
# better with a stricter onset gate: (0.6, 0.25) scored 0.468 vs the default (0.5, 0.3)'s
# 0.445 note-F in a sweep on babyslakh ground-truth stems. Bass/vocals keep the defaults
# (monophonic post-filter already absorbs spurious notes).
ONSET_FRAME = {"other": (0.6, 0.25)}
_DEFAULT_ONSET_FRAME = (0.5, 0.3)


# --- Device selection -------------------------------------------------------


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- Separation (Demucs) ----------------------------------------------------
# Uses the stable demucs 4.0.x APIs (pretrained.get_model + apply.apply_model): demucs.api
# only exists in the unreleased git tree, and demucs.audio.AudioFile shells out to ffmpeg —
# librosa loads the audio instead, so the env stays self-contained.

_demucs_model = None
_demucs_name = None


def _get_demucs(model: str):
    global _demucs_model, _demucs_name
    if _demucs_model is not None and _demucs_name == model:
        return _demucs_model
    from demucs.pretrained import get_model

    print(f"[stems] loading demucs '{model}'", file=sys.stderr, flush=True)
    _demucs_model = get_model(model)
    _demucs_model.eval()
    _demucs_name = model
    return _demucs_model


def _separate_demucs(audio: str, out_dir: Path, model: str) -> dict[str, str]:
    """Split ``audio`` into 4 stems with Demucs, write wavs, return {stem: path}."""
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model

    dm = _get_demucs(model)
    device = _select_device()
    y, _sr = librosa.load(audio, sr=dm.samplerate, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])  # demucs expects stereo
    wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    # Same input normalisation as the demucs CLI (undone on the way out).
    ref = wav.mean(0)
    mean, std = ref.mean().item(), max(ref.std().item(), 1e-8)
    wav = (wav - mean) / std
    with torch.no_grad():
        sources = apply_model(dm, wav[None], device=device, split=True, overlap=0.25)[0]
    sources = sources * std + mean  # (n_sources, channels, samples)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    by_name = dict(zip(dm.sources, sources, strict=True))
    for name in STEM_ORDER:
        if name not in by_name:
            continue
        dest = out_dir / f"{name}.wav"
        sf.write(dest, by_name[name].cpu().numpy().T, dm.samplerate)
        paths[name] = str(dest)
    return paths


# --- Separation (SCNet XL IHF, vendored) -------------------------------------
# Vendored from ZFTurbo/Music-Source-Separation-Training (MIT) under ./scnet/. The XL IHF
# MUSDB18 checkpoint won our Slakh-test separation A/B on SI-SDR and pitched note-F.

_SCNET_BASE = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download"
_SCNET_FILES = {
    "config": (f"{_SCNET_BASE}/v1.0.15/config_musdb18_scnet_xl_more_wide_v5.yaml", 1_000),
    "weights": (f"{_SCNET_BASE}/v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt", 200_000_000),
}


def _is_scnet(model: str) -> bool:
    return model.lower().startswith("scnet")


class _AttrDict(dict):
    """Minimal recursive attribute-access dict for the vendored demix config."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc
        return _AttrDict(v) if isinstance(v, dict) else v


def _scnet_cache_dir() -> Path:
    import os

    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "jams" / "scnet"


def _download_scnet_file(kind: str) -> Path:
    """Fetch config/weights on first use; verify size so a truncated download can't ship."""
    import urllib.request

    url, min_bytes = _SCNET_FILES[kind]
    dest = _scnet_cache_dir() / url.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[stems] downloading SCNet {kind} -> {dest}", file=sys.stderr, flush=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - pinned release URL
    if not dest.exists() or dest.stat().st_size < min_bytes:
        size = dest.stat().st_size if dest.exists() else 0
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"SCNet {kind} download truncated ({size} bytes < {min_bytes}); retry."
        )
    return dest


_scnet_model = None
_scnet_cfg = None
_scnet_device = None


def _set_scnet_device(device) -> None:
    global _scnet_device
    _scnet_device = device


def _get_scnet():
    """Load (once) the vendored SCNet + XL IHF checkpoint; probe the device with a
    tiny forward pass and fall back cuda -> mps -> cpu on op gaps (explicit log —
    output is identical across devices, only speed differs)."""
    global _scnet_model, _scnet_cfg, _scnet_device
    if _scnet_model is not None:
        return _scnet_model, _scnet_cfg, _scnet_device

    import torch
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scnet import SCNet

    cfg_path = _download_scnet_file("config")
    ckpt_path = _download_scnet_file("weights")
    cfg = _AttrDict(yaml.load(cfg_path.read_text(), Loader=yaml.FullLoader))
    model = SCNet(**cfg["model"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state" in state:
        state = state["state"]
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    for dev in (_select_device(), "cpu"):
        try:
            device = torch.device(dev)
            model.to(device)
            with torch.inference_mode():
                model(torch.zeros(1, cfg["model"]["audio_channels"], 44100, device=device))
            print(f"[stems] SCNet ready on {dev}", file=sys.stderr, flush=True)
            _scnet_model, _scnet_cfg, _scnet_device = model, cfg, device
            return model, cfg, device
        except Exception as exc:  # noqa: BLE001 - op-gap probe; fall through to cpu
            if dev == "cpu":
                raise
            print(f"[stems] SCNet probe failed on {dev} ({exc}); using cpu",
                  file=sys.stderr, flush=True)
    raise RuntimeError("unreachable")


def _separate_scnet(audio: str, out_dir: Path) -> dict[str, str]:
    """Split ``audio`` into 4 stems with the vendored SCNet XL IHF."""
    import librosa
    import numpy as np
    import soundfile as sf
    from scnet.demix import demix  # sys.path set up by _get_scnet

    model, cfg, device = _get_scnet()
    sr = cfg["audio"]["sample_rate"]
    y, _ = librosa.load(audio, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    if device.type != "cuda":
        # The 11 s x batch-4 chunks OOM 32 GB unified memory on MPS; batch 1 fits and is
        # numerically identical (chunk windowing is per-chunk).
        cfg["inference"]["batch_size"] = 1
    try:
        sources = demix(cfg, model, y, device)  # {instrument: (channels, samples)}
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or device.type == "cpu":
            raise
        # Same output on any device — an explicit device downgrade, never a quality change.
        print(f"[stems] SCNet {device.type} OOM; retrying on cpu", file=sys.stderr, flush=True)
        import torch
        if device.type == "mps":
            torch.mps.empty_cache()
        model.to("cpu")
        _set_scnet_device(torch.device("cpu"))
        sources = demix(cfg, model, y, torch.device("cpu"))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name in STEM_ORDER:
        if name not in sources:
            continue
        dest = out_dir / f"{name}.wav"
        sf.write(dest, sources[name].T, sr)
        paths[name] = str(dest)
    return paths


def separate_stems(audio: str, out_dir: Path, model: str) -> dict[str, str]:
    """Split ``audio`` into 4 stems, write wavs into ``out_dir``, return {stem: path}."""
    if _is_scnet(model):
        return _separate_scnet(audio, out_dir)
    return _separate_demucs(audio, out_dir, model)


# --- Pitched transcription (basic-pitch) ------------------------------------

_bp_model = None


def _get_bp_model():
    global _bp_model
    if _bp_model is None:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model

        _bp_model = Model(ICASSP_2022_MODEL_PATH)
    return _bp_model


def _monophonic_filter(notes: list[dict]) -> list[dict]:
    """Collapse overlapping notes to a single voice, keeping the loudest at each moment."""
    accepted: list[dict] = []
    for n in sorted(notes, key=lambda x: (-x["velocity"], x["onset"])):
        if any(n["onset"] < a["offset"] and a["onset"] < n["offset"] for a in accepted):
            continue
        accepted.append(n)
    accepted.sort(key=lambda x: x["onset"])
    return accepted


def transcribe_pitched(wav: str, stem_type: str) -> list[dict]:
    """Transcribe a pitched stem to notes with basic-pitch."""
    from basic_pitch.inference import predict

    fmin, fmax = FREQ_RANGE.get(stem_type, (None, None))
    onset_t, frame_t = ONSET_FRAME.get(stem_type, _DEFAULT_ONSET_FRAME)
    _model_output, _midi, note_events = predict(
        wav,
        _get_bp_model(),
        onset_threshold=onset_t,
        frame_threshold=frame_t,
        minimum_note_length=90.0 if stem_type in MONOPHONIC_STEMS else 58.0,
        minimum_frequency=fmin,
        maximum_frequency=fmax,
        multiple_pitch_bends=False,
    )
    # note_events: list of (start_s, end_s, pitch_midi, amplitude[0-1], [pitch_bends])
    notes = [
        {
            "onset": float(ev[0]),
            "offset": float(ev[1]),
            "pitch": int(ev[2]),
            "velocity": max(1, min(127, int(round(ev[3] * 127)))),
        }
        for ev in note_events
    ]
    if stem_type in MONOPHONIC_STEMS:
        notes = _monophonic_filter(notes)
    if stem_type == "bass":
        notes = [{**n, "pitch": min(127, n["pitch"] + BASS_OCTAVE_SHIFT)} for n in notes]
    return notes


# --- Orchestration ----------------------------------------------------------


def analyze(req: dict) -> dict:
    """Separate (or accept oracle stems) and transcribe the pitched stems. See docstring."""
    out_dir = Path(req.get("out_dir") or _default_out_dir(req))
    out_dir.mkdir(parents=True, exist_ok=True)

    if req.get("stems"):  # oracle: caller supplies ground-truth stems; skip separation
        stem_paths: dict[str, str] = {k: v for k, v in req["stems"].items() if v}
    else:
        stem_paths = separate_stems(req["audio"], out_dir, req.get("model", "htdemucs"))

    transcriptions: list[dict] = []
    for stem_type in PITCHED_STEMS:
        wav = stem_paths.get(stem_type)
        if not wav:
            continue
        transcriptions.append(
            {
                "stem_type": stem_type,
                "gm_program": GM_PROGRAM.get(stem_type, 0),
                "is_drums": False,
                "notes": transcribe_pitched(wav, stem_type),
                "method": "basic-pitch",
            }
        )

    return {
        "stems": [{"stem_type": s, "audio_path": p} for s, p in stem_paths.items()],
        "transcriptions": transcriptions,
        "duration_sec": _duration(stem_paths),
    }


def _default_out_dir(req: dict) -> str:
    import tempfile

    key = req.get("audio") or next(iter(req.get("stems", {}).values()), "stems")
    return str(Path(tempfile.gettempdir()) / "jams_stems" / Path(key).stem)


def _duration(stem_paths: dict[str, str]) -> float | None:
    if not stem_paths:
        return None
    try:
        import soundfile as sf

        info = sf.info(next(iter(stem_paths.values())))
        return round(info.frames / info.samplerate, 3)
    except Exception:
        return None


# --- Entry points -----------------------------------------------------------


def _serve() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            # Keep the protocol stdout clean: model-load warnings (basic-pitch/coremltools/
            # resampy) print to stdout and would corrupt the JSONL — send them to stderr.
            with contextlib.redirect_stdout(sys.stderr):
                res = analyze(json.loads(line))
            out = {"ok": True, "result": res}
        except Exception as exc:  # noqa: BLE001 - report any failure to the caller
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true", help="JSONL stdin/stdout server mode")
    ap.add_argument("--audio", help="Mix file to separate + transcribe (single-shot)")
    ap.add_argument("--out-dir")
    ap.add_argument("--model", default="htdemucs")
    args = ap.parse_args()

    if args.serve:
        _serve()
        return
    if not args.audio:
        ap.error("provide --audio FILE or --serve")
    print(json.dumps(analyze({"audio": args.audio, "out_dir": args.out_dir, "model": args.model}),
                     indent=2))


if __name__ == "__main__":
    main()
