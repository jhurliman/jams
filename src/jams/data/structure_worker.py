#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["all-in-one-mps>=0.1", "numpy>=1.26"]
# ///
"""All-In-One (PyTorch/MPS) song-structure worker for jams.

Self-contained uv script: ``uv run --script structure_worker.py ...`` resolves and
caches its own environment from the inline metadata above. It runs in a *separate*
interpreter from jams itself (jams is pinned to Python 3.14 for essentia, while
All-In-One needs torch/natten-mps/demucs, which have no 3.14 wheels). jams invokes
it as a subprocess via ``uv run`` and never imports it. Two modes:

  single-shot:  structure_worker.py --audio FILE [--target-bpm F] [--model NAME]
                -> prints one JSON object to stdout

  serve (JSONL): structure_worker.py --serve
                reads one JSON request per line on stdin:
                  {"audio": "...", "target_bpm": 174.0|null, "model": "all-all"}
                writes one JSON response per line on stdout:
                  {"ok": true, "result": {...}} | {"ok": false, "error": "..."}
                The All-In-One model is loaded lazily and kept resident, so the
                ~15-20s cold start is paid once per worker, not per request.

``target_bpm`` is applied as a *post-hoc octave correction* of All-In-One's native
beat grid (see ``_octave_correct``): the native beats are high quality and usually at
the right octave, so we keep them untouched and only densify/thin the grid when
``target_bpm`` shows the native tempo is a clean half/double (the half-time-genre case,
e.g. D&B/dubstep read an octave low). An earlier approach re-ran the DBN beat tracker
with a tight ``min_bpm/max_bpm = target_bpm +/- 1`` window — that wrecked beat-F (0.99 ->
0.70) even at the correct tempo, so it was replaced.

Heavy imports live inside functions so the module is import-safe in jams' env.
"""
from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import os
import statistics
import subprocess
import sys
import tempfile


def _beat_index(timestamp: float, beats: list[float]) -> int:
    """Nearest beat number (1-indexed) for a segment boundary timestamp."""
    if not beats:
        return 0
    idx = bisect.bisect_left(beats, timestamp)
    if idx == 0:
        return 1
    if idx >= len(beats):
        return len(beats)
    if timestamp - beats[idx - 1] <= beats[idx] - timestamp:
        return idx
    return idx + 1


def _densify(times: list[float]) -> list[float]:
    """Insert the midpoint between each consecutive pair (doubles the grid density)."""
    if len(times) < 2:
        return times
    out: list[float] = []
    for a, b in zip(times, times[1:], strict=False):
        out.append(a)
        out.append((a + b) / 2.0)
    out.append(times[-1])
    return out


def _octave_correct(beats: list[float], downbeats: list[float], native_bpm: float,
                    target_bpm: float) -> tuple[list[float], list[float], float]:
    """Scale the native beat grid to ``target_bpm``'s octave on a clean half/double.

    All-In-One's native beats are high quality and usually at the right octave already; a
    tight DBN re-track to fix the rare half/double-time error wrecks beat-F. Instead we keep
    the native grid and only adjust its *density* when the native tempo is ~2× (heard
    half-time → densify) or ~0.5× (heard double-time → thin) the target. When the octave
    already matches — the common case — the native beats pass through untouched, so there is
    no precision penalty.
    """
    if not native_bpm or native_bpm <= 0:
        return beats, downbeats, native_bpm
    ratio = target_bpm / native_bpm
    if 1.6 <= ratio <= 2.4:        # native heard half-time → double the grid
        return _densify(beats), _densify(downbeats), native_bpm * 2.0
    if 0.42 <= ratio <= 0.62:      # native heard double-time → thin the grid
        return beats[::2], downbeats[::2], native_bpm / 2.0
    return beats, downbeats, native_bpm  # octave already correct → untouched


# Pop+EDM-trained ("all") and EDM-only ("raveform") checkpoints live in the same
# HuggingFace repo (``taejunkim/allinone``) the loader already downloads from, but
# upstream's name map only registers the Harmonix folds. Filenames are the exact HF
# object names. ``all-all`` is the 8-fold Pop+EDM ensemble = the paper's best Raveform
# result (beat .991 / downbeat .965 / HR.5F .835).
_EXTRA_FILES = {
    "all-fold0": "all-fold0-40pa2vpn.pth",
    "all-fold1": "all-fold1-ixhnrlbv.pth",
    "all-fold2": "all-fold2-b9yx1jtt.pth",
    "all-fold3": "all-fold3-ri1y9ns9.pth",
    "all-fold4": "all-fold4-u6l5vhox.pth",
    "all-fold5": "all-fold5-m7dx7spr.pth",
    "all-fold6": "all-fold6-4j7nqihf.pth",
    "all-fold7": "all-fold7-f0qjbkoz.pth",
    "raveform-fold3": "raveform-fold3-mrkbf2f8.pth",
}
_EXTRA_ENSEMBLES = {"all-all": [f"all-fold{i}" for i in range(8)]}

# Raveform's 11-class functional vocabulary, in the trained classifier's index order
# (calibrated empirically against the dataset's labelled segments — see eval/). The
# upstream port hard-codes the 10-class Harmonix vocab, so we swap this in when running
# an EDM model. start/end are boundary sentinels, matching the Harmonix convention.
_RAVEFORM_LABELS = [
    "start", "end", "altintro", "altoutro", "intro", "outro",
    "breakdown", "buildup", "cooldown", "bridge", "drop",
]

# Boundary peak-strength threshold for functional segmentation. Upstream (both the mps port
# AND mir-aidj's original) hard-codes ``> 0.0``, which keeps every peak above the local mean
# and 2-3x over-segments (~22 segments vs ~11 true on Raveform) — boundary HR@0.5 collapses
# to ~0.56. 0.2 is the sweet spot from a 90-track genre-balanced sweep: HR@0.5 (trim) peaks at
# 0.741 (vs 0.720 at the model's configured 0.1) and the segment count (~11.7) best matches the
# ground truth (~10.8); per-genre tuning adds only ~0.006 so a single global value is used.
# ``None`` => fall back to the model's configured ``threshold_section`` (0.1 for EDM).
_BOUNDARY_THRESHOLD: float | None = 0.2

# Positional label prior (from Raveform ground truth across 1423 tracks: "intro"/"altintro" never
# start past ~27% of a track, "outro"/"altoutro" never before the back half). Before taking the
# function-label argmax we mask positionally-impossible classes, so the model falls back to its best
# *valid* label instead of e.g. an "intro" five minutes in. Thresholds carry a small safety margin.
_INTRO_MAX_FRAC = 0.12
_OUTRO_MIN_FRAC = 0.5


def _postprocess_functional(logits, cfg):
    """Drop-in for allin1's ``postprocess_functional_structure`` with a tunable boundary
    threshold (upstream hard-codes 0.0). Mirrors the original otherwise."""
    import numpy as np
    import torch
    from allin1.postprocessing import functional as _fnl
    from allin1.postprocessing.helpers import (
        event_frames_to_time,
        local_maxima,
        peak_picking,
    )
    from allin1.typings import Segment

    raw_prob_sections = torch.sigmoid(logits.logits_section[0])
    raw_prob_functions = torch.softmax(logits.logits_function[0], dim=0)
    prob_sections, _ = local_maxima(raw_prob_sections, filter_size=4 * cfg.min_hops_per_beat + 1)
    prob_sections = prob_sections.cpu().numpy()
    prob_functions = raw_prob_functions.cpu().numpy()

    candidates = peak_picking(prob_sections, window_past=12 * cfg.fps, window_future=12 * cfg.fps)
    thr = _BOUNDARY_THRESHOLD
    if thr is None:
        thr = float(getattr(cfg, "threshold_section", 0.1) or 0.1)
    boundary = candidates > thr

    duration = len(prob_sections) * cfg.hop_size / cfg.sample_rate
    times = event_frames_to_time(boundary, cfg)
    if len(times) == 0:
        times = np.array([0.0, duration], dtype=float)
    else:
        if times[0] != 0:
            times = np.insert(times, 0, 0)
        if times[-1] != duration:
            times = np.append(times, duration)
    pred_boundaries = np.stack([times[:-1], times[1:]]).T

    indices = np.flatnonzero(boundary)
    indices = indices[indices > 0]
    prob_segment_function = np.split(prob_functions, indices, axis=1)

    labels = _fnl.HARMONIX_LABELS  # swapped to the EDM vocab by _set_label_vocab when needed
    early = {labels.index(n) for n in ("intro", "altintro") if n in labels}
    late = {labels.index(n) for n in ("outro", "altoutro") if n in labels}

    segments = []
    for (s, e), probs in zip(pred_boundaries, prob_segment_function, strict=False):
        mean = probs.mean(axis=1).copy()
        frac = s / duration if duration else 0.0
        if frac > _INTRO_MAX_FRAC:
            for i in early:
                mean[i] = -1.0
        if frac < _OUTRO_MIN_FRAC:
            for i in late:
                mean[i] = -1.0
        segments.append(Segment(start=s, end=e, label=labels[int(mean.argmax())]))
    return segments


def _remap_v2_to_v1(state_dict: dict) -> dict:
    """Rewrite an ``all-fold*`` (v2) state dict into the v1 ``AllInOne`` layout.

    The v2 checkpoints share the v1 trunk verbatim (encoder/embeddings/beat/downbeat/
    section heads, identical keys + shapes) but add a ``dataset_classifier`` and split
    ``function_classifier`` into per-dataset heads (``.harmonix`` 10-class, ``.raveform``
    11-class). For EDM inference we keep the raveform head, rename it to the flat v1 key,
    and drop the harmonix head + dataset_classifier — yielding exactly the 491 keys the
    installed port expects. (``raveform-fold3`` is already in v1 form, so this is a no-op.)
    """
    out = {}
    for key, value in state_dict.items():
        if key.startswith(("dataset_classifier", "function_classifier.harmonix")):
            continue
        if key.startswith("function_classifier.raveform."):
            key = key.replace("function_classifier.raveform.", "function_classifier.")
        out[key] = value
    return out


def _register_extra_models() -> None:
    """Register ``all-fold*`` / ``raveform-fold3`` / ``all-all`` and a v2->v1 remap loader."""
    from allin1.models import loaders

    for name, filename in _EXTRA_FILES.items():
        loaders.NAME_TO_FILE.setdefault(name, filename)
    for name, folds in _EXTRA_ENSEMBLES.items():
        loaders.ENSEMBLE_MODELS.setdefault(name, list(folds))

    if getattr(loaders, "_jams_patched", False):
        return
    loaders._jams_patched = True

    import torch
    from allin1.models.allinone import AllInOne
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    _orig = loaders.load_pretrained_model

    def _load_edm(model_name, cache_dir, device):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        path = hf_hub_download(
            repo_id="taejunkim/allinone", filename=_EXTRA_FILES[model_name], cache_dir=cache_dir)
        checkpoint = torch.load(path, map_location=device)
        config = OmegaConf.create(checkpoint["config"])
        OmegaConf.set_struct(config, False)
        config.data.num_labels = len(_RAVEFORM_LABELS)  # v2 configs nest this per-dataset
        model = AllInOne(config).to(device)
        model.load_state_dict(_remap_v2_to_v1(checkpoint["state_dict"]))
        model.eval()
        return model

    # Upstream reloads weights on every analyze() call; cache loaded models by name so
    # the resident worker keeps them in memory (big speedup when scoring many tracks or
    # cycling fold models, and for production library scans). Inference is read-only, so
    # sharing a model instance across requests is safe.
    cache: dict = {}

    def _patched(model_name=None, cache_dir=None, device=None):
        key = (model_name, str(device))
        if key in cache:
            return cache[key]
        if model_name in loaders.ENSEMBLE_MODELS:
            model = loaders.load_ensemble_model(model_name, cache_dir, device)
        elif model_name in _EXTRA_FILES:
            model = _load_edm(model_name, cache_dir, device)
        else:
            model = _orig(model_name, cache_dir, device)
        cache[key] = model
        return model

    # ``allin1/analyze.py`` did ``from .models import load_pretrained_model`` at import time, so
    # it holds its own module-level binding — patching only ``loaders`` routes ensembles (resolved
    # inside loaders) correctly but leaves single ``all-foldN`` loads on the un-remapped path. We
    # must patch the binding in the analyze module's namespace too. Reach the real module objects
    # via ``sys.modules`` — ``import allin1.analyze`` would bind the re-exported *function*
    # (``allin1.__init__`` does ``from .analyze import analyze``), not the submodule.
    import allin1.analyze  # noqa: F401 - ensure the submodule is imported
    import allin1.helpers  # noqa: F401
    import allin1.models  # noqa: F401

    loaders.load_pretrained_model = _patched
    sys.modules["allin1.models"].load_pretrained_model = _patched
    sys.modules["allin1.analyze"].load_pretrained_model = _patched

    # Tunable boundary threshold (upstream hard-codes 0.0 -> over-segmentation). ``run_inference``
    # in allin1.helpers did ``from .postprocessing import postprocess_functional_structure``, so
    # patch that binding (same re-export caveat as the loader above).
    sys.modules["allin1.helpers"].postprocess_functional_structure = _postprocess_functional


def _set_label_vocab(model: str) -> None:
    """Point allin1's segment-label decoder at the right vocabulary for ``model``.

    The port decodes function classes through a module-level ``HARMONIX_LABELS`` list;
    EDM models (``all-*``/``raveform-*``) use the 11-class Raveform vocab instead.
    """
    import allin1.config
    import allin1.postprocessing.functional as fnl

    is_edm = model.startswith(("all-", "raveform-"))
    fnl.HARMONIX_LABELS = _RAVEFORM_LABELS if is_edm else allin1.config.HARMONIX_LABELS


# All-In-One's DiNAT inference breaks past ~2**16 frames (655 s at 100 fps): it emits beats for
# only the first ~40 s and collapses the rest into one segment. Process longer tracks in
# overlapping windows (each safely under the cap) and stitch. Threshold is well below the cap.
_LEN_CAP_SEC = 600.0
_CHUNK_SEC = 480.0
_CHUNK_OVERLAP_SEC = 30.0


def _audio_duration(audio: str) -> float:
    out = subprocess.run(  # noqa: S603,S607
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", audio],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def _slice_audio(audio: str, start: float, length: float) -> str:
    """Write a `length`-second slice of `audio` from `start` to a temp WAV; return its path."""
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(  # noqa: S603,S607
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
         "-i", audio, "-ac", "2", "-ar", "44100", tmp],
        check=True)
    return tmp


def _run_single(
    audio: str, model: str,
) -> tuple[list[float], list[float], list[tuple], float | None]:
    """Run All-In-One on a single (short-enough) audio path -> beats, downbeats, segments, bpm."""
    import allin1

    with contextlib.redirect_stdout(sys.stderr):  # keep our stdout protocol clean
        r = allin1.analyze(
            paths=audio, model=model, include_activations=False, keep_byproducts=False)
    beats = [float(b) for b in r.beats]
    downbeats = [float(d) for d in r.downbeats]
    segments = [(float(s.start), float(s.end), s.label) for s in r.segments]
    bpm = float(r.bpm) if r.bpm is not None else None
    return beats, downbeats, segments, bpm


def _run_chunked(audio: str, model: str, duration: float):
    """Analyse a long track in overlapping windows and stitch the results.

    Each window stays under the model's frame cap. Outputs are assigned to whichever window
    *owns* each timestamp — the seam sits at the middle of each overlap — so beats/downbeats
    dedupe cleanly and segments are clipped to their window, then same-label runs merge across
    the seam.
    """
    step = _CHUNK_SEC - _CHUNK_OVERLAP_SEC
    starts: list[float] = []
    s = 0.0
    while s < duration:
        starts.append(s)
        s += step

    chunks = []  # (start, end, beats, downbeats, segments)
    for cs in starts:
        length = min(_CHUNK_SEC, duration - cs)
        if length < 1.0:
            continue
        tmp = _slice_audio(audio, cs, length)
        try:
            b, db, segs, _ = _run_single(tmp, model)
        finally:
            os.remove(tmp)
        chunks.append((cs, cs + length, b, db, segs))

    bounds = [0.0]
    for i in range(len(chunks) - 1):
        bounds.append((chunks[i + 1][0] + chunks[i][1]) / 2.0)  # middle of the overlap
    bounds.append(duration)

    beats: list[float] = []
    downbeats: list[float] = []
    raw_segs: list[tuple] = []
    for i, (cs, _ce, b, db, segs) in enumerate(chunks):
        lo, hi = bounds[i], bounds[i + 1]
        beats += [cs + t for t in b if lo <= cs + t < hi]
        downbeats += [cs + t for t in db if lo <= cs + t < hi]
        for st, en, lab in segs:
            a, z = max(cs + st, lo), min(cs + en, hi)
            if z - a > 0.05:
                raw_segs.append((a, z, lab))
    beats.sort()
    downbeats.sort()
    raw_segs.sort(key=lambda x: x[0])

    merged: list[tuple] = []
    for st, en, lab in raw_segs:
        if merged and merged[-1][2] == lab and st - merged[-1][1] < 1.0:
            merged[-1] = (merged[-1][0], en, lab)
        else:
            merged.append((st, en, lab))

    diffs = [b - a for a, b in zip(beats, beats[1:], strict=False) if b > a]
    bpm = round(60.0 / statistics.median(diffs), 2) if diffs else None
    return beats, downbeats, merged, bpm


def analyze(audio: str, target_bpm: float | None, model: str) -> dict:
    _register_extra_models()
    _set_label_vocab(model)

    try:
        duration = _audio_duration(audio)
    except Exception:  # noqa: BLE001 - ffprobe unavailable -> just run the model directly
        duration = 0.0

    if duration > _LEN_CAP_SEC:
        beats, downbeats, raw_segs, bpm = _run_chunked(audio, model, duration)
        method = f"allin1-mps-local:{model}+chunked"
    else:
        beats, downbeats, raw_segs, bpm = _run_single(audio, model)
        method = f"allin1-mps-local:{model}"

    # Keep the native beats (high quality); only octave-correct when target_bpm says the
    # native tempo is a clean half/double. Octave-correct tracks pass through untouched.
    if target_bpm is not None and bpm:
        new_beats, new_downbeats, new_bpm = _octave_correct(
            beats, downbeats, bpm, float(target_bpm))
        if new_bpm != bpm:
            beats, downbeats, bpm = new_beats, new_downbeats, new_bpm
            method += f"+octave{bpm:g}"

    segments = [
        {
            "start": st, "end": en, "label": lab,
            "start_beat": _beat_index(st, beats),
            "end_beat": _beat_index(en, beats),
        }
        for st, en, lab in raw_segs
    ]
    return {
        "bpm": bpm, "beats": beats, "downbeats": downbeats,
        "segments": segments, "method": method,
    }


def _serve() -> None:
    """Read JSONL requests on stdin, write JSONL responses on stdout (model resident)."""
    # Touch the import once up front so failures surface immediately.
    import allin1  # noqa: F401

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = analyze(req["audio"], req.get("target_bpm"), req.get("model", "all-all"))
            out = {"ok": True, "result": res}
        except Exception as exc:  # noqa: BLE001 - report any failure to the caller
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="All-In-One structure worker for jams")
    ap.add_argument("--serve", action="store_true", help="persistent JSONL stdin/stdout mode")
    ap.add_argument("--audio", help="audio file (single-shot mode)")
    ap.add_argument("--target-bpm", type=float, default=None)
    ap.add_argument("--model", default="all-all")
    args = ap.parse_args()

    if args.serve:
        _serve()
        return
    if not args.audio:
        ap.error("either --serve or --audio is required")
    print(json.dumps(analyze(args.audio, args.target_bpm, args.model)))


if __name__ == "__main__":
    main()
