#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["all-in-one-mps>=0.1", "numpy>=1.26", "numba>=0.61"]
# ///
"""All-In-One (PyTorch/MPS) song-structure worker for jams.

Self-contained uv script: ``uv run --script structure_worker.py ...`` resolves and
caches its own environment from the inline metadata above. It runs in a *separate*
interpreter from jams itself (All-In-One's pinned torch/natten-mps/demucs stack
conflicts with jams' own env). jams invokes
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
# to ~0.56. 0.2 is the sweet spot from a 90-track genre-balanced sweep on the boundary HR@0.5
# metric (HR peaks at 0.741; segment count ~11.7 best matches the ~10.8 ground truth) — but a
# fixed 0.2 leaves clearly *under-segmented* tracks stranded (e.g. a 7-min track stuck at 4
# segments), which tanks label-accuracy. ``None`` => the model's configured ``threshold_section``.
_BOUNDARY_THRESHOLD: float | None = 0.2

# Adaptive boundary threshold: start at the strict 0.2 (preserving the HR-tuned value for tracks
# that are already well segmented) and step DOWN the ladder only when a track is under-segmented,
# until the boundary count reaches a length-based target (~1 per 40 s) or the floor. From a
# 100-track capture-and-sweep (worst-60 + control-40), this lifts mean label-accuracy +3.7pt
# overall / +5.5pt on the worst tracks while moving the control set only +0.9pt (it lowers the
# threshold on just ~30% of well-segmented tracks). Set ``_BOUNDARY_ADAPTIVE=False`` to force the
# fixed ``_BOUNDARY_THRESHOLD`` (e.g. to reproduce the HR-tuned numbers). Overridable per-run via
# ``JAMS_BOUNDARY_ADAPTIVE=0`` so the eval harness can A/B adaptive vs fixed without a code edit.
_BOUNDARY_ADAPTIVE = os.environ.get("JAMS_BOUNDARY_ADAPTIVE", "1").lower() \
    not in ("0", "false", "no")
_BOUNDARY_LADDER = (0.20, 0.15, 0.12, 0.10, 0.08, 0.06)
_BOUNDARY_TARGET_SEC = 40.0


def _select_boundary_threshold(candidates, duration: float) -> float:
    """Pick the strictest ladder threshold whose boundary count meets the length-based target."""
    import numpy as np

    target = max(3.0, duration / _BOUNDARY_TARGET_SEC)
    thr = _BOUNDARY_LADDER[0]
    for thr in _BOUNDARY_LADDER:
        # +1: the implicit track-start boundary at t=0 isn't a candidate peak.
        if int(np.count_nonzero(candidates > thr)) + 1 >= target:
            break
    return thr

# Positional label prior (from Raveform ground truth across 1423 tracks: "intro"/"altintro" never
# start past ~27% of a track, "outro"/"altoutro" never before the back half). Before taking the
# function-label argmax we mask positionally-impossible classes, so the model falls back to its best
# *valid* label instead of e.g. an "intro" five minutes in. Thresholds carry a small safety margin.
_INTRO_MAX_FRAC = 0.12
_OUTRO_MIN_FRAC = 0.5

# Boundary-label correction (from the same 1423-track GT): a track's first segment is an
# intro-family label 99.9% of the time (intro 86.7% / altintro 13.2%; only 1 track opens on
# anything else) and its last non-marker segment is outro/altoutro/drop/cooldown ~99.9%. So a
# SHORT opening/closing segment carrying a different label (e.g. a "breakdown"/"buildup" opening)
# is almost certainly a label error, and snapping it to intro/outro recovers the match. The
# length guard is essential: long mislabelled boundary segments usually have a wrong *boundary*
# too, so relabelling them wholesale overcorrects (validated — without the guard the worst case
# was -0.38). At frac<0.30 AND <45 s the correction is zero-regression across all Raveform preds.
# 'start' (never in GT) is skipped; 'end' (a real trailing GT marker) is preserved.
_HEAD_LABELS = ("intro", "altintro")
_TAIL_OK_LABELS = ("outro", "altoutro", "drop", "cooldown")
_BOUNDARY_MAX_FRAC = 0.30
_BOUNDARY_MAX_SEC = 45.0

# A LONG opening labelled "breakdown" is a different error: GT never opens on breakdown, and when
# the model calls the opening "breakdown" it's a mislabelled "altintro" ~75% of the time (an
# atmospheric no-drums intro and a mid-track breakdown are acoustically alike). Unlike the short
# case above we can't snap it to plain "intro" — these are long, energy-light sections, so altintro
# is the right family. Validated on the 12 affected Raveform preds: 9 improve (+15..+44pt), 3
# regress (-9..-15pt, all under-segmented openings the adaptive threshold tends to split first).
_BREAKDOWN_OPENING_MIN_SEC = 45.0


def _fix_boundary_labels(segs: list[tuple], duration: float) -> list[tuple]:
    """Snap a clearly-mislabelled first/last segment to its intro/outro family (see notes above)."""
    if not segs:
        return segs
    dur = duration or max((s[1] for s in segs), default=0.0)
    if dur <= 0:
        return segs
    out = [list(s) for s in segs]
    lim = min(_BOUNDARY_MAX_FRAC * dur, _BOUNDARY_MAX_SEC)
    h = 0
    while h < len(out) and out[h][2] == "start":  # skip leading marker (never in GT)
        h += 1
    if h < len(out):
        head_len = out[h][1] - out[h][0]
        if out[h][2] == "breakdown" and head_len >= _BREAKDOWN_OPENING_MIN_SEC:
            out[h][2] = "altintro"   # long breakdown opening -> mislabelled altintro
        elif out[h][2] not in _HEAD_LABELS and head_len < lim:
            out[h][2] = "intro"      # short non-intro opening -> mislabelled intro
    t = len(out) - 1
    while t >= 0 and out[t][2] == "end":  # preserve trailing marker (real GT label)
        t -= 1
    if t > h and out[t][2] not in _TAIL_OK_LABELS and (out[t][1] - out[t][0]) < lim:
        out[t][2] = "outro"
    return [tuple(s) for s in out]


def _candidate_segments(cand_frames, cand_strengths, threshold, frame_rate,
                        label_probs, label_frame_rate, duration, labels):
    """Threshold sparse boundary candidates and label the spans between them (pure numpy).

    The one shared implementation behind both the model path (full-resolution activations,
    inside ``_postprocess_functional``) and ``resegment_from_activations`` (the pooled blob
    behind the annotator's section-count slider) — so a slider rethreshold reproduces the
    model path exactly. Boundary/label logic mirrors upstream's
    ``postprocess_functional_structure``: frame -> time is ``frame / fps`` (librosa's
    ``frames_to_time``), a 0/duration edge is added when missing, per-span labels are the
    argmax of the mean class probabilities with the positional prior applied.
    """
    import numpy as np

    cand_frames = np.asarray(cand_frames, dtype=np.int64)
    cand_strengths = np.asarray(cand_strengths, dtype=np.float64)
    times = cand_frames[cand_strengths > threshold] / frame_rate

    if times.size == 0:
        edges = np.array([0.0, duration])
    else:
        edges = times
        if edges[0] != 0:
            edges = np.insert(edges, 0, 0.0)
        if edges[-1] != duration:
            edges = np.append(edges, duration)
    spans = np.stack([edges[:-1], edges[1:]]).T

    probs = np.asarray(label_probs, dtype=np.float64)
    # A candidate at frame 0 shapes no span (mirrors upstream's ``indices[indices > 0]``).
    split_at = [int(round(t * label_frame_rate)) for t in times if t > 0]
    prob_per_span = np.split(probs, split_at, axis=1)

    early = {labels.index(n) for n in ("intro", "altintro") if n in labels}
    late = {labels.index(n) for n in ("outro", "altoutro") if n in labels}

    out = []
    for (s, e), span_probs in zip(spans, prob_per_span, strict=False):
        # else-branch: span narrower than one label frame (a boundary at the very end)
        mean = span_probs.mean(axis=1).copy() if span_probs.shape[1] else np.zeros(len(labels))
        frac = s / duration if duration else 0.0
        if frac > _INTRO_MAX_FRAC:
            for i in early:
                mean[i] = -1.0
        if frac < _OUTRO_MIN_FRAC:
            for i in late:
                mean[i] = -1.0
        out.append((float(s), float(e), labels[int(mean.argmax())]))
    return out


# Raw activations of the most recent inference, stashed by ``_postprocess_functional`` so
# ``analyze`` can emit them as a compact JSON blob when the request asks for activations
# (the annotator's section-count slider rethresholds that blob without re-running the
# model). The worker serves one request at a time (see ``_serve``), so a global is safe.
_LAST_ACTIVATIONS: dict | None = None

# Label activations are mean-pooled to this rate for the blob: ~5 fps keeps a 5-minute
# track around 130 KB of JSON (vs ~2.5 MB at the native 100 fps) and is far finer than any
# real section. Boundary candidates stay at native resolution — peak picking spaces them
# >= 12 s apart, so they are only a handful of (frame, strength) pairs.
_ACTIVATIONS_LABEL_FPS = 5.0


def _activations_blob(cap: dict) -> dict:
    """Shrink a raw activation capture into the JSON blob served to API clients."""
    import numpy as np

    keep = cap["cand_strengths"] > 0
    factor = max(1, int(round(cap["frame_rate"] / _ACTIVATIONS_LABEL_FPS)))
    probs = cap["prob_functions"]
    pad = (-probs.shape[1]) % factor
    if pad:
        probs = np.pad(probs, ((0, 0), (0, pad)), mode="edge")
    pooled = probs.reshape(probs.shape[0], -1, factor).mean(axis=2)
    return {
        "version": 1,
        "duration": round(float(cap["duration"]), 3),
        "frame_rate": float(cap["frame_rate"]),
        "candidates": [
            [int(f), round(float(s), 5)]
            for f, s in zip(cap["cand_frames"][keep], cap["cand_strengths"][keep], strict=True)
        ],
        "labels": list(cap["labels"]),
        "label_frame_rate": cap["frame_rate"] / factor,
        "label_probs": [[round(float(v), 4) for v in col] for col in pooled.T],
        "threshold": round(float(cap["threshold"]), 4),
    }


def _threshold_for_sections(strengths: list[float], target_sections: int) -> float:
    """Threshold that keeps exactly the ``target_sections - 1`` strongest boundaries
    (midpoint between neighboring strengths, so it is robust to blob rounding)."""
    n_bounds = max(0, int(target_sections) - 1)
    ordered = sorted(strengths, reverse=True)
    if n_bounds == 0:
        return ordered[0] + 1.0 if ordered else 1.0
    if n_bounds >= len(ordered):
        return ordered[-1] / 2 if ordered else 0.0
    return (ordered[n_bounds - 1] + ordered[n_bounds]) / 2


def resegment_from_activations(
    activations: dict,
    *,
    threshold: float | None = None,
    target_sections: int | None = None,
    beats: list[float] | None = None,
) -> dict:
    """Recompute structure segments from a cached activations blob — no model, no torch.

    Powers the annotator's section-count slider: the same thresholding + labelling +
    boundary-label correction as the import-time path (``_candidate_segments`` +
    ``_fix_boundary_labels``), in pure numpy, so jams itself can serve it instantly.
    Give ``threshold`` OR ``target_sections`` (which picks the threshold for you); with
    neither, the analysis-time threshold stored in the blob is reused.
    """
    import numpy as np

    if threshold is not None and target_sections is not None:
        raise ValueError("pass either 'threshold' or 'target_sections', not both")
    cands = activations.get("candidates") or []
    frames = [int(c[0]) for c in cands]
    strengths = [float(c[1]) for c in cands]
    duration = float(activations["duration"])
    labels = [str(x) for x in activations["labels"]]
    probs = np.asarray(activations["label_probs"], dtype=np.float64).T  # -> classes x frames
    if probs.ndim != 2 or probs.shape[1] == 0:
        raise ValueError("activations blob has no label_probs")

    if threshold is None:
        if target_sections is not None:
            interior = [s for f, s in zip(frames, strengths, strict=True) if f > 0]
            threshold = _threshold_for_sections(interior, target_sections)
        else:
            threshold = float(activations["threshold"])

    triples = _candidate_segments(
        frames, strengths, threshold, float(activations["frame_rate"]),
        probs, float(activations["label_frame_rate"]), duration, labels,
    )
    triples = _fix_boundary_labels(triples, duration)
    beats = beats or []
    segments = [
        {
            "start": st, "end": en, "label": lab,
            "start_beat": _beat_index(st, beats) if beats else None,
            "end_beat": _beat_index(en, beats) if beats else None,
        }
        for st, en, lab in triples
    ]
    return {"segments": segments, "threshold": float(threshold)}


def _postprocess_functional(logits, cfg):
    """Drop-in for allin1's ``postprocess_functional_structure`` with a tunable boundary
    threshold (upstream hard-codes 0.0). Mirrors the original otherwise."""
    import numpy as np
    import torch
    from allin1.postprocessing import functional as _fnl
    from allin1.postprocessing.helpers import local_maxima, peak_picking
    from allin1.typings import Segment

    raw_prob_sections = torch.sigmoid(logits.logits_section[0])
    raw_prob_functions = torch.softmax(logits.logits_function[0], dim=0)
    prob_sections, _ = local_maxima(raw_prob_sections, filter_size=4 * cfg.min_hops_per_beat + 1)
    prob_sections = prob_sections.cpu().numpy()
    prob_functions = raw_prob_functions.cpu().numpy()

    candidates = peak_picking(prob_sections, window_past=12 * cfg.fps, window_future=12 * cfg.fps)
    duration = len(prob_sections) * cfg.hop_size / cfg.sample_rate
    frame_rate = cfg.sample_rate / cfg.hop_size
    if _BOUNDARY_ADAPTIVE:
        thr = _select_boundary_threshold(candidates, duration)
    elif _BOUNDARY_THRESHOLD is not None:
        thr = _BOUNDARY_THRESHOLD
    else:
        thr = float(getattr(cfg, "threshold_section", 0.1) or 0.1)

    cand_frames = np.flatnonzero(candidates)
    cand_strengths = candidates[cand_frames]
    labels = _fnl.HARMONIX_LABELS  # swapped to the EDM vocab by _set_label_vocab when needed

    triples = _candidate_segments(
        cand_frames, cand_strengths, thr, frame_rate,
        prob_functions, frame_rate, duration, labels,
    )

    global _LAST_ACTIVATIONS
    _LAST_ACTIVATIONS = {
        "cand_frames": cand_frames, "cand_strengths": cand_strengths,
        "prob_functions": prob_functions, "frame_rate": frame_rate,
        "duration": duration, "threshold": thr, "labels": list(labels),
    }
    return [Segment(start=s, end=e, label=lab) for s, e, lab in triples]


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


# Demix performance knobs (Apple Silicon). Defaults are the fast settings; every knob has an
# env override so the eval harness / a regression hunt can reproduce the legacy behavior:
#   JAMS_DEMIX_SHIFTS=1 JAMS_DEMIX_OVERLAP=0.25 JAMS_DEMIX_BATCH=1 JAMS_DEMIX_FP16=0
# - shifts: upstream allin1 runs demucs with shifts=1 (one randomly-shifted pass). shifts=0
#   drops the 0.5 s shift padding; output differences are inaudible at feature level.
# - overlap: upstream 0.25 reprocesses a quarter of every chunk; 0.10 keeps a linear
#   crossfade with ~17% fewer chunk forwards.
# - batch: upstream processes chunks serially; stacking full-length chunks per forward
#   raises MPS utilization. The trailing short chunk keeps the stock unbatched path so its
#   model-internal end-padding matches upstream exactly.
# - fp16: autocast the demucs forward on MPS. Default OFF: measured on an M-series MBP the
#   per-op autocast casts made htdemucs ~60% SLOWER (7.7s -> 12.6s on a 263s track) — MPS
#   fp32 is already the fast path for this graph. The knob stays for future torch versions.
_DEMIX_SHIFTS = int(os.environ.get("JAMS_DEMIX_SHIFTS", "0"))
_DEMIX_OVERLAP = float(os.environ.get("JAMS_DEMIX_OVERLAP", "0.10"))
_DEMIX_BATCH = int(os.environ.get("JAMS_DEMIX_BATCH", "4"))
_DEMIX_FP16 = os.environ.get("JAMS_DEMIX_FP16", "0").lower() not in ("0", "false", "no")


def _batched_split_apply(model, mix, device, overlap: float, batch_size: int,
                         use_autocast: bool):
    """demucs ``apply_model``'s split branch with chunk batching.

    Mirrors the stock triangle-window overlap-add exactly (transition_power=1). Interior
    chunks all have length ``segment_length`` and are stacked ``batch_size`` at a time; the
    trailing short chunk (if any) goes through the stock single-chunk path so demucs'
    model-internal end padding is preserved. Batching only changes evaluation order.
    """
    import torch as th
    from demucs.apply import TensorChunk, apply_model, tensor_chunk

    _batch, _channels, length = mix.shape
    segment_length = int(model.samplerate * model.segment)
    stride = int((1 - overlap) * segment_length)
    offsets = list(range(0, length, stride))
    weight = th.cat([
        th.arange(1, segment_length // 2 + 1),
        th.arange(segment_length - segment_length // 2, 0, -1),
    ]).float()
    weight = weight / weight.max()

    out = th.zeros(_batch, len(model.sources), _channels, length)
    sum_weight = th.zeros(length)
    mix_chunk = tensor_chunk(mix)

    full = [off for off in offsets if off + segment_length <= length]
    tail = [off for off in offsets if off not in set(full)]

    autocast_ctx = (
        th.autocast(device_type="mps", dtype=th.float16)
        if use_autocast else contextlib.nullcontext()
    )
    model.to(device)
    model.eval()
    for i in range(0, len(full), batch_size):
        group = full[i:i + batch_size]
        stacked = th.cat(
            [TensorChunk(mix_chunk, off, segment_length).padded(segment_length)
             for off in group], dim=0).to(device)
        with th.no_grad(), autocast_ctx:
            outs = model(stacked)
        outs = outs.float().cpu()
        for j, off in enumerate(group):
            out[..., off:off + segment_length] += weight * outs[j:j + 1]
            sum_weight[off:off + segment_length] += weight
    for off in tail:  # stock leaf path for the short trailing chunk
        chunk = TensorChunk(mix_chunk, off, segment_length)
        with autocast_ctx:
            chunk_out = apply_model(model, chunk, device=device, shifts=0,
                                    split=False, progress=False).cpu().float()
        chunk_length = chunk_out.shape[-1]
        out[..., off:off + chunk_length] += weight[:chunk_length] * chunk_out
        sum_weight[off:off + chunk_length] += weight[:chunk_length]
    assert float(sum_weight.min()) > 0
    out /= sum_weight
    return out


def _fast_run_demucs_inprocess(path, out_dir, device) -> None:
    """Drop-in for ``allin1.demix._run_demucs_inprocess`` honoring the JAMS_DEMIX_* knobs.

    Identical I/O contract and normalization/seeding; only the demucs invocation differs.
    """
    import random as _random
    import time
    from pathlib import Path as _Path

    import soundfile as sf
    from demucs.apply import apply_model
    from demucs.audio import AudioFile, prevent_clip
    from demucs.pretrained import get_model

    model = get_model("htdemucs")
    model.cpu()
    model.eval()

    wav = AudioFile(path).read(
        streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std()
    _random.seed(0)

    use_fp16 = _DEMIX_FP16 and str(device) == "mps"
    t0 = time.monotonic()
    # ``get_model('htdemucs')`` wraps the single HTDemucs in a BagOfModels; unwrap it for
    # the batched path (exact for one sub-model with unit weights — anything else keeps
    # the stock path, which handles bags itself).
    from demucs.apply import BagOfModels

    single = model
    if isinstance(model, BagOfModels) and len(model.models) == 1:
        single = model.models[0]
    if _DEMIX_BATCH > 1 and _DEMIX_SHIFTS == 0 and not isinstance(single, BagOfModels):
        sources = _batched_split_apply(
            single, wav[None], device, _DEMIX_OVERLAP, _DEMIX_BATCH, use_fp16)[0]
    else:
        import torch as th

        ctx = (th.autocast(device_type="mps", dtype=th.float16)
               if use_fp16 else contextlib.nullcontext())
        with ctx:
            sources = apply_model(
                model, wav[None], device=str(device), shifts=_DEMIX_SHIFTS,
                split=True, overlap=_DEMIX_OVERLAP, progress=True,
            )[0].float()
    print(
        f"[jams] demix {time.monotonic() - t0:.1f}s "
        f"(shifts={_DEMIX_SHIFTS} overlap={_DEMIX_OVERLAP} "
        f"batch={_DEMIX_BATCH} fp16={int(use_fp16)})",
        file=sys.stderr,
    )

    sources *= ref.std()
    sources += ref.mean()
    sources = prevent_clip(sources, mode="rescale")

    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for source, name in zip(sources, model.sources, strict=False):
        sf.write(out_dir / f"{name}.wav", source.cpu().numpy().T,
                 model.samplerate, subtype="PCM_16")


def _patch_demix() -> None:
    """Route allin1's demix through the knob-aware implementation (module-global lookup,
    so patching the attribute is sufficient — same pattern as the loader patches)."""
    import allin1.demix

    allin1.demix._run_demucs_inprocess = _fast_run_demucs_inprocess


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


def analyze(
    audio: str, target_bpm: float | None, model: str, include_activations: bool = False,
) -> dict:
    global _LAST_ACTIVATIONS
    _register_extra_models()
    _patch_demix()
    _set_label_vocab(model)
    _LAST_ACTIVATIONS = None

    try:
        duration = _audio_duration(audio)
    except Exception:  # noqa: BLE001 - ffprobe unavailable -> just run the model directly
        duration = 0.0

    if duration > _LEN_CAP_SEC:
        beats, downbeats, raw_segs, bpm = _run_chunked(audio, model, duration)
        method = f"allin1-mps-local:{model}+chunked"
        # Per-chunk activations don't stitch into one coherent blob; the section-count
        # slider is unavailable for tracks long enough to need chunking.
        activations = None
    else:
        beats, downbeats, raw_segs, bpm = _run_single(audio, model)
        method = f"allin1-mps-local:{model}"
        activations = (
            _activations_blob(_LAST_ACTIVATIONS)
            if include_activations and _LAST_ACTIVATIONS is not None
            else None
        )

    # Keep the native beats (high quality); only octave-correct when target_bpm says the
    # native tempo is a clean half/double. Octave-correct tracks pass through untouched.
    if target_bpm is not None and bpm:
        new_beats, new_downbeats, new_bpm = _octave_correct(
            beats, downbeats, bpm, float(target_bpm))
        if new_bpm != bpm:
            beats, downbeats, bpm = new_beats, new_downbeats, new_bpm
            method += f"+octave{bpm:g}"

    # Snap short, positionally-impossible opening/closing labels (runs on the FINAL stitched
    # segments, so it's correct for both the single and chunked paths).
    raw_segs = _fix_boundary_labels(raw_segs, duration)

    segments = [
        {
            "start": st, "end": en, "label": lab,
            "start_beat": _beat_index(st, beats),
            "end_beat": _beat_index(en, beats),
        }
        for st, en, lab in raw_segs
    ]
    result = {
        "bpm": bpm, "beats": beats, "downbeats": downbeats,
        "segments": segments, "method": method,
    }
    if include_activations:
        result["activations"] = activations
    return result


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
            res = analyze(
                req["audio"], req.get("target_bpm"), req.get("model", "all-all"),
                include_activations=bool(req.get("activations", False)),
            )
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
    ap.add_argument("--activations", action="store_true",
                    help="include the resegmentation activations blob in the output")
    args = ap.parse_args()

    if args.serve:
        _serve()
        return
    if not args.audio:
        ap.error("either --serve or --audio is required")
    print(json.dumps(analyze(args.audio, args.target_bpm, args.model,
                             include_activations=args.activations)))


if __name__ == "__main__":
    main()
