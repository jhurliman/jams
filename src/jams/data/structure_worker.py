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
import sys


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
    import allin1.models  # noqa: F401

    loaders.load_pretrained_model = _patched
    sys.modules["allin1.models"].load_pretrained_model = _patched
    sys.modules["allin1.analyze"].load_pretrained_model = _patched


def _set_label_vocab(model: str) -> None:
    """Point allin1's segment-label decoder at the right vocabulary for ``model``.

    The port decodes function classes through a module-level ``HARMONIX_LABELS`` list;
    EDM models (``all-*``/``raveform-*``) use the 11-class Raveform vocab instead.
    """
    import allin1.config
    import allin1.postprocessing.functional as fnl

    is_edm = model.startswith(("all-", "raveform-"))
    fnl.HARMONIX_LABELS = _RAVEFORM_LABELS if is_edm else allin1.config.HARMONIX_LABELS


def analyze(audio: str, target_bpm: float | None, model: str) -> dict:
    import allin1

    _register_extra_models()
    _set_label_vocab(model)

    # All-In-One prints progress to stdout; keep our stdout protocol clean.
    with contextlib.redirect_stdout(sys.stderr):
        result = allin1.analyze(
            paths=audio, model=model, include_activations=False, keep_byproducts=False,
        )

    beats = [float(b) for b in result.beats]
    downbeats = [float(d) for d in result.downbeats]
    bpm = float(result.bpm) if result.bpm is not None else None
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
            "start": float(s.start), "end": float(s.end), "label": s.label,
            "start_beat": _beat_index(float(s.start), beats),
            "end_beat": _beat_index(float(s.end), beats),
        }
        for s in result.segments
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
