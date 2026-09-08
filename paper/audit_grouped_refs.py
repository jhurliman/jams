#!/usr/bin/env python
"""Audit the grouped Slakh2100-redux test references (REVIEW.md item 4).

For each of the 151 test multitracks and each group (drums / bass / other) built by
eval/acquire_slakh.py, checks:
  completeness  every stem FLAC on disk is classified as the script does; a stem whose
                FLAC exists but cannot be read drops out of the audio sum while its MIDI
                still merges ("phantom" MIDI); the reverse (audio, no MIDI) is also listed.
  duration      the sum is truncated to the shortest member; report the truncation and
                whether the grouped WAV length equals min(member frames); MIDI notes whose
                onset/offset lie beyond the audio end.
  clipping      recompute the float64 sum exactly as sum_audio does; peak |sum| and the
                number/fraction of samples outside [-1, 1] (these clip in the PCM_16 write);
                confirm the grouped WAV subtype and that it equals clip(sum) to 1 LSB.
  alignment     cross-correlate the grouped WAV's onset-strength envelope with the merged
                MIDI's onset impulse train (full duration, +-500 ms); report the global best lag
                (positive = audio later than MIDI), the best lag within 0..70 ms (the expected
                note-on -> acoustic-peak delay incl. ~1 frame estimator bias), their ratio, and
                the curve at lags -8..+8 frames.

Runs on the host holding the redux and the grouped references; writes one JSON report
(committed as paper/evidence/grouped_reference_audit.json). Usage: python paper/audit_grouped_refs.py [workers]
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
import yaml

import os

# Defaults are the author workstation (aleph0) layout used for the 2026-09-08 audit;
# override with environment variables to run elsewhere.
REDUX = Path(os.environ.get("SLAKH_REDUX_TEST", Path.home() / "s7/slakh_home/slakh2100_flac_redux/test"))
GROUPED = Path(os.environ.get("SLAKH_GROUPED", Path.home() / "s7/jams/eval/data/slakh/redux"))
OUT = Path(os.environ.get("AUDIT_OUT", Path.home() / "s7/audit_grouped_refs.json"))
HOP = 512
ALIGN_SECONDS = None  # full duration
LSB = 1 / 32768


def classify(m: dict) -> str:
    if m.get("is_drum"):
        return "drums"
    prog = m.get("program_num")
    if (prog is not None and 32 <= prog <= 39) or m.get("inst_class") == "Bass":
        return "bass"
    return "other"


def audit_track(tid: str) -> dict:
    tdir = REDUX / tid
    meta = yaml.safe_load(open(tdir / "metadata.yaml"))
    groups: dict[str, list[dict]] = {"drums": [], "bass": [], "other": []}
    stem_rows = []
    for sid, m in sorted(meta["stems"].items()):
        flac = tdir / "stems" / f"{sid}.flac"
        mid = tdir / "MIDI" / f"{sid}.mid"
        row = {"stem": sid, "cls": classify(m), "audio_rendered": bool(m.get("audio_rendered")),
               "flac_on_disk": flac.exists(), "midi_on_disk": mid.exists(), "flac_readable": False,
               "frames": None, "sr": None, "channels": None}
        if flac.exists():
            try:
                info = sf.info(str(flac))
                row.update(flac_readable=True, frames=info.frames, sr=info.samplerate, channels=info.channels)
            except Exception as exc:  # noqa: BLE001
                row["read_error"] = str(exc)
            groups[row["cls"]].append(row)
        stem_rows.append(row)
    mix_info = sf.info(str(tdir / "mix.flac"))
    out = {"track_id": tid, "mix_frames": mix_info.frames, "mix_sr": mix_info.samplerate,
           "stems": stem_rows, "groups": {}}
    for cls, members in groups.items():
        if not members:
            out["groups"][cls] = {"members": 0}
            continue
        readable = [m for m in members if m["flac_readable"]]
        g = {"members": len(members), "audio_members": len(readable),
             "midi_members": sum(1 for m in members if m["midi_on_disk"]),
             "phantom_midi_stems": [m["stem"] for m in members if m["midi_on_disk"] and not m["flac_readable"]],
             "audio_without_midi": [m["stem"] for m in members if m["flac_readable"] and not m["midi_on_disk"]]}
        if not readable:
            out["groups"][cls] = g
            continue
        frames = [m["frames"] for m in readable]
        n, sr = min(frames), readable[0]["sr"]
        ch = max(m["channels"] for m in readable)
        g.update(sr=sr, min_frames=n, max_frames=max(frames), truncated_samples=max(frames) - n,
                 truncated_seconds=(max(frames) - n) / sr, mix_minus_group_frames=mix_info.frames - n)
        mix = np.zeros((n, ch), dtype=np.float64)
        for m in readable:
            d, _ = sf.read(str(tdir / "stems" / f"{m['stem']}.flac"), always_2d=True)
            d = d[:n]
            if d.shape[1] == 1 and ch == 2:
                d = np.repeat(d, 2, axis=1)
            mix[:, : d.shape[1]] += d
        over = int((np.abs(mix) > 1.0).sum())
        g.update(sum_peak=float(np.abs(mix).max()), clipped_samples=over,
                 clipped_fraction=over / mix.size, sum_channels=ch)
        wavp, midp = GROUPED / tid / f"{cls}.wav", GROUPED / tid / f"{cls}.mid"
        g.update(wav_exists=wavp.exists(), mid_exists=midp.exists())
        w = None
        if wavp.exists():
            info = sf.info(str(wavp))
            w, _ = sf.read(str(wavp), always_2d=True)
            g.update(wav_subtype=info.subtype, wav_frames=info.frames, wav_channels=info.channels,
                     wav_duration_s=info.frames / info.samplerate,
                     wav_frames_match_min=(info.frames == n),
                     wav_full_scale_samples=int((np.abs(w) >= 32767 / 32768).sum()))
            if info.frames == n and w.shape[1] == ch:
                g["max_abs_diff_vs_recomputed"] = float(np.abs(w - np.clip(mix, -1, 1)).max())
        if midp.exists():
            pm = pretty_midi.PrettyMIDI(str(midp))
            notes = [nt for inst in pm.instruments for nt in inst.notes]
            g.update(midi_notes=len(notes),
                     midi_first_onset_s=min((nt.start for nt in notes), default=None),
                     midi_last_offset_s=max((nt.end for nt in notes), default=None))
            if w is not None and notes:
                dur = g["wav_duration_s"]
                g["notes_onset_after_audio_end"] = sum(1 for nt in notes if nt.start > dur)
                g["notes_offset_after_audio_end"] = sum(1 for nt in notes if nt.end > dur)
                mono = w.mean(axis=1).astype(np.float32)
                if ALIGN_SECONDS:
                    mono = mono[: int(ALIGN_SECONDS * sr)]
                env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=HOP)
                fr = sr / HOP
                imp = np.zeros_like(env)
                for nt in notes:
                    f = int(round(nt.start * fr))
                    if 0 <= f < len(imp):
                        imp[f] += 1.0
                env = (env - env.mean()) / (env.std() + 1e-9)
                imp = (imp - imp.mean()) / (imp.std() + 1e-9)
                maxlag = int(round(0.5 * fr))
                curve = {}
                for lag in range(-maxlag, maxlag + 1):
                    if lag >= 0:
                        c = float(np.dot(env[lag:], imp[: len(imp) - lag])) / len(env)
                    else:
                        c = float(np.dot(env[:lag], imp[-lag:])) / len(env)
                    curve[lag] = c
                best = max(curve.items(), key=lambda kv: kv[1])
                # Expected lag is small and positive: the estimator adds ~1 frame (measured on
                # synthetic impulses) and sampled-instrument attacks peak after MIDI note-on.
                near = {lag: c for lag, c in curve.items() if 0 <= lag <= 6}  # 0..70 ms (bass attacks ~58 ms)
                near_best = max(near.items(), key=lambda kv: kv[1])
                g.update(align_lag_ms=best[0] / fr * 1000.0, align_corr=best[1],
                         align_corr_at_zero=curve[0], align_frame_ms=1000.0 / fr,
                         align_near_lag_ms=near_best[0] / fr * 1000.0, align_near_corr=near_best[1],
                         align_near_ratio=(near_best[1] / best[1]) if best[1] > 0 else None,
                         align_near_is_global_max=(near_best[0] == best[0]),
                         align_curve_lags_m8_p8=[round(curve[l], 4) for l in range(-8, 9)])
        out["groups"][cls] = g
    return out


def summarize(rows: list[dict]) -> dict:
    s = {"tracks": len(rows), "groups": 0, "phantom_midi_groups": [], "audio_without_midi_groups": [],
         "truncated_groups": 0, "max_truncated_seconds": 0.0, "clipped_groups": 0,
         "max_clipped_fraction": 0.0, "max_sum_peak": 0.0, "wav_subtypes": {},
         "wav_length_mismatch": [], "wav_vs_recomputed_over_1lsb": [],
         "notes_after_audio_end_groups": [], "align_abs_lag_over_46ms": [],
         "align_lag_ms_hist": {}, "missing_wav": [], "missing_mid": [],
         "align_global_max_near_zero": 0, "align_near_ratio_below_0_9": [], "align_low_corr_below_0_3": [],
         "align_near_lag_ms_hist_by_class": {"drums": {}, "bass": {}, "other": {}}}
    for r in rows:
        for cls, g in r["groups"].items():
            if g.get("members", 0) == 0:
                continue
            s["groups"] += 1
            key = f"{r['track_id']}/{cls}"
            if g["phantom_midi_stems"]:
                s["phantom_midi_groups"].append([key, g["phantom_midi_stems"]])
            if g["audio_without_midi"]:
                s["audio_without_midi_groups"].append([key, g["audio_without_midi"]])
            if g.get("truncated_samples", 0) > 0:
                s["truncated_groups"] += 1
                s["max_truncated_seconds"] = max(s["max_truncated_seconds"], g["truncated_seconds"])
            if g.get("clipped_samples", 0) > 0:
                s["clipped_groups"] += 1
                s["max_clipped_fraction"] = max(s["max_clipped_fraction"], g["clipped_fraction"])
            s["max_sum_peak"] = max(s["max_sum_peak"], g.get("sum_peak", 0.0))
            if not g.get("wav_exists"):
                s["missing_wav"].append(key)
            else:
                s["wav_subtypes"][g["wav_subtype"]] = s["wav_subtypes"].get(g["wav_subtype"], 0) + 1
                if not g.get("wav_frames_match_min"):
                    s["wav_length_mismatch"].append([key, g.get("wav_frames"), g.get("min_frames")])
                if g.get("max_abs_diff_vs_recomputed", 0.0) > LSB:
                    s["wav_vs_recomputed_over_1lsb"].append([key, g["max_abs_diff_vs_recomputed"]])
            if not g.get("mid_exists"):
                s["missing_mid"].append(key)
            if g.get("notes_onset_after_audio_end", 0) or g.get("notes_offset_after_audio_end", 0):
                s["notes_after_audio_end_groups"].append([key, g.get("notes_onset_after_audio_end"), g.get("notes_offset_after_audio_end")])
            if "align_lag_ms" in g:
                b = str(int(round(g["align_lag_ms"] / 10.0)) * 10)
                s["align_lag_ms_hist"][b] = s["align_lag_ms_hist"].get(b, 0) + 1
                if abs(g["align_lag_ms"]) > 46:
                    s["align_abs_lag_over_46ms"].append([key, round(g["align_lag_ms"], 1), round(g["align_corr"], 3), round(g["align_near_lag_ms"], 1), round(g["align_near_corr"], 3)])
                if g["align_near_is_global_max"]:
                    s["align_global_max_near_zero"] += 1
                nb = str(int(round(g["align_near_lag_ms"])))
                h = s["align_near_lag_ms_hist_by_class"][cls]
                h[nb] = h.get(nb, 0) + 1
                if g["align_near_ratio"] is None or g["align_near_ratio"] < 0.9:
                    s["align_near_ratio_below_0_9"].append([key, g["align_near_ratio"], round(g["align_lag_ms"], 1)])
                if g["align_corr"] < 0.3:
                    s["align_low_corr_below_0_3"].append([key, round(g["align_corr"], 3), g.get("midi_notes")])
    return s


def main() -> None:
    tids = sorted(p.name for p in REDUX.iterdir() if p.is_dir() and p.name.startswith("Track"))
    t0 = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as pool:
        rows = pool.map(audit_track, tids)
    report = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": "aleph0",
              "redux": str(REDUX), "grouped": str(GROUPED), "align_window_s": ALIGN_SECONDS,
              "versions": {"numpy": np.__version__, "soundfile": sf.__version__,
                           "librosa": librosa.__version__, "pretty_midi": pretty_midi.__version__},
              "summary": summarize(rows), "tracks": rows}
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["summary"], indent=1))
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
