#!/usr/bin/env python
"""Independent rescore of T2b (YourMT3+ on reference Slakh stems) from raw note predictions.

Rebuilds the grouped bass/other reference MIDI exactly as eval/acquire_slakh.py does
(classify + merge_midi, restricted to stems whose audio is present in the redux), reads it
back with evaluate_transcription._midi_to_notes, and scores with note_prf.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]  # repository root
EVIDENCE = W / "paper" / "evidence"
EVAL = W / "eval"
DATA_HOME = EVAL / "data" / "slakh_home"
REF_OUT = EVAL / "data" / "slakh_rescore"  # merged group MIDI written here (gitignored)
NOTES_JSONL = EVAL / "data" / "results_aws" / "yourmt3_notes.jsonl"
ARCHIVE = EVAL / "data" / "results_aws" / "yourmt3_oracle_per_track.json"
SCORES = EVAL / "data" / "results_aws" / "yourmt3_scores.json"
PUB = EVIDENCE / "publication_verified.json"
REMOTE_STEMS = EVIDENCE / "slakh_test_stems_listing.txt"
OUT_JSON = EVIDENCE / "rescore_t2b.json"

sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(W / "src"))

import mir_eval  # noqa: E402
import mirdata  # noqa: E402
import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402
import yaml  # noqa: E402

# --- scorer: import the project's functions verbatim -----------------------
scorer_source = "eval/evaluate_transcription.py (imported)"
try:
    from evaluate_transcription import _midi_to_notes, note_prf  # type: ignore
except Exception as exc:  # noqa: BLE001
    scorer_source = f"copied verbatim from eval/evaluate_transcription.py (import failed: {exc!r})"

    def _midi_to_notes(midi_path: str, canon_drums: bool) -> list[dict]:
        assert not canon_drums
        pm = pretty_midi.PrettyMIDI(midi_path)
        notes: list[dict] = []
        for inst in pm.instruments:
            for n in inst.notes:
                notes.append(
                    {
                        "onset": float(n.start),
                        "offset": float(n.end),
                        "pitch": int(n.pitch),
                        "velocity": int(n.velocity),
                    }
                )
        notes.sort(key=lambda x: x["onset"])
        return notes

    def _midi_hz(pitches: list[int]) -> np.ndarray:
        return np.array([440.0 * 2 ** ((p - 69) / 12) for p in pitches], dtype=float)

    def note_prf(ref: list[dict], est: list[dict]) -> dict:
        if not ref:
            return {"note_f": None, "note_p": None, "note_r": None}
        if not est:
            return {"note_f": 0.0, "note_p": 0.0, "note_r": 0.0}
        ref_i = np.array([[n["onset"], max(n["onset"] + 1e-3, n["offset"])] for n in ref])
        est_i = np.array([[n["onset"], max(n["onset"] + 1e-3, n["offset"])] for n in est])
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_i,
            _midi_hz([n["pitch"] for n in ref]),
            est_i,
            _midi_hz([n["pitch"] for n in est]),
            onset_tolerance=0.05,
            pitch_tolerance=50.0,
            offset_ratio=None,
        )
        return {"note_f": float(f), "note_p": float(p), "note_r": float(r)}


from acquire_slakh import classify, merge_midi  # noqa: E402  (imports mirdata, soundfile)

try:
    from jams.analysis import gm  # noqa: E402

    shift_bass_notes = gm.shift_bass_notes
    monophonic_filter = gm.monophonic_filter
    gm_source = "src/jams/analysis/gm.py (imported)"
except Exception as exc:  # noqa: BLE001
    gm_source = f"copied verbatim from src/jams/analysis/gm.py (import failed: {exc!r})"

    def shift_bass_notes(notes):
        return [{**n, "pitch": min(127, n["pitch"] + 12)} for n in notes]

    def monophonic_filter(notes):
        accepted: list[dict] = []
        for n in sorted(notes, key=lambda x: (-x["velocity"], x["onset"])):
            if any(n["onset"] < a["offset"] and a["onset"] < n["offset"] for a in accepted):
                continue
            accepted.append(n)
        accepted.sort(key=lambda x: x["onset"])
        return accepted


# --- inputs -----------------------------------------------------------------
sha = hashlib.sha256(NOTES_JSONL.read_bytes()).hexdigest()
rows = [json.loads(x) for x in NOTES_JSONL.read_text().splitlines() if x.strip()]
est_by = {(r["track_id"], r["stem"]): r["notes"] for r in rows}
assert len(est_by) == len(rows), "duplicate (track, stem) rows in jsonl"

archive = json.load(open(ARCHIVE))
arch_by = {(r["track_id"], r["stem"]): r for r in archive["per_track"]}
pub = json.load(open(PUB))
pub_ids = sorted(pub["archives"]["T2b"]["track_ids"])
assert len(pub_ids) == 151

remote_audio = set()
for line in REMOTE_STEMS.read_text().splitlines():
    # test/TrackXXXXX/stems/SNN.flac
    parts = line.strip().split("/")
    if len(parts) == 4 and parts[0] == "test":
        remote_audio.add((parts[1], parts[3].removesuffix(".flac")))
print(f"remote audio stems: {len(remote_audio)}", file=sys.stderr)

# --- mirdata (metadata parsing only; never download audio) -----------------
ds = mirdata.initialize("slakh", data_home=str(DATA_HOME), version="2100-redux")
index_fetched = False
if not Path(ds.index_path).exists():
    print("=> fetching mirdata slakh index (small JSON only)", file=sys.stderr)
    ds.download(partial_download=["index"])
    index_fetched = True
print(f"index: {ds.index_path}", file=sys.stderr)

# --- build grouped references, mirroring acquire_slakh.build_mtrack -----------
per_track: list[dict] = []
group_info: dict[str, dict] = {}
eligible = {"bass": [], "other": []}
issues: list[str] = []
audio_flag_mismatch: list[str] = []

test_ids = []
for mid in ds.mtrack_ids:
    mt = ds.multitrack(mid)
    try:
        split = mt.split
    except Exception:  # noqa: BLE001
        split = None
    if split != "test":
        continue
    test_ids.append(mid)
test_ids.sort()
print(f"mirdata test mtracks: {len(test_ids)}", file=sys.stderr)

for mid in test_ids:
    mt = ds.multitrack(mid)
    track_dir = DATA_HOME / "slakh2100_flac_redux" / "test" / mid
    meta = yaml.safe_load((track_dir / "metadata.yaml").read_text())
    groups: dict[str, list] = {"drums": [], "bass": [], "other": []}
    n_index_stems = 0
    for stem_id, stem in mt.tracks.items():
        n_index_stems += 1
        sid = stem_id.split("-")[-1]  # mirdata track ids look like "Track01876-S00"
        # acquire_slakh: skip if audio_path missing on disk. We have no audio locally, so
        # use the remote listing of stems/*.flac from the same redux copy.
        audio_present = bool(stem.audio_path) and (mid, sid) in remote_audio
        rendered_flag = bool(meta["stems"].get(sid, {}).get("audio_rendered"))
        if audio_present != rendered_flag:
            audio_flag_mismatch.append(
                f"{mid}/{sid}: remote_flac={audio_present} audio_rendered={rendered_flag}"
            )
        if not audio_present:
            continue
        groups[classify(stem)].append((sid, stem))
    info = {"n_index_stems": n_index_stems, "n_meta_stems": len(meta["stems"])}
    for cls in ("bass", "other"):
        members = groups[cls]
        midi_paths = [s.midi_path for _, s in members if s.midi_path and Path(s.midi_path).exists()]
        info[cls] = {
            "stems": [
                {
                    "id": sid,
                    "program": s.program_number,
                    "inst_class": s.instrument,
                    "is_drum": s.is_drum,
                }
                for sid, s in members
            ],
            "n_midi": len(midi_paths),
        }
        if not midi_paths:
            info[cls]["ref_midi"] = None
            continue
        out = REF_OUT / mid / f"{cls}.mid"
        ok = merge_midi(midi_paths, is_drum_group=False, out_path=out)
        info[cls]["ref_midi"] = str(out) if ok else None
        if ok:
            eligible[cls].append(mid)
    info["drums_stems"] = [sid for sid, _ in groups["drums"]]
    group_info[mid] = info

# --- eligibility checks -------------------------------------------------------
checks = {
    "mirdata_test_ids_eq_pub_151": sorted(test_ids) == pub_ids,
    "n_mirdata_test": len(test_ids),
    "other_eligible_n": len(eligible["other"]),
    "other_eligible_eq_pub_151": sorted(eligible["other"]) == pub_ids,
    "bass_eligible_n": len(eligible["bass"]),
    "bass_eligible_eq_archive_bass_ids": sorted(eligible["bass"])
    == sorted(t for (t, s) in arch_by if s == "bass"),
    "bass_eligible_eq_jsonl_bass_ids": sorted(eligible["bass"])
    == sorted(t for (t, s) in est_by if s == "bass"),
    "other_empty_tracks": sorted(set(test_ids) - set(eligible["other"])),
    "bass_empty_tracks": sorted(set(test_ids) - set(eligible["bass"])),
    "audio_flag_mismatches": audio_flag_mismatch,
}
print(json.dumps(checks, indent=1), file=sys.stderr)

# --- scoring ------------------------------------------------------------------
BASS_VARIANTS = {
    "raw": lambda n: n,
    "shift12": shift_bass_notes,
    "mono": monophonic_filter,
    "mono_shift12": lambda n: shift_bass_notes(monophonic_filter(n)),
}

results: list[dict] = []
for mid in test_ids:
    for cls in ("other", "bass"):
        ref_path = group_info[mid][cls]["ref_midi"]
        est = est_by.get((mid, cls))
        arch = arch_by.get((mid, cls))
        if ref_path is None and est is None and arch is None:
            continue
        row: dict = {
            "track_id": mid,
            "stem": cls,
            "ref_midi": ref_path,
            "has_est": est is not None,
            "in_archive": arch is not None,
        }
        if ref_path is None or est is None:
            issues.append(
                f"{mid}/{cls}: ref={ref_path is not None} est={est is not None} archive={arch is not None}"
            )
            row["archived"] = arch
            results.append(row)
            continue
        ref = _midi_to_notes(ref_path, canon_drums=False)
        row["n_ref_notes"] = len(ref)
        row["n_est_notes_raw"] = len(est)
        if cls == "other":
            row["scores"] = {"raw": note_prf(ref, est)}
        else:
            row["scores"] = {}
            for name, fn in BASS_VARIANTS.items():
                e = fn(est)
                s = note_prf(ref, e)
                s["n_est_notes"] = len(e)
                row["scores"][name] = s
        row["archived"] = {k: arch[k] for k in ("note_f", "note_p", "note_r")} if arch else None
        results.append(row)


def compare(stem: str, variant: str) -> dict:
    diffs_f, diffs_p, diffs_r, vals, arch_vals = [], [], [], [], []
    for r in results:
        if r["stem"] != stem or "scores" not in r or r["archived"] is None:
            continue
        s = r["scores"][variant]
        vals.append(s["note_f"])
        arch_vals.append(r["archived"]["note_f"])
        diffs_f.append(abs(s["note_f"] - r["archived"]["note_f"]))
        diffs_p.append(abs(s["note_p"] - r["archived"]["note_p"]))
        diffs_r.append(abs(s["note_r"] - r["archived"]["note_r"]))
    n = len(vals)
    return {
        "n": n,
        "recomputed_mean_note_f": round(statistics.mean(vals), 6) if n else None,
        "archived_mean_note_f": round(statistics.mean(arch_vals), 6) if n else None,
        "max_abs_diff_note_f": max(diffs_f) if n else None,
        "max_abs_diff_note_p": max(diffs_p) if n else None,
        "max_abs_diff_note_r": max(diffs_r) if n else None,
        "n_tracks_diff_gt_1e-4": sum(d > 1e-4 for d in diffs_f),
        "n_tracks_diff_gt_1e-6": sum(d > 1e-6 for d in diffs_f),
    }


summary = {
    "other": {"raw": compare("other", "raw")},
    "bass": {v: compare("bass", v) for v in BASS_VARIANTS},
}
# Which bass variant reproduces the archive?
summary["bass_variant_matching_archive"] = [
    v
    for v, c in summary["bass"].items()
    if c["max_abs_diff_note_f"] is not None and c["max_abs_diff_note_f"] <= 1e-6
]
summary["archived_aggregate"] = archive["aggregate_note_f"]
summary["yourmt3_scores_json"] = json.load(open(SCORES))

# note-count table
note_counts = {r["track_id"]: {} for r in results}
for r in results:
    if "n_ref_notes" in r:
        note_counts[r["track_id"]][r["stem"]] = {
            "ref": r["n_ref_notes"],
            "est_raw": r["n_est_notes_raw"],
        }

out = {
    "method": "independent rescore of T2b from eval/data/results_aws/yourmt3_notes.jsonl",
    "date": "2026-09-08",
    "inputs": {
        "yourmt3_notes_jsonl": str(NOTES_JSONL.relative_to(W)),
        "yourmt3_notes_jsonl_sha256": sha,
        "yourmt3_notes_jsonl_rows": len(rows),
        "archive": str(ARCHIVE.relative_to(W)),
        "slakh_source": "slakh2100_flac_redux test/ MIDI + metadata.yaml + all_src.mid copied from LAN host aleph0:/home/jhurliman/s7/slakh_home (audio not copied; stems/*.flac presence taken from a directory listing of the same copy)",
        "slakh_local": str(DATA_HOME.relative_to(W)),
        "mirdata_index": str(ds.index_path),
        "mirdata_index_fetched_now": index_fetched,
    },
    "scoring": {
        "scorer": scorer_source,
        "gm_helpers": gm_source,
        "note_prf": "mir_eval.transcription.precision_recall_f1_overlap onset_tolerance=0.05 pitch_tolerance=50 cents offset_ratio=None (one-to-one bipartite matching), no beat quantization",
        "reference": "acquire_slakh.classify + merge_midi over stems with rendered audio and existing MIDI; merged MIDI written to disk and read back with evaluate_transcription._midi_to_notes (as the historical pipeline did)",
        "bass_variants": {
            "raw": "jsonl notes as-is",
            "shift12": "gm.shift_bass_notes (+12, cap 127)",
            "mono": "gm.monophonic_filter",
            "mono_shift12": "monophonic_filter then shift_bass_notes (production order in stems.py)",
        },
        "other": "no shift, no filter",
    },
    "versions": {
        "python": sys.version.split()[0],
        "mir_eval": mir_eval.__version__,
        "pretty_midi": pretty_midi.__version__,
        "numpy": np.__version__,
        "mirdata": mirdata.__version__,
    },
    "eligibility": checks,
    "summary": summary,
    "issues": issues,
    "note_counts": note_counts,
    "group_composition": group_info,
    "per_track": results,
}
# --- gate: only a fully successful rescore may overwrite the canonical evidence ------
failures: list[str] = []
for key in (
    "mirdata_test_ids_eq_pub_151",
    "other_eligible_eq_pub_151",
    "bass_eligible_eq_archive_bass_ids",
    "bass_eligible_eq_jsonl_bass_ids",
):
    if checks[key] is not True:
        failures.append(f"eligibility check {key} is {checks[key]!r}")
if checks["other_eligible_n"] != 151:
    failures.append(f"other support {checks['other_eligible_n']} != 151")
if checks["bass_eligible_n"] != 143:
    failures.append(f"bass support {checks['bass_eligible_n']} != 143")
if checks["audio_flag_mismatches"]:
    failures.append(f"audio_rendered flag mismatches: {checks['audio_flag_mismatches']}")
if issues:
    failures.append(f"{len(issues)} per-track issues (missing ref/est/archive rows)")
for stem, variant, n_expected in (("other", "raw", 151), ("bass", "shift12", 143)):
    c = summary[stem][variant]
    if c["n"] != n_expected:
        failures.append(f"{stem}/{variant}: compared {c['n']} tracks, expected {n_expected}")
    for metric in ("max_abs_diff_note_f", "max_abs_diff_note_p", "max_abs_diff_note_r"):
        if c[metric] is None or c[metric] > 1e-6:
            failures.append(f"{stem}/{variant}: {metric} = {c[metric]!r} exceeds 1e-6")
out["gate"] = {
    "passed": not failures,
    "failures": failures,
    "rule": "all eligibility checks true; supports 151/143; no per-track issues; "
    "other(raw) and bass(shift12) P/R/F within 1e-6 of the archive on every track",
}
print(json.dumps(summary, indent=1))
print(f"issues: {issues}")
if failures:
    failed_path = OUT_JSON.with_name(OUT_JSON.stem + ".FAILED.json")
    failed_path.write_text(json.dumps(out, indent=1))
    print(
        "RESCORE GATE FAILED; canonical evidence NOT written. Report saved to",
        failed_path,
        file=sys.stderr,
    )
    for f in failures:
        print("  -", f, file=sys.stderr)
    sys.exit(1)
OUT_JSON.write_text(json.dumps(out, indent=1))
print(f"gate passed; wrote {OUT_JSON}")
