"""Validate the corpus: (a) stem-sum residual, (b) shipped-SCNet SI-SDR realism DISTRIBUTION.

(a) confirms GT stems sum to the premaster mix (near -inf on the float buffers; a 16-bit-FLAC
    quantization floor on disk).
(b) runs the SHIPPED single-pass SCNet XL IHF on rendered premaster mixes spanning sub-styles
    and reports the per-stem SI-SDR *distribution* (not per-track), to confirm we sit in
    "regime (a)" — an imperfect, real-music difficulty band — and haven't drifted trivially-clean
    or broken. A tight cluster is a red flag even if the mean looks good.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

_DEFAULT_WORKER = "/Users/jhurliman/Documents/Code/jhurliman/jams/.claude/worktrees/es2-dnb/" \
                  "src/jams/data/stems_worker.py"
STEMS_WORKER = os.environ.get("SYNTH_STEMS_WORKER", _DEFAULT_WORKER)
_PITCHED = ("drums", "bass", "other")


def _mono(path: str) -> np.ndarray:
    y, _ = sf.read(path)
    return y.mean(1) if y.ndim > 1 else y


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float | None:
    ref = ref - ref.mean()
    est = est - est.mean()
    if (ref ** 2).sum() < 1e-9:
        return None
    alpha = (est * ref).sum() / (ref ** 2).sum()
    proj = alpha * ref
    noise = est - proj
    return float(10 * np.log10((proj ** 2).sum() / ((noise ** 2).sum() + 1e-12) + 1e-12))


def _stemsum_residual(track_dir: Path) -> float:
    mix = _mono(str(track_dir / "mix_premaster.flac"))
    recon = np.zeros_like(mix)
    for s in ("drums", "bass", "other", "vocals"):
        y = _mono(str(track_dir / f"{s}.flac"))
        n = min(len(recon), len(y))
        recon[:n] += y[:n]
    n = min(len(mix), len(recon))
    res = mix[:n] - recon[:n]
    return round(float(20 * np.log10((np.sqrt((res ** 2).mean()) + 1e-12)
                                     / (np.sqrt((mix[:n] ** 2).mean()) + 1e-12))), 1)


class _SCNet:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            ["uv", "run", "--script", STEMS_WORKER, "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def separate(self, audio: str, out_dir: str) -> dict[str, str]:
        req = json.dumps({"audio": audio, "out_dir": out_dir, "model": "scnet_xl_ihf"})
        self.proc.stdin.write(req + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "scnet failed"))
        return {s["stem_type"]: s["audio_path"] for s in resp["result"]["stems"]}

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.proc.stdin.close()
            self.proc.terminate()


def _select(manifest: list[dict], split: dict, n_per_sub: int) -> list[dict]:
    val = set(split.get("val", []))
    by_sub: dict[str, list[dict]] = {}
    for r in manifest:
        if not val or r["track_id"] in val:
            by_sub.setdefault(r["substyle"], []).append(r)
    picks = []
    for sub in sorted(by_sub):
        rows = sorted(by_sub[sub], key=lambda r: r["seed"])
        step = max(1, len(rows) // n_per_sub)
        picks.extend(rows[::step][:n_per_sub])
    return picks


def _dist(vals: list[float]) -> dict:
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {}
    return {"n": int(a.size), "min": round(float(a.min()), 1),
            "p25": round(float(np.percentile(a, 25)), 1),
            "median": round(float(np.median(a)), 1),
            "p75": round(float(np.percentile(a, 75)), 1),
            "max": round(float(a.max()), 1), "mean": round(float(a.mean()), 1),
            "std": round(float(a.std()), 1)}


def run(corpus: str, n_per_sub: int, out_path: str) -> dict:
    cdir = Path(corpus)
    manifest = [json.loads(x) for x in (cdir / "manifest.jsonl").read_text().splitlines() if x]
    split = json.loads((cdir / "split.json").read_text()) if (cdir / "split.json").exists() else {}
    picks = _select(manifest, split, n_per_sub)

    residuals = {r["track_id"]: _stemsum_residual(cdir / "audio" / r["track_id"])
                 for r in picks[: min(len(picks), 8)]}

    scnet = _SCNet()
    per_track: list[dict] = []
    by_sub_stem: dict[str, dict[str, list]] = {}
    overall: dict[str, list] = {s: [] for s in _PITCHED}
    with tempfile.TemporaryDirectory() as td:
        for r in picks:
            tdir = cdir / "audio" / r["track_id"]
            wav = str(Path(td) / f"{r['track_id']}.wav")
            sf.write(wav, _mono(str(tdir / "mix_premaster.flac")), 44100)
            odir = str(Path(td) / r["track_id"])
            est = scnet.separate(wav, odir)
            row = {"track_id": r["track_id"], "substyle": r["substyle"]}
            for s in _PITCHED:
                gt = _mono(str(tdir / f"{s}.flac"))
                pred = _mono(est[s])
                m = min(len(gt), len(pred))
                v = si_sdr(pred[:m], gt[:m])
                row[s] = None if v is None else round(v, 1)
                overall[s].append(v)
                by_sub_stem.setdefault(r["substyle"], {}).setdefault(s, []).append(v)
            per_track.append(row)
            print(f"  {r['track_id']}: "
                  + " ".join(f"{s}={row[s]}" for s in _PITCHED), flush=True)
    scnet.close()

    report = {
        "n_tracks_scnet": len(per_track),
        "stemsum_residual_db_on_disk_flac16": residuals,
        "si_sdr_overall_distribution": {s: _dist(overall[s]) for s in _PITCHED},
        "si_sdr_by_substyle": {
            sub: {s: _dist(by_sub_stem[sub].get(s, [])) for s in _PITCHED}
            for sub in sorted(by_sub_stem)},
        "per_track": per_track,
    }
    Path(out_path).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n-per-sub", type=int, default=3)
    ap.add_argument("--out", default="validation_report.json")
    args = ap.parse_args()
    report = run(args.corpus, args.n_per_sub, args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "per_track"}, indent=2))


if __name__ == "__main__":
    main()
