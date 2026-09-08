"""Corpus diversity report — aggregate the manifest to show the realized variety.

Confirms the corpus is not monotimbral: engine mix (Surge / Dexed / Vitalium), Surge osc engines,
bass families, drum real-source kits, tempo/key/sidechain spread, arrangement lengths.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _rows(corpus: str) -> list[dict]:
    p = Path(corpus) / "manifest.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x]


def report(corpus: str) -> dict:
    rows = _rows(corpus)
    substyle = Counter(r["substyle"] for r in rows)
    scale = Counter(r["scale"] for r in rows)
    key_root = Counter(r["key_root_pc"] for r in rows)
    sidechain = Counter(r["sidechain"]["style"] for r in rows)
    bass_fam = Counter()
    engines = Counter()
    surge_osc = Counter()
    drum_src = Counter()
    patterns = Counter()
    n_bass_layers = Counter()
    n_synth_layers = Counter()
    for r in rows:
        t = r.get("timbres", {})
        for d in t.get("bass", []):
            if d.get("family"):
                bass_fam[d["family"]] += 1
            eng = d.get("engine", "surge" if d.get("osc") else "surge")
            engines[eng] += 1
            if d.get("osc"):
                surge_osc[d["osc"]] += 1
        for d in t.get("other", []):
            eng = d.get("engine", "surge")
            engines[eng] += 1
            if eng == "surge" and d.get("osc"):
                surge_osc[d["osc"]] += 1
        dd = t.get("drums", {})
        drum_src[dd.get("real_source", "none")] += 1
        patterns[dd.get("pattern", "?")] += 1
        n_bass_layers[len(r.get("bass_families", []))] += 1
        n_synth_layers[len(r.get("synth_roles", []))] += 1

    tempos = [r["bpm"] for r in rows]
    lufs = [r["lufs_target"] for r in rows]
    bars = [r["total_bars"] for r in rows]
    return {
        "n_tracks": len(rows),
        "substyle_counts": dict(substyle),
        "synth_engine_mix": dict(engines),
        "surge_osc_engines": dict(surge_osc.most_common()),
        "bass_family_counts": dict(bass_fam.most_common()),
        "drum_real_source": dict(drum_src),
        "drum_patterns": dict(patterns.most_common()),
        "scale_modes": dict(scale.most_common()),
        "key_roots_used": len(key_root),
        "sidechain_styles": dict(sidechain),
        "bass_layer_counts": dict(sorted(n_bass_layers.items())),
        "synth_layer_counts": dict(sorted(n_synth_layers.items())),
        "tempo_bpm": {"min": round(min(tempos), 1), "max": round(max(tempos), 1),
                      "mean": round(sum(tempos) / len(tempos), 1)},
        "lufs_target": {"min": round(min(lufs), 1), "max": round(max(lufs), 1)},
        "arrangement_bars": {"min": min(bars), "max": max(bars)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rep = report(args.corpus)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
