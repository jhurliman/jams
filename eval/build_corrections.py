#!/usr/bin/env -S uv run --extra eval
"""Build eval/tempo_corrections.csv — curated fixes for wrong GiantSteps Key tempo labels.

The GiantSteps *Key* dataset carries Beatport metadata tempos, which are wrong for
half-time genres (Drum & Bass labeled ~87 when the true tempo is ~174, etc.). We
correct them from two sources, in priority order:

  1. AUTHORITATIVE — GiantSteps *Tempo* v2 (Schreiber's expert re-annotation of the
     same Beatport tracks). Joined by Beatport ID (the Key title prefix). Exact values.
  2. CONVENTION — for D&B/jungle/dubstep/footwork tracks NOT covered by (1) that are
     labeled in clean half-time (78-98 BPM), double to the conventional full tempo.
     Flagged needs_review=yes for manual confirmation.

Writes track_id, title, genre, key_label, v2, model_raw, corrected_tempo, source,
confidence, needs_review. Re-run after re-acquiring datasets:  uv run eval/build_corrections.py
"""
import csv, glob, json, os, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL_TEMPO_GENRES = ("drum & bass", "drum and bass", "jungle", "dubstep", "future bass", "footwork")


def near(a, b, tol=0.04):
    return bool(a and b and abs(a - b) <= tol * max(a, b))


def main():
    import mirdata
    dt = mirdata.initialize("giantsteps_tempo", data_home=str(HERE / "data" / "giantsteps_tempo"))
    try:
        dt.download(["annotations"], force_overwrite=False)
    except Exception:
        pass
    base = glob.glob(str(HERE / "data" / "giantsteps_tempo" / "giantsteps-tempo-dataset-*"))
    if not base:
        raise SystemExit("GiantSteps Tempo annotations not found; run with network access.")
    v2 = {}
    for f in glob.glob(f"{base[0]}/annotations_v2/tempo/*.bpm"):
        try:
            v2[os.path.basename(f).split('.')[0]] = float(open(f).read().strip())
        except ValueError:
            pass

    man = [json.loads(l) for l in (HERE / "data" / "manifest.jsonl").read_text().splitlines() if l.strip()]
    res_path = HERE / "data" / "results_sota.json"
    model = {}
    if res_path.is_file():
        model = {r["track_id"]: r.get("pred_tempo") for r in json.loads(res_path.read_text())}

    def bid(t):
        m = re.match(r'(\d+)', t or ""); return m.group(1) if m else None

    rows = []
    for r in man:
        if not r.get("ref_tempo"):
            continue
        g = (r.get("genres") or ["?"])[0]
        k = r["ref_tempo"]
        v2c = v2.get(bid(r.get("title")))
        corr = src = conf = rev = None
        if v2c and not near(v2c, k):
            corr, src, conf, rev = v2c, "giantsteps-tempo-v2", "high", "no"
        elif any(x in g.lower() for x in FULL_TEMPO_GENRES) and 78 <= k <= 98:
            corr, src, conf, rev = round(k * 2), "convention(full-tempo x2)", "medium", "yes"
        if corr:
            rows.append(dict(track_id=r["track_id"], title=(r.get("title") or "")[:46], genre=g,
                             key_label=k, v2=(v2c or ""), model_raw=model.get(r["track_id"], ""),
                             corrected_tempo=corr, source=src, confidence=conf, needs_review=rev))
    rows.sort(key=lambda x: (x["confidence"] != "high", x["genre"]))
    out = HERE / "tempo_corrections.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    hi = sum(1 for x in rows if x["confidence"] == "high")
    print(f"wrote {out}: {len(rows)} corrections ({hi} v2-authoritative, {len(rows)-hi} convention/needs-review)")


if __name__ == "__main__":
    main()
