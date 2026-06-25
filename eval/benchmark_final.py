#!/usr/bin/env -S uv run --extra eval
"""Final method shoot-out (essentia-tensorflow env: classic + deep models together).

Key   : edma profile  vs  ensemble vote across {edma, bgate, shaath}
Tempo : rhythm2013(multifeature)  vs  TempoCNN deepsquare  vs  octave-corrected
        ensemble (rhythm2013 fine value snapped to the TempoCNN octave).

    uv run eval/benchmark_final.py --limit 200
"""
from __future__ import annotations
import argparse, json, warnings
from collections import Counter, defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")

NOTES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
PITCH = {n:i for i,n in enumerate(NOTES)}
FLAT = {"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#","Cb":"B","Fb":"E","E#":"F","B#":"C"}

def nk(t,s):
    t = FLAT.get(t,t)
    return f"{t} {'minor' if 'min' in s.lower() else 'major'}" if t in PITCH else None
def parse(k):
    p = k.split() if k else []
    return (PITCH[p[0]],p[1]) if len(p)==2 and p[0] in PITCH else None
def mirex(ref,pred):
    r,p = parse(ref),parse(pred)
    if not r or not p: return 0.0
    iv=(p[0]-r[0])%12
    if iv==0 and p[1]==r[1]: return 1.0
    if p[1]==r[1] and iv in (7,5): return 0.5
    if r[1]=="major" and p[1]=="minor" and iv==9: return 0.3
    if r[1]=="minor" and p[1]=="major" and iv==3: return 0.3
    if iv==0 and p[1]!=r[1]: return 0.2
    return 0.0
def tacc(ref,pred,tol=0.04):
    if not ref or not pred: return None
    a1 = abs(pred-ref)<=tol*ref
    a2 = a1 or any(abs(pred*f-ref)<=tol*ref for f in (0.5,2.0,1/3,3.0))
    return a1,a2
def octave_snap(fine, anchor):
    """Snap `fine` (precise but octave-prone) to the octave nearest `anchor`."""
    if not fine or not anchor: return fine
    best = min((0.25,0.5,1,2,3,4), key=lambda f: abs(fine*f-anchor))
    return fine*best

def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--manifest", type=Path, default=here/"data"/"manifest.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    import essentia
    essentia.log.infoActive = essentia.log.warningActive = False
    import essentia.standard as es
    from tqdm import tqdm
    from jams.analysis.tempo import _MODEL_PATH

    edma = es.KeyExtractor(profileType="edma")
    bgate = es.KeyExtractor(profileType="bgate")
    shaath = es.KeyExtractor(profileType="shaath")
    r2013 = es.RhythmExtractor2013(method="multifeature")
    deepsq = es.TempoCNN(graphFilename=str(_MODEL_PATH))

    recs = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("audio_exists")][:args.limit]

    kw = defaultdict(list); kx = defaultdict(list)
    th = defaultdict(lambda:[0,0,0])
    for r in tqdm(recs, desc="final", unit="trk"):
        a44 = es.MonoLoader(filename=r["audio_path"], sampleRate=44100)()
        a11 = es.MonoLoader(filename=r["audio_path"], sampleRate=11025)()
        ref_k = r["ref_key"]; ref_t = r.get("ref_tempo")

        ke = nk(*edma(a44)[:2]); kb = nk(*bgate(a44)[:2]); ks = nk(*shaath(a44)[:2])
        votes = Counter([x for x in (ke,kb,ks) if x])
        kens = ke if (votes and votes.most_common(1)[0][1]==1) else (votes.most_common(1)[0][0] if votes else ke)
        for name,pred in (("edma",ke),("ensemble",kens)):
            w = mirex(ref_k,pred); kw[name].append(w); kx[name].append(1.0 if w==1.0 else 0.0)

        bpm_r = float(r2013(a44)[0])
        bpm_d = float(deepsq(a11)[0])
        bpm_e = octave_snap(bpm_r, bpm_d)
        for name,bpm in (("rhythm2013",bpm_r),("deepsquare",bpm_d),("ens_octcorr",bpm_e)):
            acc = tacc(ref_t,bpm)
            if acc: h=th[name]; h[0]+=1; h[1]+=acc[0]; h[2]+=acc[1]

    n=len(recs)
    print(f"\n==== KEY (n={n}) ====")
    for name in ("edma","ensemble"):
        print(f"  {name:12} MIREX {sum(kw[name])/n:.4f}  exact {sum(kx[name])/n:.4f}")
    print("  [baseline librosa] MIREX 0.6138 exact 0.5291")
    print(f"\n==== TEMPO ====")
    for name in ("rhythm2013","deepsquare","ens_octcorr"):
        h=th[name]
        if h[0]: print(f"  {name:12} n={h[0]} Acc1 {h[1]/h[0]:.4f}  Acc2 {h[2]/h[0]:.4f}")
    print("  [baseline librosa] Acc1 0.8297 Acc2 0.8690")

if __name__=="__main__":
    main()
