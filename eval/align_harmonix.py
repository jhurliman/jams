#!/usr/bin/env -S uv run --extra eval
"""Per-track audio<->annotation alignment for the Harmonix structure eval.

Harmonix annotations (beats / downbeats / segments) are public but the original
audio is not — we source audio from YouTube. A YouTube upload relates to the audio
the annotators used in one of three ways:

  case1  same master, only a START OFFSET             -> usable, recover offset b
  case2  uniform SPEED change (sped/slowed reupload)  -> usable after t_audio = a*t_anno + b
  case3  a different EDIT (cuts / rearrangement)      -> NOT usable, down-weight / drop

We do not have the original master, so we cannot align audio-to-audio. Instead we
align the YouTube audio to the ANNOTATION EVENT GRID: the onset-strength envelope of
the audio should line up with a template of impulses at the annotated beat times.

Pipeline (all on one fixed time base, hop_length/sr -> ~100 fps):

  1. onset-strength envelope of the audio (librosa.onset.onset_strength)
  2. annotation beat template: impulses at each beat (downbeat weight 2, beat weight 1)
     then Gaussian-smoothed (~40 ms). Downbeat weighting reduces beat-period aliasing.
  3. stage 1 -- OFFSET b: cross-correlate envelope vs template over a bounded lag (+-30 s);
     global argmax = b. peak_prominence = peak / next-highest local peak.
  4. stage 2 -- AFFINE a: small grid a in [0.94, 1.06] x offset near b, maximizing the
     onset strength sampled at the mapped beat times. Keep best (a, b).
  5. CONFIDENCE in [0,1]: with best (a, b) applied, for each beat find the nearest local
     maximum of the onset envelope; confidence = onset-weighted fraction of beats whose
     nearest local max is within +-50 ms. Model-independent -> this is the per-track weight.
  6. CLASSIFY: case1 if conf>=THR and |a-1|<A_TOL; case2 if conf>=THR and |a-1|>=A_TOL;
     else case3 (unusable).

Output: eval/data/harmonix/alignment.jsonl, one row per track:
  {stem, a, b, confidence, peak_prominence, nieto_score, klass}

Run the full set later with:
    uv run --extra eval eval/align_harmonix.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------- #
# Fixed time base and method constants.
SR = 22050
HOP = 220  # 22050 / 220 ~= 100.2 fps
FPS = SR / HOP

MAX_LAG_S = 30.0  # +-30 s offset search
TEMPLATE_SIGMA_S = 0.040  # ~40 ms Gaussian smoothing of beat template
DOWNBEAT_WEIGHT = 2.0
BEAT_WEIGHT = 1.0

A_GRID = np.arange(0.94, 1.06 + 1e-9, 0.002)  # tempo-scale search
B_REFINE_S = 0.5  # +- window (s) around stage-1 b in the affine refine
B_REFINE_STEP_S = 0.010  # 10 ms

MATCH_TOL_S = 0.050  # +-50 ms beat<->onset-peak match window for confidence

# Classification thresholds (tuned on the validation tracks; see report).
CONF_THR = 0.45
A_TOL = 0.01

# ---------------------------------------------------------------------------- #
# Paths.
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "harmonix"
AUDIO_DIR = DATA / "audio"
DATASET = DATA / "harmonixset" / "dataset"
BEATS_DIR = DATASET / "beats_and_downbeats"
SEGMENTS_DIR = DATASET / "segments"
SCORES_CSV = DATASET / "youtube_alignment_scores.csv"
DEFAULT_OUT = DATA / "alignment.jsonl"
DEFAULT_VIZ_DIR = DATA / "alignment_viz"


@dataclass
class AlignResult:
    stem: str
    a: float
    b: float
    confidence: float
    peak_prominence: float
    nieto_score: float | None
    klass: str


def load_beats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (beat_times, weights). Downbeat (beat_in_bar == 1) gets DOWNBEAT_WEIGHT."""
    times: list[float] = []
    weights: list[float] = []
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            t = float(parts[0])
            beat_in_bar = int(float(parts[1]))
            times.append(t)
            weights.append(DOWNBEAT_WEIGHT if beat_in_bar == 1 else BEAT_WEIGHT)
    return np.asarray(times, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def load_segments(path: Path) -> list[tuple[float, str]]:
    """Return [(time, label), ...] for the segment boundaries, or [] if absent."""
    if not path.exists():
        return []
    out: list[tuple[float, str]] = []
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            out.append((float(parts[0]), parts[1]))
    return out


def build_template(beat_times: np.ndarray, weights: np.ndarray, n_frames: int) -> np.ndarray:
    """Impulses at beat times (weighted), Gaussian-smoothed, on the FPS time base."""
    tmpl = np.zeros(n_frames, dtype=np.float64)
    frames = np.round(beat_times * FPS).astype(int)
    inside = (frames >= 0) & (frames < n_frames)
    np.add.at(tmpl, frames[inside], weights[inside])
    sigma_frames = TEMPLATE_SIGMA_S * FPS
    return gaussian_filter1d(tmpl, sigma_frames, mode="constant")


# Number of top stage-1 offset candidates handed to the affine refine. A single global
# argmax of the cross-correlation can land on a spurious lag for onset-dense tracks whose
# beat template spans the whole song; carrying several candidates and letting the
# beat-sampled objective pick the winner avoids that trap.
N_OFFSET_CANDIDATES = 6
CANDIDATE_SEP_S = 2.0  # min separation between retained candidate offsets
# Dense band of small offsets always handed to the affine refine (spaced <= 2*B_REFINE_S so
# the per-seed +-B_REFINE_S windows tile continuously over +-4 s).
SMALL_OFFSET_SEEDS = [round(x, 3) for x in np.arange(-4.0, 4.0 + 1e-9, 1.0)]
# Small-offset prior. A tiny start offset is far more likely a priori than a multi-bar shift,
# so when scoring offset candidates we subtract a gentle penalty proportional to |offset|.
# A larger offset is only chosen if its confidence beats a smaller one by more than the
# penalty difference -- i.e. it must earn the extra distance with materially better alignment.
# At ~0.004 / s, a 25 s offset must win by >0.10 confidence over a near-zero one; a genuine
# long intro (whose only candidates are all large, e.g. 0001_12step) is unaffected.
OFFSET_PENALTY_PER_S = 0.004
# Among offsets whose penalized score ties, prefer the one with the strongest onset evidence.
SCORE_TIE_MARGIN = 0.02


def stage1_offsets(env: np.ndarray, tmpl: np.ndarray) -> tuple[list[float], float]:
    """Cross-correlate env vs template over +-MAX_LAG_S.

    Returns (candidate_offsets_seconds, peak_prominence). The first candidate is the global
    argmax; the rest are the next-best well-separated local maxima.

    Lag convention: positive lag means the audio onset envelope is DELAYED relative to the
    annotation grid, i.e. t_audio = t_anno + b with b > 0 (a start offset in the upload).
    We normalize the cross-correlation by a sliding window of the audio's local energy so the
    score reflects shape agreement, not raw onset density at a given lag.
    """
    n = len(env)
    max_lag = int(round(MAX_LAG_S * FPS))
    a = env - env.mean()
    t = tmpl - tmpl.mean()
    full = np.correlate(a, t, mode="full")  # length 2n-1, zero-lag at index n-1
    zero = n - 1
    lags = np.arange(-max_lag, max_lag + 1)
    idx = zero + lags
    valid = (idx >= 0) & (idx < len(full))
    lags = lags[valid]
    scores = full[idx[valid]].astype(np.float64)

    best_i = int(np.argmax(scores))
    best_lag = int(lags[best_i])

    # Prominence: global peak / next-highest peak >= 1 s away (low -> aliasing/ambiguity).
    sep1 = int(round(1.0 * FPS))
    far = np.abs(lags - best_lag) >= sep1
    if far.any():
        runner_up = float(scores[far].max())
        prom = float(scores[best_i] / runner_up) if runner_up > 1e-9 else float("inf")
    else:
        prom = float("inf")

    # Greedily collect the top N well-separated offsets as affine seeds.
    sep = int(round(CANDIDATE_SEP_S * FPS))
    order = np.argsort(scores)[::-1]
    chosen: list[int] = []
    for i in order:
        lag = int(lags[i])
        if all(abs(lag - c) >= sep for c in chosen):
            chosen.append(lag)
        if len(chosen) >= N_OFFSET_CANDIDATES:
            break
    candidates = [c / FPS for c in chosen]
    return candidates, prom


def _refine_one_seed(
    env: np.ndarray, scaled_cache: dict[float, np.ndarray], beat_times: np.ndarray,
    weights: np.ndarray, seed: float,
) -> tuple[float, float, float]:
    """For one offset seed, find (a, b) near it maximizing weighted onset strength at beats.

    Returns (a, b, raw_score). The fine raw-sum objective gives sub-beat precision; the
    caller selects between seeds by chance-corrected confidence, not by this raw score.
    """
    b_offsets = np.arange(-B_REFINE_S, B_REFINE_S + 1e-9, B_REFINE_STEP_S)
    n = len(env)
    best_score = -np.inf
    best_a, best_b = 1.0, seed
    for a in A_GRID:
        scaled = scaled_cache.get(a)
        if scaled is None:
            scaled = a * beat_times
            scaled_cache[a] = scaled
        for db in b_offsets:
            b = seed + db
            frames = np.round((scaled + b) * FPS).astype(int)
            inside = (frames >= 0) & (frames < n)
            if not inside.any():
                continue
            score = float((env[frames[inside]] * weights[inside]).sum())
            if score > best_score:
                best_score = score
                best_a, best_b = float(a), float(b)
    return best_a, best_b, best_score


def affine_refine(
    env: np.ndarray, beat_times: np.ndarray, weights: np.ndarray, b_seeds: list[float]
) -> tuple[float, float]:
    """Refine (a, b) around each stage-1 offset seed, then pick the best seed by confidence.

    Within a seed we maximize raw weighted onset strength (precise sub-beat localization).
    Across seeds we choose by chance-corrected confidence -- this avoids beat-period aliasing,
    where an offset shifted by a whole number of beats scores nearly as high on the raw sum
    but is structurally wrong. The first seed (cross-correlation global argmax) wins ties.
    """
    if not b_seeds:
        return 1.0, 0.0
    scaled_cache: dict[float, np.ndarray] = {}
    cand: list[tuple[float, float, float, float]] = []  # (a, b, raw, conf)
    for seed in b_seeds:
        a, b, raw = _refine_one_seed(env, scaled_cache, beat_times, weights, seed)
        conf = compute_confidence(env, beat_times, weights, a, b)
        cand.append((a, b, raw, conf))

    # Score each candidate by chance-corrected confidence (robust to onset density) minus a
    # gentle small-offset prior. This breaks beat-period aliasing toward the true grid: a
    # bar-shifted copy with marginally higher confidence loses to a smaller offset, while a
    # genuine long intro (whose candidates are all large) is unaffected. Among penalized-score
    # ties we prefer the smallest |offset|, then the strongest onset evidence.
    scored = [(a, b, raw, conf - OFFSET_PENALTY_PER_S * abs(b)) for (a, b, raw, conf) in cand]
    best_score = max(s[3] for s in scored)
    near = [s for s in scored if s[3] >= best_score - SCORE_TIE_MARGIN]
    a, b, _, _ = min(near, key=lambda s: (abs(s[1]), -s[2]))
    return a, b


def _onset_peak_times(env: np.ndarray) -> np.ndarray:
    """Times (s) of local maxima of the onset envelope above the noise floor."""
    if len(env) < 3:
        return np.empty(0)
    is_peak = (env[1:-1] >= env[:-2]) & (env[1:-1] > env[2:])
    peak_frames = np.nonzero(is_peak)[0] + 1
    if peak_frames.size == 0:
        return np.empty(0)
    thresh = float(np.median(env))  # drop trivially small peaks (noise floor)
    peak_frames = peak_frames[env[peak_frames] > thresh]
    return peak_frames / FPS


def _match_fraction(
    peak_times: np.ndarray, beat_times: np.ndarray, weights: np.ndarray, dur: float,
    a: float, b: float,
) -> float:
    """Onset-weighted fraction of beats with a peak within +-MATCH_TOL_S of t = a*beat + b."""
    if peak_times.size == 0:
        return 0.0
    mapped = a * beat_times + b
    inside = (mapped >= 0) & (mapped <= dur)
    mapped = mapped[inside]
    w = weights[inside]
    if mapped.size == 0:
        return 0.0
    pos = np.clip(np.searchsorted(peak_times, mapped), 1, len(peak_times) - 1)
    left = peak_times[pos - 1]
    right = peak_times[pos]
    nearest = np.where(np.abs(mapped - left) <= np.abs(mapped - right), left, right)
    matched = np.abs(mapped - nearest) <= MATCH_TOL_S
    total_w = float(w.sum())
    if total_w <= 0:
        return 0.0
    return float(w[matched].sum() / total_w)


# Offsets (s) used to estimate the chance / null match fraction. Beats shifted by these
# large amounts no longer correspond to the audio, so the match fraction there reflects how
# easily the grid matches by accident (high for onset-dense rap/EDM where the grid is
# uninformative). Confidence is the true alignment's LIFT above this null.
NULL_OFFSETS_S = np.array([-23.0, -17.0, -11.0, 11.0, 17.0, 23.0])


def _chance_corrected(
    peak_times: np.ndarray, beat_times: np.ndarray, weights: np.ndarray, dur: float,
    a: float, b: float,
) -> float:
    """(raw - null) / (1 - null): match-fraction lift above the by-accident rate."""
    if beat_times.size == 0:
        return 0.0
    raw = _match_fraction(peak_times, beat_times, weights, dur, a, b)
    null_vals = [
        _match_fraction(peak_times, beat_times, weights, dur, a, b + off)
        for off in NULL_OFFSETS_S
    ]
    null = float(np.median(null_vals))
    if null >= 1.0:
        return 0.0
    return float(np.clip((raw - null) / (1.0 - null), 0.0, 1.0))


def compute_confidence(
    env: np.ndarray, beat_times: np.ndarray, weights: np.ndarray, a: float, b: float
) -> float:
    """Chance-corrected alignment confidence in [0, 1].

    For both the full beat grid and the (sparser, more structural) downbeat grid we compute
    a chance-corrected match-fraction lift -- how much better beats land on onset peaks at
    the estimated (a, b) than at several large bogus offsets. The two are averaged.

    Splitting out downbeats matters for onset-dense tracks (rap / busy pop) where almost any
    offset matches some onset: the dense all-beat fraction saturates, but the sparse downbeat
    grid still separates a true alignment from a different edit. A well-aligned track scores
    high on both; a different edit (or an uninformative grid) collapses to ~0.
    """
    peak_times = _onset_peak_times(env)
    dur = len(env) / FPS

    conf_all = _chance_corrected(peak_times, beat_times, weights, dur, a, b)

    db_mask = weights >= DOWNBEAT_WEIGHT
    db_times = beat_times[db_mask]
    db_w = np.ones_like(db_times)
    conf_db = _chance_corrected(peak_times, db_times, db_w, dur, a, b)

    return float(0.5 * (conf_all + conf_db))


def classify(confidence: float, a: float) -> str:
    if confidence >= CONF_THR and abs(a - 1.0) < A_TOL:
        return "case1"
    if confidence >= CONF_THR and abs(a - 1.0) >= A_TOL:
        return "case2"
    return "case3"


def align_track(stem: str, audio_path: Path, nieto: float | None) -> AlignResult:
    # m4a decoding falls back to audioread, which emits deprecation/format warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    env = np.asarray(env, dtype=np.float64)
    n_frames = len(env)

    beat_times, weights = load_beats(BEATS_DIR / f"{stem}.txt")
    tmpl = build_template(beat_times, weights, n_frames)

    seeds, prom = stage1_offsets(env, tmpl)
    # Always evaluate a dense band of SMALL offsets too: the most likely case is a tiny start
    # offset, but the cross-correlation can rank a bar-shifted alias above the true near-zero
    # offset for periodic songs, so the true small offset may be absent from the seeds. Adding
    # this band (spaced to overlap the +-B_REFINE_S window for continuous coverage) guarantees
    # the small-offset prior in affine_refine can actually choose a near-zero offset when one
    # aligns as well as the larger alias.
    seeds = list(seeds) + SMALL_OFFSET_SEEDS
    a, b = affine_refine(env, beat_times, weights, seeds)
    conf = compute_confidence(env, beat_times, weights, a, b)
    klass = classify(conf, a)

    return AlignResult(
        stem=stem,
        a=round(a, 4),
        b=round(b, 4),
        confidence=round(conf, 4),
        peak_prominence=round(prom, 4) if np.isfinite(prom) else 999.0,
        nieto_score=round(nieto, 4) if nieto is not None else None,
        klass=klass,
    )


def _worker(args: tuple[str, str, float | None]) -> AlignResult:
    stem, audio_path, nieto = args
    return align_track(stem, Path(audio_path), nieto)


# Distinct, repeating palette for segment labels (verse / chorus / ...).
_SEG_COLORS = [
    "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
    "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd",
]


def render_alignment(res: AlignResult, audio_path: Path, out_dir: Path) -> Path:
    """Draw the audio waveform + onset envelope with the affine-mapped annotations on top.

    All annotation times t_anno are mapped to audio time t_audio = a*t_anno + b before
    plotting, so a good alignment shows beats sitting on onset peaks and segment boundaries
    falling on musical changes. Requires matplotlib (run with `--with matplotlib`).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    env = np.asarray(librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP), dtype=np.float64)
    dur = len(y) / SR
    env_t = np.arange(len(env)) / FPS
    env_n = env / (env.max() + 1e-9)

    beat_times, weights = load_beats(BEATS_DIR / f"{res.stem}.txt")
    segments = load_segments(SEGMENTS_DIR / f"{res.stem}.txt")
    a, b = res.a, res.b

    fig, (ax_w, ax_o) = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

    # --- top: waveform with segment regions + boundaries ---
    wav_t = np.arange(len(y)) / SR
    ax_w.plot(wav_t, y, lw=0.4, color="#333333", alpha=0.7)
    seg_labels: dict[str, str] = {}
    for i, (t0, label) in enumerate(segments):
        x0 = a * t0 + b
        x1 = a * segments[i + 1][0] + b if i + 1 < len(segments) else dur
        color = _SEG_COLORS[hash(label) % len(_SEG_COLORS)]
        seg_labels[label] = color
        ax_w.axvspan(x0, x1, color=color, alpha=0.15)
        ax_w.axvline(x0, color=color, lw=1.2)
        ax_w.text(x0 + 0.2, 0.92, label, transform=ax_w.get_xaxis_transform(),
                  fontsize=8, color=color, rotation=90, va="top")
    ax_w.set_ylabel("waveform")
    ax_w.set_title(
        f"{res.stem}   a={a:.3f}  b={b:+.2f}s  conf={res.confidence:.3f}  "
        f"prom={res.peak_prominence:.2f}  nieto={res.nieto_score}  [{res.klass}]"
    )
    if seg_labels:
        ax_w.legend(handles=[Patch(color=c, label=lab) for lab, c in seg_labels.items()],
                    ncol=min(len(seg_labels), 8), fontsize=7, loc="lower center",
                    bbox_to_anchor=(0.5, 1.18))

    # --- bottom: onset envelope with mapped beats / downbeats ---
    ax_o.plot(env_t, env_n, lw=0.6, color="#1f77b4", label="onset strength")
    db_mask = weights >= DOWNBEAT_WEIGHT
    for t in a * beat_times[~db_mask] + b:
        ax_o.axvline(t, color="#888888", lw=0.5, alpha=0.6)
    for t in a * beat_times[db_mask] + b:
        ax_o.axvline(t, color="#d62728", lw=1.0, alpha=0.85)
    ax_o.set_ylabel("onset (mapped beats)")
    ax_o.set_xlabel("audio time (s)")
    ax_o.set_xlim(0, dur)
    ax_o.legend(
        handles=[
            plt.Line2D([], [], color="#1f77b4", label="onset strength"),
            plt.Line2D([], [], color="#d62728", label="downbeat"),
            plt.Line2D([], [], color="#888888", label="beat"),
        ],
        fontsize=7, loc="upper right",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{res.stem}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def load_nieto_scores() -> dict[str, float]:
    scores: dict[str, float] = {}
    if not SCORES_CSV.exists():
        return scores
    with SCORES_CSV.open() as fh:
        for row in csv.DictReader(fh):
            try:
                scores[row["File"]] = float(row["score"])
            except (KeyError, ValueError):
                continue
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N tracks")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output jsonl path")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated stems to process (substring match), e.g. 0001,0005")
    ap.add_argument("--viz", action="store_true",
                    help="also render a PNG per track (waveform + mapped annotations); "
                         "needs matplotlib (run with `--with matplotlib`)")
    ap.add_argument("--viz-dir", type=Path, default=DEFAULT_VIZ_DIR, help="PNG output dir")
    args = ap.parse_args()

    nieto = load_nieto_scores()

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    stems = []
    for audio in sorted(AUDIO_DIR.glob("*.m4a")):
        stem = audio.stem
        if not (BEATS_DIR / f"{stem}.txt").exists():
            continue
        if only is not None and not any(o in stem for o in only):
            continue
        stems.append((stem, str(audio), nieto.get(stem)))
    if args.limit is not None:
        stems = stems[: args.limit]

    if not stems:
        print("No tracks found (need audio + beats).", file=sys.stderr)
        return 1

    print(f"Aligning {len(stems)} tracks (sr={SR}, hop={HOP}, {FPS:.1f} fps)...\n")

    results: list[AlignResult] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_worker, s): s[0] for s in stems}
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for s in stems:
            results.append(_worker(s))

    results.sort(key=lambda r: r.stem)

    # Table.
    hdr = f"{'stem':<24} {'a':>7} {'b':>9} {'conf':>6} {'prom':>7} {'nieto':>7}  class"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ni = f"{r.nieto_score:.3f}" if r.nieto_score is not None else "   -  "
        print(
            f"{r.stem:<24} {r.a:>7.3f} {r.b:>9.3f} {r.confidence:>6.3f} "
            f"{r.peak_prominence:>7.3f} {ni:>7}  {r.klass}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")
    print(f"\nWrote {len(results)} rows -> {args.out}")

    if args.viz:
        audio_by_stem = {stem: Path(p) for stem, p, _ in stems}
        print(f"Rendering {len(results)} alignment plots -> {args.viz_dir}")
        for r in results:
            out = render_alignment(r, audio_by_stem[r.stem], args.viz_dir)
            print(f"  {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
