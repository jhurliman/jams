"""Per-track PROCEDURAL Surge patch randomization — the anti-monotimbre lever.

Every melodic layer draws a fresh timbre from Surge's oscillator engines (subtractive Classic,
Wavetable, FM2/FM3, Window, Modern, Twist, String), filter models, waveshapers, unison and
envelopes — so one plugin yields hundreds of distinct spectra for ~zero cost. This is the #1
synth->real transfer risk (a single-patch corpus overfits SCNet to one spectrum), so bass and
synth voices are randomized within musically-sensible per-role ranges. Returns (cfg, descriptor)
where the descriptor is recorded per track for diversity auditing.
"""

from __future__ import annotations

from .surge import P

# Role-appropriate oscillator engines (span subtractive + wavetable + FM + physical + macro).
_OSC = {
    "sub": ["Sine", "Classic"],
    "reese": ["Classic", "Wavetable", "FM3", "Modern"],
    "growl": ["Classic", "Wavetable", "FM3", "Modern"],
    "wobble": ["Classic", "Wavetable", "Modern"],
    "jumpup_bounce": ["Classic", "Modern", "FM2"],
    "foghorn": ["Classic", "Wavetable", "Window"],
    "pad": ["Classic", "Wavetable", "Window", "Modern"],
    "reese_pad": ["Classic", "Wavetable", "FM3"],
    "stab": ["Classic", "Wavetable", "Twist", "Modern"],
    "lead": ["Classic", "Twist", "FM2", "Wavetable"],
    "pluck": ["Classic", "Wavetable", "Twist"],
    "atmos": ["Wavetable", "Window", "String", "Twist"],
    "rhodes": ["Sine", "Wavetable"],
}
_LP = ["LP 12 dB", "LP 24 dB", "LP OB-Xd 12 dB", "LP OB-Xd 24 dB", "LP K35",
       "LP Diode Ladder", "LP Vintage Ladder"]
_BP = ["BP 12 dB", "BP 24 dB"]
_SHAPERS = ["Soft", "OJD", "Medium", "Digital", "Fuzz Soft Clip", "Single Fold",
            "Soft Harmonic 3", "West Coast Fold"]

# amp ADSR ranges (ms unless noted) and unison per role.
_ENV = {
    "sub": (2, 2, 8, 80, 0.6, 0.85),
    "reese": (2, 200, 400, 60, 150, 0.7),
    "growl": (2, 150, 350, 55, 120, 0.7),
    "wobble": (5, 300, 500, 70, 200, 0.75),
    "jumpup_bounce": (1, 80, 180, 15, 40, 0.3),
    "foghorn": (8, 400, 800, 90, 300, 0.8),
    "pad": (200, 400, 700, 70, 400, 0.85),
    "reese_pad": (150, 350, 650, 65, 350, 0.8),
    "stab": (2, 120, 240, 20, 40, 0.3),
    "lead": (3, 150, 300, 50, 90, 0.7),
    "pluck": (1, 90, 160, 5, 12, 0.1),
    "atmos": (400, 500, 1000, 60, 600, 0.75),
    "rhodes": (2, 300, 500, 30, 200, 0.4),
}


def _apply(s, role: str, rng, cutoff_lo: float, cutoff_hi: float,
           allow_bp: bool = False) -> dict:
    osc = _OSC.get(role, ["Classic"])[int(rng.integers(len(_OSC.get(role, ["Classic"]))))]
    s.set_choice(P["o1_type"], osc)
    if osc in ("FM2", "FM3", "Twist", "String"):
        s.set(P["o1_shape"], float(rng.uniform(0.35, 0.9)))   # FM depth / macro: keep it audible
    elif osc in ("Classic", "Modern", "Wavetable", "Window"):
        s.set(P["o1_shape"], float(rng.uniform(0.1, 0.8)))
    voices = int(rng.integers(1, 8)) if role not in ("sub", "rhodes") else 1
    detune = float(rng.uniform(4, 38)) if voices > 1 else 0.0
    if voices > 1:
        s.set_num(P["o1_uni_voices"], voices)
        s.set_num(P["o1_uni_detune"], detune)

    ftype = _BP[int(rng.integers(len(_BP)))] if (allow_bp and rng.random() < 0.25) \
        else _LP[int(rng.integers(len(_LP)))]
    cutoff = float(rng.uniform(cutoff_lo, cutoff_hi))
    reso = float(rng.uniform(0, 35))
    s.set_choice(P["f1_type"], ftype)
    s.set_num(P["f1_cutoff"], cutoff)
    s.set_num(P["f1_reso"], reso, lo=0.0, hi=0.5)

    shaper = None
    if role in ("reese", "growl", "wobble", "foghorn") or rng.random() < 0.2:
        shaper = _SHAPERS[int(rng.integers(len(_SHAPERS)))]
        s.set_choice(P["waveshaper_type"], shaper)
        s.set_num(P["waveshaper_drive"], rng.integers(2, 12))
        s.set(P["filter_balance"], 1.0)

    a, d, dc, s_hi, r_hi, vca = _ENV[role]
    amp_a = float(rng.uniform(max(1, a * 0.5), a * 1.5 + 1))
    amp_d = float(rng.uniform(d * 0.6, d * 1.4 + 1))
    amp_s = float(rng.uniform(s_hi * 0.6, min(0.95, s_hi))) if s_hi <= 1 \
        else float(rng.uniform(dc * 0.4, dc)) / 100.0
    amp_r = float(rng.uniform(r_hi * 0.6, r_hi * 1.4 + 1))
    s.set_num(P["amp_a"], amp_a)
    s.set_num(P["amp_d"], amp_d)
    s.set_num(P["amp_s"], amp_s, lo=0.0, hi=1.0)
    s.set_num(P["amp_r"], amp_r)
    s.set(P["vca_gain"], float(vca) if vca <= 1 else 0.5)
    return {"osc": osc, "filter": ftype, "cutoff": round(cutoff), "reso": round(reso, 1),
            "unison": voices, "waveshaper": shaper}


def rand_bass_cfg(family: str, rng):
    ranges = {
        "sub": (120, 220), "reese": (600, 1600), "growl": (500, 1400),
        "wobble": (900, 2400), "jumpup_bounce": (900, 1800), "foghorn": (400, 1100),
    }
    lo, hi = ranges.get(family, (700, 1600))
    desc: dict = {}

    def cfg(s):
        if family == "sub":
            s.set_choice(P["o1_type"], "Sine")
            s.set_num(P["amp_a"], 2)
            s.set_num(P["amp_r"], float(rng.uniform(80, 200)))
            s.set(P["vca_gain"], 0.62)
            desc.update({"osc": "Sine", "filter": "none"})
        else:
            desc.update(_apply(s, family, rng, lo, hi, allow_bp=(family in ("reese", "growl"))))
    return cfg, desc


def rand_synth_cfg(role: str, rng):
    ranges = {
        "pad": (1400, 3000), "reese_pad": (1200, 2600), "stab": (2500, 6000),
        "lead": (3000, 7500), "pluck": (2800, 6500), "atmos": (900, 2200),
        "rhodes": (1800, 4000),
    }
    lo, hi = ranges.get(role, (2000, 5000))
    desc: dict = {}

    def cfg(s):
        desc.update(_apply(s, role, rng, lo, hi, allow_bp=(role in ("stab", "lead"))))
    return cfg, desc
