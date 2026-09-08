"""Sub-style definitions + per-track diversity sampling for the D&B generator.

A ``SubstyleSpec`` captures the characteristic tempo / drum / bass / synth / sidechain / master
distributions of one D&B sub-style. ``sample_track_spec`` draws a fully-resolved ``TrackSpec``
(every pre-registered diversity axis randomized) from a seeded RNG, so the whole corpus is
reproducible and no two tracks are the same template.

Diversity axes (ES2 pre-reg): tempo, key, sub-style/arrangement, drum family + ghost density +
break choice, bass family + layer count, synth layer count, sidechain style, master loudness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import theory

SUBSTYLES_ORDER = ["jumpup", "neurofunk", "liquid", "jungle", "techstep", "dancefloor"]

# Bass design families (implemented in bass.py). "sub" is the mandatory low layer.
BASS_FAMILIES = ["reese", "wobble", "jumpup_bounce", "foghorn", "growl"]

# Synth roles for the ``other`` bus (implemented in synths.py / dexed.py).
SYNTH_ROLES = ["pad", "rhodes", "stab", "lead", "pluck", "atmos", "reese_pad"]


@dataclass(frozen=True)
class SubstyleSpec:
    name: str
    tempo_lo: float
    tempo_hi: float
    tempo_mode: float
    half_time_prob: float
    scale_weights: dict[str, float]
    bass_family_weights: dict[str, float]
    bass_mid_layers: tuple[int, int]          # (min, max) mid-bass layers on top of the sub
    synth_role_weights: dict[str, float]
    other_layers: tuple[int, int]             # (min, max) simultaneous "other" voices
    swing: tuple[float, float]                # 16th swing fraction range
    ghost_density: tuple[float, float]
    break_prob: float                         # chance of a chopped real-break layer
    real_layer_prob: float                    # chance real one-shots layer the kit
    kit_bright: tuple[float, float]           # hat/top brightness scaler
    sidechain_style: str                      # "duck" (tight DnB) | "pump" (housier)
    sidechain_amt: tuple[float, float]        # duck depth 0..1
    sidechain_ms: tuple[float, float]         # release ms
    lufs_lo: float
    lufs_hi: float
    section_templates: list[list[tuple[str, int]]] = field(default_factory=list)


def _tpl(*sections: tuple[str, int]) -> list[tuple[str, int]]:
    return list(sections)


# Compact two-drop templates (bars). Section ORDER + phrase quantization follow the pre-reg;
# per-section bar counts are kept on the short end to bound corpus size to offline disk.
_TWO_DROP = [
    _tpl(("intro", 4), ("build1", 4), ("drop1", 8), ("break", 4),
         ("build2", 4), ("drop2", 8), ("outro", 2)),
    _tpl(("intro", 8), ("build1", 4), ("drop1", 8), ("break", 4),
         ("build2", 4), ("drop2", 8)),
    _tpl(("intro", 4), ("build1", 4), ("drop1", 8), ("break", 8),
         ("build2", 4), ("drop2", 8), ("outro", 2)),
]
_LONG_DROP = [
    _tpl(("intro", 4), ("build1", 4), ("drop1", 16), ("break", 4), ("drop2", 8)),
    _tpl(("intro", 8), ("build1", 4), ("drop1", 8), ("break", 4), ("build2", 4),
         ("drop2", 8), ("outro", 4)),
]


SUBSTYLES: dict[str, SubstyleSpec] = {
    "jumpup": SubstyleSpec(
        name="jumpup", tempo_lo=172, tempo_hi=176, tempo_mode=174.0, half_time_prob=0.0,
        scale_weights={"natural_minor": 3, "harmonic_minor": 2, "phrygian": 1},
        bass_family_weights={"jumpup_bounce": 4, "growl": 2, "reese": 1},
        bass_mid_layers=(1, 2),
        synth_role_weights={"stab": 3, "lead": 3, "pluck": 2, "pad": 1},
        other_layers=(2, 3),
        swing=(0.0, 0.08), ghost_density=(0.2, 0.45), break_prob=0.25, real_layer_prob=0.7,
        kit_bright=(0.9, 1.15), sidechain_style="duck", sidechain_amt=(0.3, 0.6),
        sidechain_ms=(30, 70), lufs_lo=-6.5, lufs_hi=-5.0,
        section_templates=_TWO_DROP + _LONG_DROP),
    "neurofunk": SubstyleSpec(
        name="neurofunk", tempo_lo=170, tempo_hi=176, tempo_mode=174.0, half_time_prob=0.15,
        scale_weights={"natural_minor": 2, "harmonic_minor": 2, "phrygian": 2, "dorian": 1},
        bass_family_weights={"reese": 4, "growl": 3, "wobble": 2, "foghorn": 1},
        bass_mid_layers=(2, 3),
        synth_role_weights={"stab": 3, "atmos": 3, "lead": 1, "pad": 1, "reese_pad": 2},
        other_layers=(2, 4),
        swing=(0.0, 0.04), ghost_density=(0.25, 0.5), break_prob=0.3, real_layer_prob=0.75,
        kit_bright=(0.85, 1.1), sidechain_style="duck", sidechain_amt=(0.35, 0.7),
        sidechain_ms=(20, 55), lufs_lo=-6.5, lufs_hi=-5.0,
        section_templates=_TWO_DROP + _LONG_DROP),
    "liquid": SubstyleSpec(
        name="liquid", tempo_lo=170, tempo_hi=176, tempo_mode=174.0, half_time_prob=0.1,
        scale_weights={"natural_minor": 2, "dorian": 3, "harmonic_minor": 1},
        bass_family_weights={"reese": 2, "wobble": 1, "growl": 1, "jumpup_bounce": 1},
        bass_mid_layers=(1, 2),
        synth_role_weights={"pad": 3, "rhodes": 4, "pluck": 2, "lead": 2, "atmos": 2},
        other_layers=(3, 4),
        swing=(0.04, 0.12), ghost_density=(0.15, 0.35), break_prob=0.2, real_layer_prob=0.8,
        kit_bright=(0.75, 1.0), sidechain_style="duck", sidechain_amt=(0.25, 0.5),
        sidechain_ms=(40, 90), lufs_lo=-8.0, lufs_hi=-6.0,
        section_templates=_TWO_DROP + _LONG_DROP),
    "jungle": SubstyleSpec(
        name="jungle", tempo_lo=160, tempo_hi=174, tempo_mode=168.0, half_time_prob=0.05,
        scale_weights={"natural_minor": 3, "phrygian": 2, "dorian": 1},
        bass_family_weights={"reese": 2, "growl": 1, "wobble": 1},
        bass_mid_layers=(1, 2),
        synth_role_weights={"stab": 4, "pad": 2, "atmos": 2, "rhodes": 1},
        other_layers=(2, 3),
        swing=(0.06, 0.16), ghost_density=(0.4, 0.7), break_prob=0.85, real_layer_prob=0.9,
        kit_bright=(0.9, 1.2), sidechain_style="duck", sidechain_amt=(0.2, 0.45),
        sidechain_ms=(30, 80), lufs_lo=-8.0, lufs_hi=-6.0,
        section_templates=_TWO_DROP),
    "techstep": SubstyleSpec(
        name="techstep", tempo_lo=170, tempo_hi=176, tempo_mode=174.0, half_time_prob=0.1,
        scale_weights={"phrygian": 3, "natural_minor": 2, "harmonic_minor": 1},
        bass_family_weights={"reese": 3, "growl": 2, "foghorn": 2, "wobble": 1},
        bass_mid_layers=(1, 2),
        synth_role_weights={"stab": 3, "atmos": 3, "lead": 1, "pad": 1},
        other_layers=(1, 3),
        swing=(0.0, 0.03), ghost_density=(0.1, 0.3), break_prob=0.3, real_layer_prob=0.7,
        kit_bright=(0.8, 1.05), sidechain_style="duck", sidechain_amt=(0.35, 0.65),
        sidechain_ms=(20, 50), lufs_lo=-7.0, lufs_hi=-5.5,
        section_templates=_TWO_DROP + _LONG_DROP),
    "dancefloor": SubstyleSpec(
        name="dancefloor", tempo_lo=172, tempo_hi=176, tempo_mode=174.0, half_time_prob=0.0,
        scale_weights={"natural_minor": 3, "harmonic_minor": 2},
        bass_family_weights={"reese": 3, "growl": 2, "jumpup_bounce": 2, "wobble": 1},
        bass_mid_layers=(1, 2),
        synth_role_weights={"lead": 4, "stab": 3, "pluck": 2, "pad": 2},
        other_layers=(2, 4),
        swing=(0.0, 0.06), ghost_density=(0.15, 0.4), break_prob=0.25, real_layer_prob=0.7,
        kit_bright=(0.95, 1.2), sidechain_style="pump", sidechain_amt=(0.4, 0.75),
        sidechain_ms=(50, 110), lufs_lo=-6.5, lufs_hi=-5.0,
        section_templates=_TWO_DROP + _LONG_DROP),
}


@dataclass
class TrackSpec:
    seed: int
    substyle: str
    bpm: float
    half_time: bool
    key: theory.Key
    progression: list[int]
    sections: list[tuple[str, int]]
    total_bars: int
    bass_families: list[str]
    synth_roles: list[str]
    swing: float
    ghost_density: float
    use_break: bool
    use_real_layer: bool
    kit_bright: float
    sidechain_style: str
    sidechain_amt: float
    sidechain_ms: float
    lufs_target: float

    def as_dict(self) -> dict:
        return {
            "seed": self.seed, "substyle": self.substyle, "bpm": round(self.bpm, 2),
            "half_time_feel": self.half_time, "key": self.key.name,
            "scale": self.key.scale, "key_root_pc": self.key.root,
            "progression_degrees": self.progression,
            "sections": [{"name": n, "bars": b} for n, b in self.sections],
            "total_bars": self.total_bars,
            "bass_families": self.bass_families, "synth_roles": self.synth_roles,
            "swing": round(self.swing, 3), "ghost_density": round(self.ghost_density, 3),
            "use_break_layer": self.use_break, "use_real_drum_layer": self.use_real_layer,
            "kit_brightness": round(self.kit_bright, 3),
            "sidechain": {"style": self.sidechain_style,
                          "amount": round(self.sidechain_amt, 3),
                          "release_ms": round(self.sidechain_ms, 1)},
            "lufs_target": round(self.lufs_target, 2),
        }


def _weighted_pick(rng: np.random.Generator, weights: dict[str, float]) -> str:
    names = list(weights)
    w = np.array([weights[n] for n in names], dtype=float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def _triangular_bpm(rng: np.random.Generator, spec: SubstyleSpec) -> float:
    return float(rng.triangular(spec.tempo_lo, spec.tempo_mode, spec.tempo_hi))


def _resolve_sections(rng: np.random.Generator, spec: SubstyleSpec) -> list[tuple[str, int]]:
    tpl = spec.section_templates[int(rng.integers(0, len(spec.section_templates)))]
    out = [(name, bars) for name, bars in tpl]
    total = sum(b for _, b in out)
    # Clamp to <= 40 bars so the corpus fits offline disk; shrink drops first if needed.
    while total > 40:
        out = [(n, b // 2 if (n.startswith("drop") and b > 8) else b) for n, b in out]
        new_total = sum(b for _, b in out)
        if new_total == total:
            break
        total = new_total
    return out


def sample_track_spec(seed: int, substyle: str) -> TrackSpec:
    """Draw a fully-resolved, reproducible track spec for a sub-style."""
    rng = np.random.default_rng(seed)
    spec = SUBSTYLES[substyle]

    bpm = _triangular_bpm(rng, spec)
    half_time = bool(rng.random() < spec.half_time_prob)
    key = theory.pick_key(rng, spec.scale_weights)
    progression = theory.pick_progression(rng)
    sections = _resolve_sections(rng, spec)
    total_bars = sum(b for _, b in sections)

    # Bass: always a sub layer, plus 1-3 mid-bass design layers (distinct families).
    n_mid = int(rng.integers(spec.bass_mid_layers[0], spec.bass_mid_layers[1] + 1))
    mids: list[str] = []
    for _ in range(n_mid):
        fam = _weighted_pick(rng, spec.bass_family_weights)
        if fam not in mids:
            mids.append(fam)
    if not mids:
        mids = [_weighted_pick(rng, spec.bass_family_weights)]
    bass_families = ["sub", *mids]

    # Synths: sample distinct "other" voices.
    n_other = int(rng.integers(spec.other_layers[0], spec.other_layers[1] + 1))
    roles: list[str] = []
    for _ in range(n_other * 2):
        r = _weighted_pick(rng, spec.synth_role_weights)
        if r not in roles:
            roles.append(r)
        if len(roles) >= n_other:
            break
    if not roles:
        roles = [_weighted_pick(rng, spec.synth_role_weights)]
    # Guarantee a sustained bed so the "other" bus is populated in intro/break (probe gap fix).
    if not any(r in ("pad", "rhodes", "atmos", "reese_pad") for r in roles):
        roles.append("pad" if substyle != "liquid" else "rhodes")

    return TrackSpec(
        seed=seed, substyle=substyle, bpm=bpm, half_time=half_time, key=key,
        progression=progression, sections=sections, total_bars=total_bars,
        bass_families=bass_families, synth_roles=roles,
        swing=float(rng.uniform(*spec.swing)),
        ghost_density=float(rng.uniform(*spec.ghost_density)),
        use_break=bool(rng.random() < spec.break_prob),
        use_real_layer=bool(rng.random() < spec.real_layer_prob),
        kit_bright=float(rng.uniform(*spec.kit_bright)),
        sidechain_style=spec.sidechain_style,
        sidechain_amt=float(rng.uniform(*spec.sidechain_amt)),
        sidechain_ms=float(rng.uniform(*spec.sidechain_ms)),
        lufs_target=float(rng.uniform(spec.lufs_lo, spec.lufs_hi)),
    )
