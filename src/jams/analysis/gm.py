"""General MIDI vocabulary + note assembly shared by the stems pipeline and eval harness.

Pure-python (numpy + pretty_midi only) so it runs in jams' own env — the heavy transcription
lives in the uv workers, but note canonicalisation, beat-grid quantization, and MIDI export are
cheap and belong next to the orchestrator (which also holds the beat grid). Keeping the GM
percussion map here gives ref (eval) and est (pipeline) one shared drum vocabulary.

A note dict is ``{"onset": s, "offset": s, "pitch": midi, "velocity": 1..127}``.
"""

from __future__ import annotations

import bisect

# General MIDI programs (0-indexed) per pitched stem: 33=Electric Bass (finger),
# 0=Acoustic Grand Piano, 85=Lead 6 (voice).
GM_PROGRAM = {"bass": 33, "other": 0, "vocals": 85}

# Bass is written an octave above where it sounds (MIDI/notation convention); transcribers
# detect the sounding pitch. +12 aligns bass MIDI with the written convention — validated on
# Slakh GT for both transcribers (basic-pitch 0.04 -> 0.80, YourMT3+ 0.12 -> 0.85 note-F).
# Applied HERE in the orchestrator, exactly once, whatever the transcriber.
BASS_OCTAVE_SHIFT = 12

# Stems rendered as a single line: keep one note at a time after transcription.
MONOPHONIC_STEMS = frozenset({"bass", "vocals"})


def shift_bass_notes(notes: list[dict]) -> list[dict]:
    """Apply the written-pitch bass convention (+12, capped at 127)."""
    return [{**n, "pitch": min(127, n["pitch"] + BASS_OCTAVE_SHIFT)} for n in notes]


def monophonic_filter(notes: list[dict]) -> list[dict]:
    """Collapse overlapping notes to a single voice, keeping the loudest at each moment.

    Greedy by velocity: accept notes loudest-first, dropping any that overlap an already-
    accepted note. Shared across transcribers so bass/vocals stay clean single lines.
    """
    accepted: list[dict] = []
    for n in sorted(notes, key=lambda x: (-x["velocity"], x["onset"])):
        if any(n["onset"] < a["offset"] and a["onset"] < n["offset"] for a in accepted):
            continue
        accepted.append(n)
    accepted.sort(key=lambda x: x["onset"])
    return accepted

# --- General MIDI percussion (channel 10) -----------------------------------
GM_KICK, GM_SNARE = 36, 38
GM_CLOSED_HAT, GM_PEDAL_HAT, GM_OPEN_HAT = 42, 44, 46
GM_TOM_LOW, GM_TOM_MID, GM_TOM_HIGH = 45, 47, 50
GM_CRASH, GM_RIDE = 49, 51

# The canonical GM drum classes the pipeline emits and the eval scores over.
GM_DRUM_CLASSES = [
    GM_KICK, GM_SNARE, GM_CLOSED_HAT, GM_PEDAL_HAT, GM_OPEN_HAT,
    GM_TOM_LOW, GM_TOM_MID, GM_TOM_HIGH, GM_CRASH, GM_RIDE,
]

# Map E-GMD / Roland reduced drum MIDI pitches (and near-equivalents the OaF model or GT MIDI
# may use) onto the canonical GM notes above, so both sides share one vocabulary.
DRUM_PITCH_CANON = {
    35: GM_KICK, 36: GM_KICK,
    37: GM_SNARE, 38: GM_SNARE, 40: GM_SNARE,  # 37 side-stick -> snare bucket
    42: GM_CLOSED_HAT, 44: GM_PEDAL_HAT, 46: GM_OPEN_HAT, 22: GM_CLOSED_HAT, 26: GM_OPEN_HAT,
    43: GM_TOM_LOW, 45: GM_TOM_LOW, 47: GM_TOM_MID, 48: GM_TOM_MID, 50: GM_TOM_HIGH, 58: GM_TOM_LOW,
    49: GM_CRASH, 52: GM_CRASH, 55: GM_CRASH, 57: GM_CRASH,
    51: GM_RIDE, 53: GM_RIDE, 59: GM_RIDE,
}


def canon_drum_pitch(pitch: int) -> int:
    return DRUM_PITCH_CANON.get(int(pitch), int(pitch))


# 5-class reduction (kick / snare / hats / toms / cymbals) — the standard ADT eval vocabulary
# and exactly what our drum CNN (drum_worker.py) emits. Maps canonical GM pitches onto one
# representative per class: 36 kick, 38 snare, 42 hats, 47 toms, 49 cymbals.
DRUM_5CLASS = {
    GM_KICK: GM_KICK,
    GM_SNARE: GM_SNARE,
    GM_CLOSED_HAT: GM_CLOSED_HAT, GM_PEDAL_HAT: GM_CLOSED_HAT, GM_OPEN_HAT: GM_CLOSED_HAT,
    GM_TOM_LOW: GM_TOM_MID, GM_TOM_MID: GM_TOM_MID, GM_TOM_HIGH: GM_TOM_MID,
    GM_CRASH: GM_CRASH, GM_RIDE: GM_CRASH,
}
GM_DRUM_5CLASSES = [GM_KICK, GM_SNARE, GM_CLOSED_HAT, GM_TOM_MID, GM_CRASH]


def reduce_drum_pitch_5(pitch: int) -> int:
    """Canonical GM pitch -> its 5-class representative (unknown pitches pass through)."""
    return DRUM_5CLASS.get(canon_drum_pitch(pitch), canon_drum_pitch(pitch))


def canon_drum_notes(notes: list[dict]) -> list[dict]:
    return [{**n, "pitch": canon_drum_pitch(n["pitch"])} for n in notes]


# --- beat-grid quantization -------------------------------------------------


def quantize_notes(notes: list[dict], beats: list[float], subdivisions: int = 4) -> list[dict]:
    """Snap note onsets to the nearest beat subdivision; shift offsets to preserve length.

    ``beats`` is the ordered beat times (seconds). Each inter-beat interval is split into
    ``subdivisions`` slots (16th notes for 4/4). Notes keep their duration.
    """
    if not beats or len(beats) < 2:
        return notes
    grid: list[float] = []
    for i in range(len(beats) - 1):
        b0, b1 = beats[i], beats[i + 1]
        for s in range(subdivisions):
            grid.append(b0 + (b1 - b0) * s / subdivisions)
    grid.append(beats[-1])
    out: list[dict] = []
    for n in notes:
        j = bisect.bisect_left(grid, n["onset"])
        cands = [k for k in (j - 1, j) if 0 <= k < len(grid)]
        snapped = min(cands, key=lambda k: abs(grid[k] - n["onset"]))
        dt = grid[snapped] - n["onset"]
        out.append(
            {
                **n,  # keep additive fields (e.g. YourMT3+ per-note "program")
                "onset": grid[snapped],
                "offset": max(grid[snapped] + 1e-3, n["offset"] + dt),
            }
        )
    return out


# --- MIDI export ------------------------------------------------------------


def _instrument(notes: list[dict], program: int, is_drums: bool, name: str):
    import pretty_midi

    inst = pretty_midi.Instrument(program=int(program), is_drum=bool(is_drums), name=name)
    for n in notes:
        inst.notes.append(
            pretty_midi.Note(
                velocity=int(n["velocity"]),
                pitch=int(n["pitch"]),
                start=float(n["onset"]),
                end=float(max(n["onset"] + 1e-3, n["offset"])),
            )
        )
    return inst


def write_stem_midi(notes: list[dict], program: int, is_drums: bool, dest: str) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(
        _instrument(notes, program, is_drums, "drums" if is_drums else "")
    )
    pm.write(str(dest))


def write_combined_midi(transcriptions: list[dict], dest: str) -> None:
    """One Type-1 multitrack MIDI: each stem its own track; drums flagged is_drum (ch. 10)."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    for tr in transcriptions:
        pm.instruments.append(
            _instrument(tr["notes"], tr["gm_program"], tr["is_drums"], tr["stem_type"])
        )
    pm.write(str(dest))
