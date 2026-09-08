"""Synthetic drum & bass stem-corpus generator (ES2).

Offline, ship-clean render pipeline that produces broad D&B multitracks (drums / bass /
other / vocals premaster buses + premaster mix + mastered mix) as fine-tune training data
for SCNet. See ``eval/synth/RUNBOOK.md`` for the toolchain + license ledger and
``paper/EXPERIMENTS.md`` (ES2) for the pre-registered spec this implements.

Engine: DawDreamer (MIT). Synths: Surge XT (GPL + output-grant) + Dexed (GPL, own patches),
driven by procedural parameters (no factory-preset content copied). Drums: our own numpy DSP
layered with real one-shots sliced from E-GMD (CC-BY 4.0, attributed in the dataset card).
"""

from __future__ import annotations

__all__ = ["config", "theory"]
