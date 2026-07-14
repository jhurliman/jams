# Vendored from ZFTurbo/Music-Source-Separation-Training (MIT License)
# https://github.com/ZFTurbo/Music-Source-Separation-Training
# commit ccc011abf7f89dd7922bb2888d48493b575c0289 (2026-06-09), models/bs_roformer/
# Author: Roman Solovyev (ZFTurbo) and contributors (upstream: lucidrains/BS-RoFormer, MIT).
# Used with the Kim Mel-Band RoFormer vocals checkpoint (KimberleyJSN/melbandroformer,
# MIT) for the two-pass separation path: vocals first, then SCNet on the instrumental.

from .mel_band_roformer import MelBandRoformer

__all__ = ["MelBandRoformer"]
