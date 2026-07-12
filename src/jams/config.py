"""Runtime configuration, sourced from ``JAMS_*`` env vars (and a local ``.env``)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JAMS_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Reject uploads larger than this (MB).
    max_upload_mb: int = 100

    # --- Key detection ------------------------------------------------------
    # Fuse edma with Deezer's S-KEY model (uv worker, src/jams/data/skey_worker.py) —
    # the honest-protocol default (GiantSteps Key weighted 0.812 / exact 0.757, heads
    # trained only on GiantSteps-MTG). Disabling falls back to the LEGACY mode-refinement
    # model, which was trained on GiantSteps Key itself (contaminated — don't quote its
    # numbers against the literature).
    key_fusion: bool = True

    # --- Song structure (All-In-One) ---------------------------------------
    # "local": run All-In-One on-device. The worker is a self-contained uv script
    # (all-in-one-mps), so it can't share jams' Python 3.14 env (no torch wheel) —
    # jams launches it via `uv run --script` as a subprocess. "replicate": call
    # the hosted model (needs a token).
    structure_backend: Literal["local", "replicate"] = "local"
    # Command used to launch the self-contained worker script. Override if `uv`
    # isn't on PATH (e.g. an absolute path to the uv binary).
    structure_uv: str = "uv"
    # All-In-One model. "all-all" = 8-fold Pop+EDM ensemble (best general accuracy,
    # esp. on electronic music); "harmonix-all" = Pop-only 8-fold ensemble; or a
    # single fold "all-foldN"/"harmonix-foldN" (used for held-out cross-validation).
    structure_model: str = "all-all"

    # Token for structure analysis on Replicate. Falls back to the standard
    # REPLICATE_API_TOKEN env var if the JAMS_-prefixed one is unset.
    replicate_api_token: str | None = None

    # --- Stem separation + MIDI transcription ------------------------------
    # Demucs + basic-pitch + the ADTOF drum model run in self-contained uv workers
    # (src/jams/data/stems_worker.py), same subprocess pattern as structure.
    stems_uv: str = "uv"
    # Separation model. "scnet_xl_ihf" (default) = vendored SCNet XL IHF, the Slakh-test
    # A/B winner (SI-SDR drums 14.3 vs htdemucs 11.6; bass note-F 0.596 -> 0.645).
    # "htdemucs" / "htdemucs_ft" select Demucs (faster / legacy comparison).
    stems_model: str = "scnet_xl_ihf"
    # Pitched-stem transcriber. "yourmt3" (default) = YourMT3+ via mt3-infer — Slakh-test
    # oracle note-F bass 0.849 / other 0.849 vs basic-pitch 0.789 / 0.490. Needs git-lfs on
    # first run (checkpoint clone). "basic-pitch" = lighter/faster, no git-lfs.
    stems_transcriber: Literal["yourmt3", "basic-pitch"] = "yourmt3"
    # Snap transcribed note onsets to jams' resolved beat grid when available.
    stems_quantize: bool = True
    # Directory the worker writes stems + MIDI into (served to the webapp). Per-track
    # subdirs are created under here; a temp dir is used when unset.
    stems_out_dir: str | None = None

    def resolved_replicate_token(self) -> str | None:
        return self.replicate_api_token or os.environ.get("REPLICATE_API_TOKEN")


@lru_cache
def get_settings() -> Settings:
    return Settings()
