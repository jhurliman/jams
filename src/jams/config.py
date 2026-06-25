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

    # --- Song structure (All-In-One) ---------------------------------------
    # "local": run All-In-One on-device. The worker is a self-contained uv script
    # (all-in-one-mps), so it can't share jams' Python 3.14 env (no torch wheel) —
    # jams launches it via `uv run --script` as a subprocess. "replicate": call
    # the hosted model (needs a token).
    structure_backend: Literal["local", "replicate"] = "local"
    # Command used to launch the self-contained worker script. Override if `uv`
    # isn't on PATH (e.g. an absolute path to the uv binary).
    structure_uv: str = "uv"
    # All-In-One model: "harmonix-all" (8-fold ensemble) or "harmonix-foldN".
    structure_model: str = "harmonix-all"

    # Token for structure analysis on Replicate. Falls back to the standard
    # REPLICATE_API_TOKEN env var if the JAMS_-prefixed one is unset.
    replicate_api_token: str | None = None

    def resolved_replicate_token(self) -> str | None:
        return self.replicate_api_token or os.environ.get("REPLICATE_API_TOKEN")


@lru_cache
def get_settings() -> Settings:
    return Settings()
