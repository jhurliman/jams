"""Runtime configuration, sourced from ``JAMS_*`` env vars (and a local ``.env``)."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JAMS_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Reject uploads larger than this (MB).
    max_upload_mb: int = 100

    # Token for structure analysis (All-In-One on Replicate). Falls back to the
    # standard REPLICATE_API_TOKEN env var if the JAMS_-prefixed one is unset.
    replicate_api_token: str | None = None

    def resolved_replicate_token(self) -> str | None:
        return self.replicate_api_token or os.environ.get("REPLICATE_API_TOKEN")


@lru_cache
def get_settings() -> Settings:
    return Settings()
