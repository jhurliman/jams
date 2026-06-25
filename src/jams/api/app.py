"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from jams import __version__
from jams.api.routes import router


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = FastAPI(
        title="jams",
        version=__version__,
        summary="On-demand music analysis: per-track key, tempo, and structure.",
    )
    app.include_router(router)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
