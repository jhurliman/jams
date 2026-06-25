"""Console entry point: ``jams`` launches the API server."""

from __future__ import annotations

from jams.config import get_settings


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "jams.api.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
