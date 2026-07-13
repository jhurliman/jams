"""API routes — analyze a track by upload or by server-side path."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from jams.analysis import analyze_track
from jams.analysis.audio import SUPPORTED_FORMATS
from jams.api.jobs import get_registry
from jams.config import get_settings
from jams.jams_export import to_jams
from jams.models import AnalyzePathRequest, AnalyzeResponse, ResegmentRequest, ResegmentResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["analyze"])

Format = Literal["native", "jams"]


def _bpm_range(bpm_min: float | None, bpm_max: float | None) -> tuple[float, float] | None:
    if bpm_min is not None and bpm_max is not None and bpm_max > bpm_min:
        return (bpm_min, bpm_max)
    return None


def _run(path: str, *, key, tempo, structure, structure_activations, stems, genre, bpm_range,
         filename, fmt: Format):
    try:
        result = analyze_track(
            path, key=key, tempo=tempo, structure=structure,
            structure_activations=structure_activations, stems=stems,
            genre=genre, bpm_range=bpm_range,
        )
    except ValueError as exc:  # bad/missing file, unsupported format
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:  # e.g. structure backend not configured
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if fmt == "jams":
        # Returning a Response bypasses response_model (the native AnalyzeResponse).
        return JSONResponse(content=to_jams(result, filename=filename))
    response = AnalyzeResponse(**result)
    response.filename = filename
    return response


def _spawn_job(path: str, *, key, tempo, structure, structure_activations, stems, genre,
               bpm_range, filename) -> str:
    """Start a background analysis thread; the temp file is owned (and removed) by it."""
    registry = get_registry()
    job = registry.create()

    def work() -> None:
        try:
            result = analyze_track(
                path, key=key, tempo=tempo, structure=structure,
                structure_activations=structure_activations, stems=stems,
                genre=genre, bpm_range=bpm_range,
                on_stage=lambda stage, event: (
                    registry.start_stage(job, stage) if event == "start"
                    else registry.end_stage(job, stage)
                ),
            )
            result["filename"] = filename
            registry.finish(job, result)
        except Exception as exc:  # noqa: BLE001 — job carries the error to the client
            logger.exception("async analysis job %s failed", job.id)
            registry.fail(job, str(exc))
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

    threading.Thread(target=work, name=f"analyze-job-{job.id[:8]}", daemon=True).start()
    return job.id


@router.post("/analyze", response_model=None, summary="Analyze an uploaded audio file")
async def analyze_upload(
    file: UploadFile = File(..., description="Audio file (wav/mp3/flac/aiff/ogg/m4a/aac)"),
    key: bool = Form(True),
    tempo: bool = Form(True),
    structure: bool = Form(False),
    activations: bool = Form(False, description="Include the structure resegmentation blob"),
    stems: bool = Form(False),
    genre: str | None = Form(None),
    bpm_min: float | None = Form(None),
    bpm_max: float | None = Form(None),
    async_: bool = Form(False, alias="async", description="Return a job id immediately"),
    format: Format = Query("native", description="'native' (default) or 'jams' (JAMS spec)"),
):
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=422, detail=f"Unsupported format '{suffix}'")

    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {settings.max_upload_mb} MB")

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()

    if async_:
        # The job thread owns (and deletes) the temp file.
        job_id = _spawn_job(
            tmp.name, key=key, tempo=tempo, structure=structure,
            structure_activations=activations, stems=stems,
            genre=genre, bpm_range=_bpm_range(bpm_min, bpm_max), filename=file.filename,
        )
        return JSONResponse(status_code=202, content={"job_id": job_id})

    try:
        return await run_in_threadpool(
            _run, tmp.name, key=key, tempo=tempo, structure=structure,
            structure_activations=activations, stems=stems,
            genre=genre, bpm_range=_bpm_range(bpm_min, bpm_max),
            filename=file.filename, fmt=format,
        )
    finally:
        os.unlink(tmp.name)


@router.get("/jobs/{job_id}", summary="Status of an async analysis job")
async def job_status(job_id: str):
    job = get_registry().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job id")
    return job.to_public()


@router.post("/analyze/path", response_model=AnalyzeResponse, summary="Analyze a file on the server filesystem")
async def analyze_path(
    req: AnalyzePathRequest,
    format: Format = Query("native", description="'native' (default) or 'jams' (JAMS spec)"),
):
    return await run_in_threadpool(
        _run, req.path, key=req.key, tempo=req.tempo, structure=req.structure,
        structure_activations=req.activations, stems=req.stems,
        genre=req.genre, bpm_range=_bpm_range(req.bpm_min, req.bpm_max),
        filename=os.path.basename(req.path), fmt=format,
    )


@router.post("/resegment", response_model=ResegmentResponse,
             summary="Rethreshold cached structure activations")
async def resegment(req: ResegmentRequest):
    """Instant re-segmentation from an ``activations`` blob returned by ``/analyze`` —
    the section-count slider's backend. Pure numpy: no model, no worker subprocess."""
    from jams.analysis.structure import resegment_structure

    try:
        result = resegment_structure(
            req.activations.model_dump(), threshold=req.threshold,
            target_sections=req.target_sections, beats=req.beats,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ResegmentResponse(**result)


@router.get("/stems/file", summary="Fetch a generated stem wav or MIDI file")
async def stems_file(path: str = Query(..., description="Path from a StemsResult")):
    """Serve a worker-generated stem/MIDI file.

    Guarded: only files under the configured stems output dir (or the default temp
    ``jams_stems`` dir) are served, so this can't read arbitrary server paths.
    """
    from fastapi.responses import FileResponse

    settings = get_settings()
    roots = [os.path.realpath(os.path.join(tempfile.gettempdir(), "jams_stems"))]
    if settings.stems_out_dir:
        roots.append(os.path.realpath(settings.stems_out_dir))
    real = os.path.realpath(path)
    if not any(real.startswith(root + os.sep) for root in roots):
        raise HTTPException(status_code=403, detail="Path outside the stems output dir")
    if not os.path.isfile(real):
        raise HTTPException(status_code=404, detail="No such file")
    return FileResponse(real)
