"""API routes — analyze a track by upload or by server-side path."""

from __future__ import annotations

import logging
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from jams.analysis import analyze_track
from jams.analysis.audio import SUPPORTED_FORMATS
from jams.config import get_settings
from jams.models import AnalyzePathRequest, AnalyzeResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["analyze"])


def _bpm_range(bpm_min: float | None, bpm_max: float | None) -> tuple[float, float] | None:
    if bpm_min is not None and bpm_max is not None and bpm_max > bpm_min:
        return (bpm_min, bpm_max)
    return None


def _run(path: str, *, key, tempo, structure, genre, bpm_range) -> AnalyzeResponse:
    try:
        result = analyze_track(
            path, key=key, tempo=tempo, structure=structure, genre=genre, bpm_range=bpm_range
        )
    except ValueError as exc:  # bad/missing file, unsupported format
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:  # e.g. structure requested without Replicate
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AnalyzeResponse(**result)


@router.post("/analyze", response_model=AnalyzeResponse, summary="Analyze an uploaded audio file")
async def analyze_upload(
    file: UploadFile = File(..., description="Audio file (wav/mp3/flac/aiff/ogg/m4a/aac)"),
    key: bool = Form(True),
    tempo: bool = Form(True),
    structure: bool = Form(False),
    genre: str | None = Form(None),
    bpm_min: float | None = Form(None),
    bpm_max: float | None = Form(None),
) -> AnalyzeResponse:
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=422, detail=f"Unsupported format '{suffix}'")

    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {settings.max_upload_mb} MB")

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        response = await run_in_threadpool(
            _run, tmp.name, key=key, tempo=tempo, structure=structure,
            genre=genre, bpm_range=_bpm_range(bpm_min, bpm_max),
        )
    finally:
        os.unlink(tmp.name)
    response.filename = file.filename
    return response


@router.post("/analyze/path", response_model=AnalyzeResponse, summary="Analyze a file on the server filesystem")
async def analyze_path(req: AnalyzePathRequest) -> AnalyzeResponse:
    response = await run_in_threadpool(
        _run, req.path, key=req.key, tempo=req.tempo, structure=req.structure,
        genre=req.genre, bpm_range=_bpm_range(req.bpm_min, req.bpm_max),
    )
    response.filename = os.path.basename(req.path)
    return response
