"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class KeyResult(BaseModel):
    key: str = Field(examples=["A# minor"])
    tonic: str = Field(examples=["A#"])
    mode: str = Field(examples=["minor"])
    confidence: float = Field(ge=0.0, le=1.0)
    method: str = Field(examples=["essentia-edma"])


class TempoResult(BaseModel):
    bpm: float = Field(examples=[174.0], description="Resolved tempo (octave-corrected when a genre/range hint is given)")
    bpm_raw: float = Field(examples=[87.0], description="Tracker output before octave resolution")
    bpm_alt: float = Field(examples=[87.0], description="The half/double-time alternative")
    octave_resolved: bool
    method: str = Field(examples=["tempocnn-deepsquare"])


class Segment(BaseModel):
    start: float
    end: float
    label: str


class StructureResult(BaseModel):
    bpm: float | None = None
    beats: list[float] = Field(default_factory=list)
    downbeats: list[float] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    method: str = "allin1-replicate"


class AnalyzeResponse(BaseModel):
    filename: str | None = None
    duration_sec: float | None = None
    key: KeyResult | None = None
    tempo: TempoResult | None = None
    structure: StructureResult | None = None


class AnalyzePathRequest(BaseModel):
    """Analyze a file already on the server's filesystem (e.g. a local DJ library)."""

    path: str = Field(examples=["/Users/me/Music/track.wav"])
    key: bool = True
    tempo: bool = True
    structure: bool = False
    genre: str | None = Field(default=None, examples=["Drum & Bass"], description="Genre hint for tempo octave resolution")
    bpm_min: float | None = Field(default=None, description="Lower bound of the expected tempo octave")
    bpm_max: float | None = Field(default=None, description="Upper bound of the expected tempo octave")
