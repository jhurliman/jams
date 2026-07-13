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
    bpm: float = Field(
        examples=[174.0],
        description="Resolved tempo (octave-corrected when a genre/range hint is given)",
    )
    bpm_raw: float = Field(examples=[87.0], description="Tracker output before octave resolution")
    bpm_alt: float = Field(examples=[87.0], description="The half/double-time alternative")
    octave_resolved: bool
    method: str = Field(examples=["tempocnn-deepsquare"])


class Segment(BaseModel):
    start: float
    end: float
    label: str
    start_beat: int | None = Field(default=None, description="1-indexed beat nearest start")
    end_beat: int | None = Field(default=None, description="1-indexed beat nearest end")


class StructureActivations(BaseModel):
    """Compact activation blob captured at analysis time (opt-in via ``activations=true``).

    Feeds ``POST /v1/resegment`` so a UI can rethreshold the section boundaries instantly
    without re-running the model — this backs the annotator's section-count slider.
    """

    version: int = 1
    duration: float = Field(description="Track duration in seconds")
    frame_rate: float = Field(description="Boundary-candidate frame rate (native, 100/s)")
    candidates: list[tuple[int, float]] = Field(
        default_factory=list, description="Sparse (frame, peak strength) boundary candidates"
    )
    labels: list[str] = Field(description="Class vocabulary in classifier index order")
    label_frame_rate: float = Field(description="Frame rate of the pooled label_probs")
    label_probs: list[list[float]] = Field(
        description="Mean-pooled per-frame class probabilities, frames x classes"
    )
    threshold: float = Field(description="Boundary threshold chosen at analysis time")


class StructureResult(BaseModel):
    bpm: float | None = None
    beats: list[float] = Field(default_factory=list)
    downbeats: list[float] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    method: str = Field(default="allin1-mps-local", examples=["allin1-mps-local:harmonix-all"])
    activations: StructureActivations | None = Field(
        default=None,
        description="Resegmentation blob; present only when requested (and the track was "
        "short enough to analyze unchunked)",
    )


class StemNote(BaseModel):
    onset: float = Field(description="Note-on time (seconds)")
    offset: float = Field(description="Note-off time (seconds)")
    pitch: int = Field(ge=0, le=127, description="MIDI pitch; GM percussion note for drums")
    velocity: int = Field(ge=1, le=127)


class StemTranscription(BaseModel):
    stem_type: str = Field(examples=["drums", "bass", "other", "vocals"])
    gm_program: int = Field(ge=0, le=127, description="General MIDI program (0-indexed)")
    is_drums: bool = Field(description="True => GM percussion on channel 10")
    notes: list[StemNote] = Field(default_factory=list)
    method: str = Field(examples=["basic-pitch", "adtof"])


class StemAudio(BaseModel):
    stem_type: str
    audio_path: str = Field(description="Server-side path to the separated stem wav")


class StemsResult(BaseModel):
    stems: list[StemAudio] = Field(default_factory=list)
    transcriptions: list[StemTranscription] = Field(default_factory=list)
    midi_paths: dict[str, str] = Field(
        default_factory=dict, description="Per-stem + 'combined' MIDI file paths"
    )
    method: str = Field(examples=["demucs-htdemucs+basic-pitch+adtof"])
    duration_sec: float | None = None


class AnalyzeResponse(BaseModel):
    filename: str | None = None
    duration_sec: float | None = None
    key: KeyResult | None = None
    tempo: TempoResult | None = None
    structure: StructureResult | None = None
    stems: StemsResult | None = None


class ResegmentRequest(BaseModel):
    """Rethreshold cached structure activations into a new segmentation (no re-analysis)."""

    activations: StructureActivations
    threshold: float | None = Field(default=None, description="Explicit boundary threshold")
    target_sections: int | None = Field(
        default=None, ge=1, description="Desired section count (picks the threshold for you)"
    )
    beats: list[float] | None = Field(
        default=None, description="Beat grid used to fill start_beat/end_beat (optional)"
    )


class ResegmentResponse(BaseModel):
    segments: list[Segment]
    threshold: float = Field(description="The boundary threshold that was applied")


class AnalyzePathRequest(BaseModel):
    """Analyze a file already on the server's filesystem (e.g. a local DJ library)."""

    path: str = Field(examples=["/Users/me/Music/track.wav"])
    key: bool = True
    tempo: bool = True
    structure: bool = False
    stems: bool = False
    activations: bool = Field(
        default=False, description="Include the structure resegmentation blob in the result"
    )
    genre: str | None = Field(
        default=None, examples=["Drum & Bass"], description="Genre hint for tempo octave resolution"
    )
    bpm_min: float | None = Field(default=None, description="Lower bound of expected tempo octave")
    bpm_max: float | None = Field(default=None, description="Upper bound of expected tempo octave")
