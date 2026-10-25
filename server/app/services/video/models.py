"""
Typed Pydantic models for the video-processing pipeline.

These models are used internally by services and adapters.
They are separate from the SQLAlchemy ORM models (app/models/) and the
FastAPI request/response schemas (app/schemas/).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TranscriptSourceType(str, Enum):
    """How the transcript was obtained."""
    MANUAL = "manual"        # Manually created captions
    GENERATED = "generated"  # Auto-generated captions
    UNKNOWN = "unknown"      # Source not determinable


# ---------------------------------------------------------------------------
# VideoMetadata
# ---------------------------------------------------------------------------

class VideoMetadata(BaseModel):
    """
    Structured representation of video metadata returned by an adapter.

    All optional fields must remain None if the provider did not supply them.
    Do not invent values for missing fields.
    """
    source_video_id: str
    platform: str
    original_url: str
    canonical_url: str
    title: Optional[str] = None
    author: Optional[str] = None
    duration_seconds: Optional[int] = None
    thumbnail_url: Optional[str] = None
    upload_date: Optional[str] = None   # ISO 8601 date string: YYYY-MM-DD
    language: Optional[str] = None      # BCP-47 language tag if known


# ---------------------------------------------------------------------------
# Transcript models
# ---------------------------------------------------------------------------

class TranscriptSegment(BaseModel):
    """A single timed caption segment."""
    start: float
    end: float
    text: str

    @field_validator("start", "end")
    @staticmethod
    def must_be_non_negative(v: float) -> float:
        if v < 0:
            raise ValueError("Timestamp must be non-negative")
        return v


class Transcript(BaseModel):
    """
    A complete transcript produced by an adapter.

    segments — ordered list of timed caption segments.
    language  — BCP-47 language tag if known, otherwise None.
    source_type — how the transcript was obtained (manual, generated, unknown).
    """
    segments: list[TranscriptSegment]
    language: Optional[str] = None
    source_type: TranscriptSourceType = TranscriptSourceType.UNKNOWN


# ---------------------------------------------------------------------------
# ProcessingResult — aggregate output of the video-processing pipeline
# ---------------------------------------------------------------------------

class VideoProcessingResult(BaseModel):
    """
    Combined output of the video-processing pipeline.
    Both fields may be absent when extraction partially fails.
    """
    metadata: Optional[VideoMetadata] = None
    transcript: Optional[Transcript] = None
