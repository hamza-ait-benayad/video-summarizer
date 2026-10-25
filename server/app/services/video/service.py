"""
Video-processing pipeline service.

Orchestrates URL validation, platform detection, adapter selection,
metadata extraction, transcript extraction, and transcript normalisation.

The service returns a VideoProcessingResult Pydantic model.
The pipeline steps are independent of the LLM provider and the database.
"""

from __future__ import annotations

from app.services.video.exceptions import VideoProcessingError
from app.services.video.extractor import get_adapter
from app.services.video.models import (
    Transcript,
    VideoMetadata,
    VideoProcessingResult,
)
from app.services.video.normaliser import normalise_transcript


class VideoService:

    def get_adapter(self, url: str):
        """Expose adapter lookup for callers that need it directly."""
        return get_adapter(url)

    def get_metadata(self, url: str) -> VideoMetadata:
        """Retrieve and return typed metadata for the given URL."""
        adapter = get_adapter(url)
        return adapter.get_metadata(url)

    def get_transcript(self, url: str) -> Transcript:
        """Retrieve, normalise, and return a typed Transcript for the given URL."""
        adapter = get_adapter(url)
        raw = adapter.get_transcript(url)
        return normalise_transcript(raw)

    def process_video(self, url: str) -> VideoProcessingResult:
        """
        Run the full extraction pipeline for a video URL.

        Metadata and transcript are fetched independently so that
        a transcript failure does not discard successfully fetched metadata.

        Returns:
            VideoProcessingResult — at least one of .metadata or .transcript
            will be populated on partial success.

        Raises:
            InvalidVideoURLError      — URL is not acceptable.
            UnsupportedPlatformError  — No adapter for this platform.
            VideoProcessingError      — Any other pipeline failure.
        """
        adapter = get_adapter(url)

        # Step 1 — Metadata (independent of transcript)
        metadata: VideoMetadata | None = None
        try:
            metadata = adapter.get_metadata(url)
        except VideoProcessingError:
            raise
        except Exception as exc:
            raise VideoProcessingError(
                code="METADATA_FETCH_ERROR",
                message=f"Metadata extraction failed: {exc}",
            ) from exc

        # Step 2 — Transcript (independent of metadata)
        transcript: Transcript | None = None
        try:
            raw = adapter.get_transcript(url)
            transcript = normalise_transcript(raw)
        except VideoProcessingError:
            raise
        except Exception as exc:
            raise VideoProcessingError(
                code="TRANSCRIPT_FETCH_ERROR",
                message=f"Transcript extraction failed: {exc}",
            ) from exc

        return VideoProcessingResult(metadata=metadata, transcript=transcript)