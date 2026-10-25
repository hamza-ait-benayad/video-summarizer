"""
Abstract base class for platform adapters.

Every adapter must implement:
  get_metadata(url) -> VideoMetadata
  get_transcript(url) -> Transcript

Adapters must raise exceptions from app.services.video.exceptions —
never return bare dicts or None to indicate failure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.video.models import Transcript, VideoMetadata


class BaseVideoAdapter(ABC):

    @abstractmethod
    def get_metadata(self, url: str) -> VideoMetadata:
        """
        Retrieve structured metadata for the given video URL.

        Returns:
            VideoMetadata — typed metadata object.

        Raises:
            InvalidVideoURLError       — URL is malformed or not supported.
            VideoNotFoundError         — Video is private, deleted, or unavailable.
            MetadataFetchError         — Retrieval failed for an unexpected reason.
            RateLimitError             — Provider rate-limited the request.
            VerificationRequiredError  — Provider requires authentication.
            NetworkTimeoutError        — Request timed out.
        """
        ...

    @abstractmethod
    def get_transcript(self, url: str) -> Transcript:
        """
        Retrieve available transcript/captions for the given video URL.

        Returns:
            Transcript — typed object containing ordered segments.

        Raises:
            InvalidVideoURLError       — URL is malformed or not supported.
            TranscriptUnavailableError — No usable transcript exists.
            RateLimitError             — Provider rate-limited the request.
            VerificationRequiredError  — Provider requires authentication.
            NetworkTimeoutError        — Request timed out.
            TranscriptFetchError       — Unexpected error during retrieval.
        """
        ...
