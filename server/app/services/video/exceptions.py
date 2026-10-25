"""
Video-processing domain exceptions.

Each exception carries a machine-readable code and a human-readable message.
The retryable flag indicates whether the caller may safely retry the operation.

Error codes
-----------
INVALID_URL            — URL scheme, hostname, or format is not accepted.
UNSUPPORTED_PLATFORM   — Host is valid but the platform has no adapter.
VIDEO_NOT_FOUND        — Video ID resolves to nothing (private, deleted, etc.).
METADATA_FETCH_ERROR   — Metadata request failed for an unexpected reason.
TRANSCRIPT_UNAVAILABLE — No usable transcript exists for this video.
RATE_LIMITED           — Provider returned HTTP 429 or equivalent.
VERIFICATION_REQUIRED  — Provider requires sign-in or CAPTCHA verification.
NETWORK_TIMEOUT        — Request timed out before the provider responded.
TRANSCRIPT_FETCH_ERROR — Unexpected error while retrieving transcript content.
"""


class VideoProcessingError(Exception):
    """Base exception for all video-processing pipeline failures."""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


class InvalidVideoURLError(VideoProcessingError):
    """URL scheme, hostname, or video-ID format is not accepted."""

    def __init__(self, message: str = "Invalid or unsupported video URL") -> None:
        super().__init__(code="INVALID_URL", message=message, retryable=False)


class UnsupportedPlatformError(VideoProcessingError):
    """The detected platform has no registered adapter."""

    def __init__(self, platform: str) -> None:
        super().__init__(
            code="UNSUPPORTED_PLATFORM",
            message=f"No adapter is available for platform: {platform}",
            retryable=False,
        )


class VideoNotFoundError(VideoProcessingError):
    """The video exists syntactically but is private, deleted, or age-restricted."""

    def __init__(self, message: str = "Video not found or is unavailable") -> None:
        super().__init__(code="VIDEO_NOT_FOUND", message=message, retryable=False)


class MetadataFetchError(VideoProcessingError):
    """Metadata retrieval failed for an unexpected reason."""

    def __init__(self, message: str = "Failed to retrieve video metadata") -> None:
        super().__init__(code="METADATA_FETCH_ERROR", message=message, retryable=False)


class TranscriptUnavailableError(VideoProcessingError):
    """No usable transcript is available for this video."""

    def __init__(self, message: str = "No captions or transcript available for this video") -> None:
        super().__init__(code="TRANSCRIPT_UNAVAILABLE", message=message, retryable=False)


class RateLimitError(VideoProcessingError):
    """Provider returned HTTP 429 or an equivalent rate-limit response."""

    def __init__(self, message: str = "Provider rate limit reached") -> None:
        super().__init__(code="RATE_LIMITED", message=message, retryable=True)


class VerificationRequiredError(VideoProcessingError):
    """Provider requires additional verification (sign-in, CAPTCHA, etc.)."""

    def __init__(self, message: str = "Provider requires sign-in or verification") -> None:
        super().__init__(code="VERIFICATION_REQUIRED", message=message, retryable=False)


class NetworkTimeoutError(VideoProcessingError):
    """Request to the provider timed out."""

    def __init__(self, message: str = "Provider request timed out") -> None:
        super().__init__(code="NETWORK_TIMEOUT", message=message, retryable=True)


class TranscriptFetchError(VideoProcessingError):
    """Unexpected error while retrieving transcript content."""

    def __init__(self, message: str = "Unexpected error retrieving transcript") -> None:
        super().__init__(code="TRANSCRIPT_FETCH_ERROR", message=message, retryable=False)
