"""
YouTube platform adapter.

Metadata — YouTube oEmbed endpoint (official, public, no credentials needed).
Transcript — youtube-transcript-api library.
             Fetches official and automatically generated captions directly.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from app.services.video.adapters.base import BaseVideoAdapter
from app.services.video.exceptions import (
    InvalidVideoURLError,
    MetadataFetchError,
    NetworkTimeoutError,
    RateLimitError,
    TranscriptFetchError,
    TranscriptUnavailableError,
    VerificationRequiredError,
    VideoProcessingError,
)
from app.services.video.models import (
    Transcript,
    TranscriptSegment,
    TranscriptSourceType,
    VideoMetadata,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 10  # seconds

YOUTUBE_HOSTS: frozenset[str] = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)

# Ordered by preference; the adapter tries these language codes first.
PREFERRED_TRANSCRIPT_LANGUAGES: list[str] = [
    "ar", "en", "fr", "es", "de", "zh", "ja", "ru", "pt",
]

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class YouTubeAdapter(BaseVideoAdapter):
    """Adapter for youtube.com and youtu.be URLs."""

    # ------------------------------------------------------------------
    # URL parsing
    # ------------------------------------------------------------------

    def _extract_video_id(self, url: str) -> str:
        """
        Extract and validate the 11-character YouTube video ID from a URL.

        Supports:
          https://www.youtube.com/watch?v=VIDEO_ID
          https://youtu.be/VIDEO_ID
          https://www.youtube.com/embed/VIDEO_ID
          https://www.youtube.com/shorts/VIDEO_ID
          https://www.youtube.com/live/VIDEO_ID
        """
        try:
            parsed = urllib.parse.urlparse(url)
        except ValueError as exc:
            raise InvalidVideoURLError(f"Malformed URL: {url}") from exc

        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https"):
            raise InvalidVideoURLError(
                f"URL scheme {scheme!r} is not supported. Only http and https are allowed."
            )

        host = (parsed.hostname or "").lower()
        if host not in YOUTUBE_HOSTS:
            raise InvalidVideoURLError(f"Host {host!r} is not a supported YouTube domain.")

        video_id: Optional[str] = None

        if host == "youtu.be":
            parts = parsed.path.strip("/").split("/")
            video_id = parts[0] if parts else None
        elif parsed.path.rstrip("/") == "/watch":
            params = urllib.parse.parse_qs(parsed.query)
            video_id = (params.get("v") or [None])[0]
        else:
            m = re.match(
                r"^/(?:embed|shorts|live)/([A-Za-z0-9_-]{11})(?:/.*)?$",
                parsed.path,
            )
            if m:
                video_id = m.group(1)

        if not video_id or not _VIDEO_ID_RE.match(video_id):
            raise InvalidVideoURLError(
                f"Could not extract a valid 11-character video ID from URL: {url}"
            )

        return video_id

    # ------------------------------------------------------------------
    # Metadata — public oEmbed endpoint (no auth/keys required)
    # ------------------------------------------------------------------

    def _fetch_metadata_oembed(self, video_id: str, original_url: str) -> VideoMetadata:
        """
        Fetch basic metadata via YouTube's public oEmbed endpoint.
        Does not require authentication. Does not return duration.
        """
        try:
            req = urllib.request.Request(
                f"https://www.youtube.com/oembed"
                f"?url={urllib.parse.quote(original_url)}&format=json",
                headers={"User-Agent": "video-summarizer/1.0"},
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except TimeoutError as exc:
            raise NetworkTimeoutError(
                f"oEmbed request timed out for video {video_id}"
            ) from exc
        except urllib.error.HTTPError as exc:
            raise MetadataFetchError(
                f"oEmbed HTTP error {exc.code} for video {video_id}"
            ) from exc

        return VideoMetadata(
            source_video_id=video_id,
            platform="youtube",
            original_url=original_url,
            canonical_url=f"https://www.youtube.com/watch?v={video_id}",
            title=data.get("title") or None,
            author=data.get("author_name") or None,
            duration_seconds=None,  # oEmbed does not expose duration
            thumbnail_url=data.get("thumbnail_url") or None,
        )

    # ------------------------------------------------------------------
    # Public interface: get_metadata
    # ------------------------------------------------------------------

    def get_metadata(self, url: str) -> VideoMetadata:
        video_id = self._extract_video_id(url)
        return self._fetch_metadata_oembed(video_id, url)

    # ------------------------------------------------------------------
    # Public interface: get_transcript
    # ------------------------------------------------------------------

    def get_transcript(self, url: str) -> Transcript:
        video_id = self._extract_video_id(url)

        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)

            # Prefer a transcript in one of the preferred languages.
            try:
                fetched = transcript_list.find_transcript(PREFERRED_TRANSCRIPT_LANGUAGES)
            except NoTranscriptFound:
                # Fall back to the first available transcript regardless of language.
                fetched = next(iter(transcript_list), None)
                if fetched is None:
                    raise TranscriptUnavailableError(
                        f"No transcripts are available for video {video_id}."
                    )

            raw_segments = fetched.fetch().to_raw_data()
            if not raw_segments:
                raise TranscriptUnavailableError(
                    f"The transcript for video {video_id} was found but contains no segments."
                )

            source_type = (
                TranscriptSourceType.GENERATED
                if getattr(fetched, "is_generated", False)
                else TranscriptSourceType.MANUAL
            )

            segments = [
                TranscriptSegment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("start", 0.0)) + float(seg.get("duration", 0.0)),
                    text=(seg.get("text") or "").strip(),
                )
                for seg in raw_segments
                if (seg.get("text") or "").strip()
            ]

            return Transcript(
                segments=segments,
                language=fetched.language_code or None,
                source_type=source_type,
            )

        except (TranscriptsDisabled,) as exc:
            raise TranscriptUnavailableError(
                f"Captions are disabled for video {video_id}."
            ) from exc

        except NoTranscriptFound as exc:
            raise TranscriptUnavailableError(
                f"No transcript was found for video {video_id}."
            ) from exc

        except CouldNotRetrieveTranscript as exc:
            msg = str(exc).lower()
            if "429" in msg or "too many requests" in msg:
                raise RateLimitError(
                    "YouTube is temporarily rate-limiting transcript requests from this IP address."
                ) from exc
            if "sign in" in msg or "bot" in msg or "verification" in msg:
                raise VerificationRequiredError(
                    "YouTube requires sign-in verification for transcript requests from this IP address."
                ) from exc
            raise TranscriptUnavailableError(
                f"Could not retrieve transcript for video {video_id}: {exc}"
            ) from exc

        except VideoProcessingError:
            raise

        except Exception as exc:
            raise TranscriptFetchError(
                f"Unexpected error retrieving transcript for video {video_id}: {exc}"
            ) from exc
