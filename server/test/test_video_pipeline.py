"""
Unit tests for the video-processing pipeline.

All external provider calls (youtube-transcript-api, YouTube Data API)
are mocked. No live network requests are made.

Run with:
    pytest test/test_video_pipeline.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.video.detector import Platform, detect_platform
from app.services.video.exceptions import (
    InvalidVideoURLError,
    RateLimitError,
    TranscriptUnavailableError,
    UnsupportedPlatformError,
    VerificationRequiredError,
)
from app.services.video.extractor import get_adapter
from app.services.video.models import (
    Transcript,
    TranscriptSegment,
    TranscriptSourceType,
    VideoMetadata,
)
from app.services.video.normaliser import normalise_transcript


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

class TestDetectPlatform:

    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/shorts/dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
    ])
    def test_youtube_urls(self, url: str):
        assert detect_platform(url) == Platform.YOUTUBE

    @pytest.mark.parametrize("url", [
        "https://vimeo.com/12345678",
        "https://www.vimeo.com/12345678",
    ])
    def test_vimeo_urls(self, url: str):
        assert detect_platform(url) == Platform.VIMEO

    @pytest.mark.parametrize("url", [
        "https://tiktok.com/@user/video/123",
        "https://example.com/video",
        "not-a-url",
        "",
    ])
    def test_unknown_urls(self, url: str):
        assert detect_platform(url) == Platform.UNKNOWN


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

class TestGetAdapter:

    def test_returns_youtube_adapter_for_youtube_url(self):
        from app.services.video.adapters.youtube import YouTubeAdapter
        adapter = get_adapter("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert isinstance(adapter, YouTubeAdapter)

    def test_raises_for_unsupported_platform(self):
        with pytest.raises(UnsupportedPlatformError) as exc_info:
            get_adapter("https://vimeo.com/12345678")
        assert exc_info.value.code == "UNSUPPORTED_PLATFORM"

    def test_raises_for_unknown_platform(self):
        with pytest.raises(UnsupportedPlatformError):
            get_adapter("https://example.com/video")


# ---------------------------------------------------------------------------
# YouTubeAdapter — video ID extraction
# ---------------------------------------------------------------------------

class TestYouTubeVideoIdExtraction:

    def _adapter(self):
        from app.services.video.adapters.youtube import YouTubeAdapter
        return YouTubeAdapter()

    @pytest.mark.parametrize("url, expected_id", [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ])
    def test_valid_url_formats(self, url: str, expected_id: str):
        assert self._adapter()._extract_video_id(url) == expected_id

    @pytest.mark.parametrize("url", [
        "https://example.com/watch?v=dQw4w9WgXcQ",   # wrong host
        "ftp://www.youtube.com/watch?v=dQw4w9WgXcQ",  # bad scheme
        "https://www.youtube.com/channel/UCxxxxxx",    # no video ID
        "https://www.youtube.com/watch?v=SHORT",       # ID too short
    ])
    def test_invalid_urls_raise(self, url: str):
        with pytest.raises(InvalidVideoURLError):
            self._adapter()._extract_video_id(url)


# ---------------------------------------------------------------------------
# YouTubeAdapter — metadata (mocked)
# ---------------------------------------------------------------------------

class TestYouTubeAdapterMetadata:

    def _adapter(self):
        from app.services.video.adapters.youtube import YouTubeAdapter
        return YouTubeAdapter()

    def _mock_oembed_response(self):
        return {
            "title": "Test Video",
            "author_name": "Test Channel",
            "thumbnail_url": "https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
        }

    @patch("app.services.video.adapters.youtube.urllib.request.urlopen")
    def test_returns_typed_video_metadata(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = __import__("json").dumps(self._mock_oembed_response()).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = self._adapter().get_metadata(url)

        assert isinstance(result, VideoMetadata)
        assert result.title == "Test Video"
        assert result.author == "Test Channel"
        assert result.duration_seconds is None
        assert result.thumbnail_url == "https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
        assert result.platform == "youtube"
        assert result.source_video_id == "dQw4w9WgXcQ"

    def test_raises_invalid_url_for_bad_host(self):
        with pytest.raises(InvalidVideoURLError):
            self._adapter().get_metadata("https://example.com/watch?v=dQw4w9WgXcQ")


# ---------------------------------------------------------------------------
# YouTubeAdapter — transcript (mocked)
# ---------------------------------------------------------------------------

class TestYouTubeAdapterTranscript:

    def _adapter(self):
        from app.services.video.adapters.youtube import YouTubeAdapter
        return YouTubeAdapter()

    def _make_raw_segments(self):
        return [
            {"start": 0.0, "duration": 2.5, "text": "Hello world"},
            {"start": 2.5, "duration": 3.0, "text": "This is a test"},
        ]

    @patch("app.services.video.adapters.youtube.YouTubeTranscriptApi")
    def test_returns_typed_transcript(self, mock_api_class):
        mock_fetched = MagicMock()
        mock_fetched.language_code = "en"
        mock_fetched.is_generated = False
        mock_fetched.fetch.return_value.to_raw_data.return_value = self._make_raw_segments()

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_transcript.return_value = mock_fetched

        mock_api = MagicMock()
        mock_api.list.return_value = mock_transcript_list
        mock_api_class.return_value = mock_api

        result = self._adapter().get_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert isinstance(result, Transcript)
        assert len(result.segments) == 2
        assert result.language == "en"
        assert result.source_type == TranscriptSourceType.MANUAL
        assert result.segments[0].text == "Hello world"
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 2.5

    @patch("app.services.video.adapters.youtube.YouTubeTranscriptApi")
    def test_raises_transcript_unavailable_when_disabled(self, mock_api_class):
        from youtube_transcript_api import TranscriptsDisabled

        mock_api = MagicMock()
        mock_api.list.side_effect = TranscriptsDisabled("dQw4w9WgXcQ")
        mock_api_class.return_value = mock_api

        with pytest.raises(TranscriptUnavailableError) as exc_info:
            self._adapter().get_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert exc_info.value.code == "TRANSCRIPT_UNAVAILABLE"

    @patch("app.services.video.adapters.youtube.YouTubeTranscriptApi")
    def test_raises_rate_limit_on_429(self, mock_api_class):
        from youtube_transcript_api import CouldNotRetrieveTranscript

        class RateLimitExc(CouldNotRetrieveTranscript):
            def __str__(self):
                return "429 too many requests"

        mock_api = MagicMock()
        mock_api.list.side_effect = RateLimitExc("dQw4w9WgXcQ")
        mock_api_class.return_value = mock_api

        with pytest.raises(RateLimitError) as exc_info:
            self._adapter().get_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert exc_info.value.code == "RATE_LIMITED"
        assert exc_info.value.retryable is True

    @patch("app.services.video.adapters.youtube.YouTubeTranscriptApi")
    def test_raises_verification_required_on_sign_in(self, mock_api_class):
        from youtube_transcript_api import CouldNotRetrieveTranscript

        class VerificationExc(CouldNotRetrieveTranscript):
            def __str__(self):
                return "sign in to confirm you're not a bot"

        mock_api = MagicMock()
        mock_api.list.side_effect = VerificationExc("dQw4w9WgXcQ")
        mock_api_class.return_value = mock_api

        with pytest.raises(VerificationRequiredError) as exc_info:
            self._adapter().get_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert exc_info.value.code == "VERIFICATION_REQUIRED"
        assert exc_info.value.retryable is False


# ---------------------------------------------------------------------------
# Transcript normaliser
# ---------------------------------------------------------------------------

class TestNormaliseTranscript:

    def _make_transcript(self, segments):
        return Transcript(segments=segments, language="en", source_type=TranscriptSourceType.MANUAL)

    def test_removes_empty_segments(self):
        t = self._make_transcript([
            TranscriptSegment(start=0.0, end=2.0, text="   "),
            TranscriptSegment(start=2.0, end=4.0, text="Hello"),
        ])
        result = normalise_transcript(t)
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello"

    def test_collapses_whitespace(self):
        t = self._make_transcript([
            TranscriptSegment(start=0.0, end=2.0, text="  hello   world  "),
        ])
        result = normalise_transcript(t)
        assert result.segments[0].text == "hello world"
        # After _WHITESPACE_RE.sub + strip: "hello world"
        result2_input = Transcript(
            segments=[TranscriptSegment(start=0.0, end=2.0, text="hello\n\tworld")],
            language="en",
            source_type=TranscriptSourceType.MANUAL,
        )
        result2 = normalise_transcript(result2_input)
        assert result2.segments[0].text == "hello world"

    def test_sorts_segments_by_start_time(self):
        t = self._make_transcript([
            TranscriptSegment(start=5.0, end=7.0, text="Second"),
            TranscriptSegment(start=0.0, end=2.0, text="First"),
        ])
        result = normalise_transcript(t)
        assert result.segments[0].text == "First"
        assert result.segments[1].text == "Second"

    def test_fixes_zero_length_timestamps(self):
        t = self._make_transcript([
            TranscriptSegment(start=3.0, end=3.0, text="Instant"),
        ])
        result = normalise_transcript(t)
        # end should have been extended
        assert result.segments[0].end > result.segments[0].start

    def test_preserves_language_and_source_type(self):
        t = Transcript(
            segments=[TranscriptSegment(start=0.0, end=1.0, text="Bonjour")],
            language="fr",
            source_type=TranscriptSourceType.GENERATED,
        )
        result = normalise_transcript(t)
        assert result.language == "fr"
        assert result.source_type == TranscriptSourceType.GENERATED

    def test_empty_input_returns_empty_output(self):
        t = Transcript(segments=[], language=None, source_type=TranscriptSourceType.UNKNOWN)
        result = normalise_transcript(t)
        assert result.segments == []


