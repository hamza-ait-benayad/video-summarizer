"""
Platform detector.

Maps a URL to a Platform enum value by inspecting the hostname.
Returns Platform.UNKNOWN for unrecognised hosts; does not raise.

Callers that require a registered adapter should use get_adapter() in
extractor.py, which converts UNKNOWN (and any platform with no adapter)
into an UnsupportedPlatformError.
"""

from __future__ import annotations

from enum import Enum
from urllib.parse import urlparse


class Platform(str, Enum):
    YOUTUBE = "youtube"
    VIMEO = "vimeo"
    UNKNOWN = "unknown"


# Mapping of recognised hostnames to their Platform value.
_PLATFORM_MAP: dict[str, Platform] = {
    "youtube.com":       Platform.YOUTUBE,
    "www.youtube.com":   Platform.YOUTUBE,
    "m.youtube.com":     Platform.YOUTUBE,
    "music.youtube.com": Platform.YOUTUBE,
    "youtu.be":          Platform.YOUTUBE,
    "vimeo.com":         Platform.VIMEO,
    "www.vimeo.com":     Platform.VIMEO,
    "player.vimeo.com":  Platform.VIMEO,
}


def detect_platform(url: str) -> Platform:
    """
    Return the Platform for the given URL.

    Returns Platform.UNKNOWN rather than raising when the host is not recognised.
    """
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except ValueError:
        return Platform.UNKNOWN

    return _PLATFORM_MAP.get(hostname, Platform.UNKNOWN)
