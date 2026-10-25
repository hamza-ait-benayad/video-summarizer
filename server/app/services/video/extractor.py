"""
Adapter registry and factory.

get_adapter(url) detects the platform and returns a ready-to-use adapter.
Raises UnsupportedPlatformError for platforms with no registered adapter.
"""

from __future__ import annotations

from app.services.video.adapters.base import BaseVideoAdapter
from app.services.video.adapters.youtube import YouTubeAdapter
from app.services.video.detector import Platform, detect_platform
from app.services.video.exceptions import UnsupportedPlatformError

_REGISTRY: dict[Platform, type[BaseVideoAdapter]] = {
    Platform.YOUTUBE: YouTubeAdapter,
}


def get_adapter(url: str) -> BaseVideoAdapter:
    """
    Return the appropriate adapter for the given URL.

    Raises:
        UnsupportedPlatformError — when no adapter is registered for the detected platform.
    """
    platform = detect_platform(url)

    adapter_class = _REGISTRY.get(platform)
    if adapter_class is None:
        raise UnsupportedPlatformError(platform.value)

    return adapter_class()
