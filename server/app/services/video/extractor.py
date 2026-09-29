from .detector import detect_platform
from .adapters.base import BaseVideoAdapter
from .adapters.youtube import YouTubeAdapter
from .adapters.vimeo import VimeoAdapter

ADAPTERS: dict[str, type[BaseVideoAdapter]] = {
    "youtube": YouTubeAdapter,
    "vimeo": VimeoAdapter,
}

def get_adapter(url: str) -> BaseVideoAdapter:
    platform = detect_platform(url)

    adapter_class = ADAPTERS.get(platform.value)

    if adapter_class is None:
        raise ValueError(f"Unsupported platform: {platform.value}")

    return adapter_class()
