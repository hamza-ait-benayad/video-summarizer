from enum import Enum
from urllib.parse import urlparse


class Platform(str, Enum):
    YOUTUBE = "youtube"
    VIMEO = "vimeo"
    TIKTOK = "tiktok"
    UNKNOWN = "unknown"


def detect_platform(url: str) -> Platform:
    hostname = (urlparse(url).hostname or "").lower()

    if hostname == "youtube.com" or hostname.endswith(".youtube.com"):
        return Platform.YOUTUBE

    if hostname == "youtu.be":
        return Platform.YOUTUBE

    if hostname == "vimeo.com" or hostname.endswith(".vimeo.com"):
        return Platform.VIMEO

    if hostname == "tiktok.com" or hostname.endswith(".tiktok.com"):
        return Platform.TIKTOK

    return Platform.UNKNOWN
