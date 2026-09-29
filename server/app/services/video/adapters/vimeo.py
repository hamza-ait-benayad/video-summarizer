from .base import BaseVideoAdapter
from typing import Dict, Any

class VimeoAdapter(BaseVideoAdapter):
    def get_metadata(self, url: str) -> dict[str, Any]:
        return {
            "platform": "Vimeo",
            "url": url,
            "title": "Example Vimeo Title",
            "duration": 0
        }
        
    def get_transcript(self, url: str) -> list[dict]:
        return [{"text": "Example Vimeo transcript", "start": 0.0, "duration": 0.0}]
