import re
from typing import Dict, Any, List
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from .base import BaseVideoAdapter

class YouTubeAdapter(BaseVideoAdapter):
    def _extract_video_id(self, url: str) -> str:
        """Extracts the YouTube video ID from a given URL."""
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if match:
            return match.group(1)
        raise ValueError("Could not extract YouTube video ID from URL")

    def get_metadata(self, url: str) -> dict[str, Any]:
        ydl_opts = {
            'skip_download': True,
            'extract_flat': False,
            'quiet': True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return {
                    "platform": "YouTube",
                    "url": url,
                    "title": info.get('title'),
                    "duration": info.get('duration'),
                    "channel": info.get('uploader'),
                    "thumbnail": info.get('thumbnail')
                }
            except Exception as e:
                raise ValueError(f"Failed to retrieve metadata: {str(e)}")
        
    def get_transcript(self, url: str) -> list[dict]:
        video_id = self._extract_video_id(url)
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            transcript = transcript_list.find_transcript(['en'])
            return transcript.fetch().to_raw_data()
        except (TranscriptsDisabled, NoTranscriptFound):
            return []
        except Exception as e:
            raise ValueError(f"Failed to retrieve transcript: {str(e)}")
