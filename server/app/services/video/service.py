from app.services.video.extractor import get_adapter

class VideoService:
    def __init__(self):
        pass

    def process_video(self, url: str):
        adapter = get_adapter(url)
        metadata = adapter.get_metadata(url)
        transcript = adapter.get_transcript(url)

        return {
            "metadata": metadata,
            "transcript": transcript,
        }