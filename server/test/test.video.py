from app.services.video.extractor import get_adapter

url = "https://www.youtube.com/watch?v=G0_wVLhI-Ds"

adapter = get_adapter(url)

metadata = adapter.get_metadata(url)
transcript = adapter.get_transcript(url)

print(metadata)
print(transcript)