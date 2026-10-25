from youtube_transcript_api import YouTubeTranscriptApi

VIDEO_ID = "aYFHW4Q06_U"
LANGUAGES = ["en", "fr", "ar", "es"]

def main():
    print(f"Testing YouTube transcript retrieval: {VIDEO_ID}")

    try:
        api = YouTubeTranscriptApi()

        # Try the preferred languages in order.
        transcript = api.fetch(
            VIDEO_ID,
            languages=LANGUAGES,
        )

        segments = transcript.to_raw_data()

        if not segments:
            print("No transcript segments returned.")
            return

        print("\nSUCCESS")
        print(f"Language: {transcript.language}")
        print(f"Language code: {transcript.language_code}")
        print(f"Generated automatically: {transcript.is_generated}")
        print(f"Number of segments: {len(segments)}")

        print("\nFirst 5 segments:")

        for segment in segments[:5]:
            start = segment["start"]
            duration = segment["duration"]
            text = segment["text"]

            print(f"[{start:.2f}s - {start + duration:.2f}s] {text}")

        full_text = " ".join(
            segment["text"] for segment in segments
        )

        print("\nTranscript preview:")
        print(full_text[:500])

    except Exception as exc:
        print("\nFAILED")
        print(f"Exception: {type(exc).__name__}")
        print(f"Details: {exc}")


if __name__ == "__main__":
    main()
