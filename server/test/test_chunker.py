from app.services.rag.chunker import chunk_transcript


def test_chunk_transcript_basic():
    segments = [
        {"start_time": 0.0, "end_time": 5.0, "text": "Today, we are going to discuss machine learning."},
        {"start_time": 5.0, "end_time": 10.0, "text": "Machine learning allows computers to learn from data."},
        {"start_time": 10.0, "end_time": 15.0, "text": "Deep learning is a subset of machine learning using neural networks."},
    ]
    chunks = chunk_transcript(segments, video_id="vid-1", chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.video_id == "vid-1"
    assert chunk.sequence == 0
    assert chunk.start_time == 0.0
    assert chunk.end_time == 15.0
    assert "Today, we are going to discuss machine learning." in chunk.text
    assert "Deep learning is a subset of machine learning" in chunk.text
    assert chunk.chunk_id is not None


def test_chunk_transcript_timestamp_preservation_and_overlap():
    # Construct segments where multiple chunks will be generated
    segments = [
        {"start_time": 0.0, "end_time": 4.0, "text": "Segment one discusses introductory artificial intelligence principles."},
        {"start_time": 4.0, "end_time": 8.0, "text": "Segment two expands on data preparation and model features."},
        {"start_time": 8.0, "end_time": 12.0, "text": "Segment three covers gradient descent and optimization algorithms."},
        {"start_time": 12.0, "end_time": 16.0, "text": "Segment four examines validation loss and regularization techniques."},
        {"start_time": 16.0, "end_time": 20.0, "text": "Segment five concludes with model evaluation and test metrics."},
    ]

    # Use small chunk size (18 tokens) and small overlap (8 tokens) to force chunking with overlap
    chunks = chunk_transcript(segments, video_id="vid-overlap", chunk_size=18, chunk_overlap=8)

    assert len(chunks) > 1

    # Check sequences and timestamps
    for i, chunk in enumerate(chunks):
        assert chunk.sequence == i
        assert chunk.start_time < chunk.end_time
        assert chunk.token_count > 0

    # First chunk starts at 0.0
    assert chunks[0].start_time == 0.0

    # Final chunk ends at 20.0
    assert chunks[-1].end_time == 20.0


def test_chunk_transcript_empty():
    assert chunk_transcript([], video_id="empty-vid") == []
    assert chunk_transcript([{"start_time": 0.0, "end_time": 2.0, "text": "   "}], video_id="empty-vid") == []


def test_chunk_transcript_very_long_segment():
    # A single segment that contains many tokens
    long_text = " ".join([f"Word{i} explaining machine learning in great detail" for i in range(50)])
    segment = {"start_time": 10.0, "end_time": 110.0, "text": long_text}

    # Set chunk_size small so it must be split
    chunks = chunk_transcript([segment], video_id="vid-long", chunk_size=20, chunk_overlap=5)

    assert len(chunks) > 1
    # Check that interpolated timestamps remain monotonic and within bounds
    assert chunks[0].start_time >= 10.0
    assert chunks[-1].end_time <= 110.0
    for i in range(len(chunks) - 1):
        assert chunks[i].start_time <= chunks[i + 1].start_time
        assert chunks[i].sequence == i


def test_chunk_transcript_deterministic_ids():
    segments = [
        {"start_time": 0.0, "end_time": 5.0, "text": "Deterministic chunk testing."},
        {"start_time": 5.0, "end_time": 10.0, "text": "Another segment for verification."},
    ]

    chunks_run1 = chunk_transcript(segments, video_id="vid-det", chunk_size=10, chunk_overlap=2)
    chunks_run2 = chunk_transcript(segments, video_id="vid-det", chunk_size=10, chunk_overlap=2)

    assert len(chunks_run1) == len(chunks_run2)
    for c1, c2 in zip(chunks_run1, chunks_run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.sequence == c2.sequence
        assert c1.text == c2.text
