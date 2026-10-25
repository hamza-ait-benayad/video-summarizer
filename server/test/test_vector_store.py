import uuid
import pytest
from app.database import SessionLocal
from app.models.video import Video, VideoStatus
from app.models.chunk import VideoChunk
from app.services.rag.chunker import TranscriptChunk
from app.services.rag.embeddings import DeterministicHashEmbeddingProvider
from app.services.rag.vector_store import VectorStore
from app.services.rag.retriever import Retriever


@pytest.fixture
def db_session():
    """Provides a transactional database session for tests and cleans up created test records."""
    db = SessionLocal()
    created_video_ids = []

    yield db, created_video_ids

    # Cleanup after test
    try:
        for vid in created_video_ids:
            db.query(VideoChunk).filter(VideoChunk.video_id == vid).delete()
            db.query(Video).filter(Video.id == vid).delete()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def test_vector_store_indexing_and_isolation(db_session):
    db, created_video_ids = db_session
    vector_store = VectorStore()
    provider = DeterministicHashEmbeddingProvider(dimension=768)

    vid_a = f"test-video-a-{uuid.uuid4()}"
    vid_b = f"test-video-b-{uuid.uuid4()}"
    created_video_ids.extend([vid_a, vid_b])

    # Insert test videos into DB
    db_v_a = Video(id=vid_a, url="https://example.com/a", platform="youtube", status=VideoStatus.COMPLETED)
    db_v_b = Video(id=vid_b, url="https://example.com/b", platform="youtube", status=VideoStatus.COMPLETED)
    db.add_all([db_v_a, db_v_b])
    db.commit()

    # Create chunks for Video A
    chunks_a = [
        TranscriptChunk(
            chunk_id=str(uuid.uuid4()),
            video_id=vid_a,
            start_time=0.0,
            end_time=10.0,
            text="Quantum computing utilizes qubits and superposition to process calculations.",
            sequence=0,
            token_count=10,
        ),
        TranscriptChunk(
            chunk_id=str(uuid.uuid4()),
            video_id=vid_a,
            start_time=10.0,
            end_time=20.0,
            text="Entanglement in quantum mechanics enables quantum teleportation protocols.",
            sequence=1,
            token_count=10,
        ),
    ]
    embeddings_a = provider.embed_texts([c.text for c in chunks_a])
    vector_store.store_chunks(db, vid_a, chunks_a, embeddings_a, provider.model_name)

    # Create chunks for Video B
    chunks_b = [
        TranscriptChunk(
            chunk_id=str(uuid.uuid4()),
            video_id=vid_b,
            start_time=0.0,
            end_time=15.0,
            text="Baking sourdough bread requires water, flour, salt, and active wild yeast starter.",
            sequence=0,
            token_count=12,
        ),
    ]
    embeddings_b = provider.embed_texts([c.text for c in chunks_b])
    vector_store.store_chunks(db, vid_b, chunks_b, embeddings_b, provider.model_name)

    # 1. Verify counts
    assert vector_store.get_chunk_count(db, vid_a) == 2
    assert vector_store.get_chunk_count(db, vid_b) == 1
    assert vector_store.is_video_indexed(db, vid_a, provider.model_name)
    assert vector_store.is_video_indexed(db, vid_b, provider.model_name)

    # 2. Test semantic search on Video A
    query_a = "Quantum computing utilizes qubits and superposition"
    query_vec_a = provider.embed_query(query_a)
    results_a = vector_store.search_similar_chunks(db, vid_a, query_vec_a, limit=5, relevance_threshold=0.0)

    assert len(results_a) > 0
    # All retrieved chunks belong to video A
    for chunk, _ in results_a:
        assert chunk.video_id == vid_a
    # Top chunk is the exact match
    top_chunk, score = results_a[0]
    assert "qubits" in top_chunk.text

    # 3. VIDEO ISOLATION TEST:
    # Querying Video B for quantum computing must NEVER return chunks from Video A!
    results_b = vector_store.search_similar_chunks(db, vid_b, query_vec_a, limit=5, relevance_threshold=0.0)
    for chunk, _ in results_b:
        assert chunk.video_id == vid_b
        assert "qubits" not in chunk.text


def test_vector_store_duplicate_prevention(db_session):
    db, created_video_ids = db_session
    vector_store = VectorStore()
    provider = DeterministicHashEmbeddingProvider(dimension=768)

    vid = f"test-video-dup-{uuid.uuid4()}"
    created_video_ids.append(vid)

    db_v = Video(id=vid, url="https://example.com/dup", platform="youtube", status=VideoStatus.COMPLETED)
    db.add(db_v)
    db.commit()

    chunks = [
        TranscriptChunk(
            chunk_id=str(uuid.uuid4()),
            video_id=vid,
            start_time=0.0,
            end_time=5.0,
            text="Initial chunk version.",
            sequence=0,
            token_count=4,
        ),
    ]
    embeddings = provider.embed_texts([c.text for c in chunks])

    # First indexing
    vector_store.store_chunks(db, vid, chunks, embeddings, provider.model_name)
    assert vector_store.get_chunk_count(db, vid) == 1

    # Second indexing with same or updated chunk
    updated_chunks = [
        TranscriptChunk(
            chunk_id=str(uuid.uuid4()),
            video_id=vid,
            start_time=0.0,
            end_time=5.0,
            text="Updated chunk version.",
            sequence=0,
            token_count=4,
        ),
    ]
    updated_embeddings = provider.embed_texts([c.text for c in updated_chunks])
    vector_store.store_chunks(db, vid, updated_chunks, updated_embeddings, provider.model_name)

    # Must still be exactly 1 chunk, avoiding duplicate indexing
    assert vector_store.get_chunk_count(db, vid) == 1
    stored = db.query(VideoChunk).filter(VideoChunk.video_id == vid).first()
    assert stored.text == "Updated chunk version."


def test_retriever_threshold_filtering(db_session):
    db, created_video_ids = db_session
    provider = DeterministicHashEmbeddingProvider(dimension=768)
    vector_store = VectorStore()
    retriever = Retriever(vector_store=vector_store, embedding_provider=provider)

    vid = f"test-video-thresh-{uuid.uuid4()}"
    created_video_ids.append(vid)

    db_v = Video(id=vid, url="https://example.com/thresh", platform="youtube", status=VideoStatus.COMPLETED)
    db.add(db_v)
    db.commit()

    chunks = [
        TranscriptChunk(
            chunk_id=str(uuid.uuid4()),
            video_id=vid,
            start_time=0.0,
            end_time=10.0,
            text="Gardening tips: plant tomatoes in full sunlight and moist soil.",
            sequence=0,
            token_count=10,
        ),
    ]
    embeddings = provider.embed_texts([c.text for c in chunks])
    vector_store.store_chunks(db, vid, chunks, embeddings, provider.model_name)

    # Completely unrelated question with high threshold
    results_empty = retriever.retrieve(db, vid, "Superconductors at room temperature", threshold=0.99)
    assert len(results_empty) == 0

    # Low threshold should retrieve
    results = retriever.retrieve(db, vid, "Tomatoes in soil", threshold=0.1)
    assert len(results) == 1
    assert results[0].video_id == vid
    assert results[0].start_time == 0.0
    assert results[0].end_time == 10.0
