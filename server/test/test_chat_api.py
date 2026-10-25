import uuid
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.database import SessionLocal
from app.models.video import Video, VideoStatus
from app.models.transcript import Transcript, TranscriptSegment
from app.models.chunk import VideoChunk
from app.services.rag.embeddings import DeterministicHashEmbeddingProvider
from app.services.rag.vector_store import VectorStore
from app.services.rag.chunker import TranscriptChunk

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_default_embedding_provider():
    """Ensure all tests in this module use the deterministic embedding provider without network calls."""
    provider = DeterministicHashEmbeddingProvider(dimension=768)
    with patch("app.services.rag.embeddings.get_embedding_provider", return_value=provider):
        with patch("app.services.rag.retriever.get_embedding_provider", return_value=provider):
            with patch("app.services.rag.indexer.get_embedding_provider", return_value=provider):
                with patch("app.api.routes.chat.chat_service.retriever.embedding_provider", provider):
                    yield provider


@pytest.fixture
def test_db():
    db = SessionLocal()
    created_video_ids = []
    yield db, created_video_ids

    try:
        for vid in created_video_ids:
            db.query(VideoChunk).filter(VideoChunk.video_id == vid).delete()
            db.query(TranscriptSegment).filter(
                TranscriptSegment.transcript_id.in_(
                    db.query(Transcript.id).filter(Transcript.video_id == vid)
                )
            ).delete(synchronize_session=False)
            db.query(Transcript).filter(Transcript.video_id == vid).delete()
            db.query(Video).filter(Video.id == vid).delete()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def test_chat_unknown_video(test_db):
    response = client.post(
        f"/api/v1/videos/non-existent-id-{uuid.uuid4()}/chat",
        json={"question": "What is this video about?"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Video not found."


def test_chat_video_still_processing(test_db):
    db, created_video_ids = test_db
    vid = f"proc-vid-{uuid.uuid4()}"
    created_video_ids.append(vid)

    v = Video(id=vid, url="https://example.com/proc", platform="youtube", status=VideoStatus.TRANSCRIBING)
    db.add(v)
    db.commit()

    response = client.post(f"/api/v1/videos/{vid}/chat", json={"question": "What is this?"})
    assert response.status_code == 409
    assert "still processing" in response.json()["detail"]


def test_chat_video_failed(test_db):
    db, created_video_ids = test_db
    vid = f"fail-vid-{uuid.uuid4()}"
    created_video_ids.append(vid)

    v = Video(id=vid, url="https://example.com/fail", platform="youtube", status=VideoStatus.FAILED)
    db.add(v)
    db.commit()

    response = client.post(f"/api/v1/videos/{vid}/chat", json={"question": "What is this?"})
    assert response.status_code == 400
    assert "processing failed" in response.json()["detail"]


def test_chat_video_missing_transcript(test_db):
    db, created_video_ids = test_db
    vid = f"no-trans-vid-{uuid.uuid4()}"
    created_video_ids.append(vid)

    v = Video(id=vid, url="https://example.com/no-trans", platform="youtube", status=VideoStatus.COMPLETED)
    db.add(v)
    db.commit()

    response = client.post(f"/api/v1/videos/{vid}/chat", json={"question": "What is this?"})
    assert response.status_code == 400
    assert "transcript" in response.json()["detail"].lower()


def test_chat_empty_or_whitespace_question(test_db):
    db, created_video_ids = test_db
    vid = f"valid-vid-{uuid.uuid4()}"
    created_video_ids.append(vid)

    v = Video(id=vid, url="https://example.com/valid", platform="youtube", status=VideoStatus.COMPLETED)
    db.add(v)
    db.commit()

    response = client.post(f"/api/v1/videos/{vid}/chat", json={"question": "    "})
    assert response.status_code == 400


def test_chat_excessively_long_question(test_db):
    vid = f"vid-{uuid.uuid4()}"
    long_q = "a" * 2001
    response = client.post(f"/api/v1/videos/{vid}/chat", json={"question": long_q})
    assert response.status_code == 422


def test_chat_successful_qa_with_sources(test_db, mock_default_embedding_provider):
    db, created_video_ids = test_db
    vid = f"success-vid-{uuid.uuid4()}"
    created_video_ids.append(vid)

    # Create video and transcript
    v = Video(id=vid, url="https://example.com/success", platform="youtube", status=VideoStatus.COMPLETED)
    t = Transcript(id=str(uuid.uuid4()), video_id=vid, text="Neural networks learn features from data.", language="en")
    seg = TranscriptSegment(id=str(uuid.uuid4()), transcript_id=t.id, start_time=10.0, end_time=25.0, text="Neural networks learn features from data.", sequence=0)
    db.add_all([v, t, seg])
    db.commit()

    # Index chunks with deterministic embedding provider
    provider = mock_default_embedding_provider
    store = VectorStore()
    chunk = TranscriptChunk(
        chunk_id=str(uuid.uuid4()),
        video_id=vid,
        start_time=10.0,
        end_time=25.0,
        text="Neural networks learn features from data.",
        sequence=0,
        token_count=6,
    )
    embeddings = provider.embed_texts([chunk.text])
    store.store_chunks(db, vid, [chunk], embeddings, provider.model_name)

    mock_answer = "Neural networks learn features by processing training data."

    with patch("app.api.routes.chat.chat_service.llm_fn", return_value=mock_answer):
        response = client.post(
            f"/api/v1/videos/{vid}/chat",
            json={"question": "Neural networks learn features from data."},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == vid
    assert data["answer"] == mock_answer
    assert len(data["sources"]) == 1
    source = data["sources"][0]
    assert source["start_time"] == 10.0
    assert source["end_time"] == 25.0
    assert "Neural networks" in source["text"]


def test_chat_auto_indexing_on_the_fly(test_db, mock_default_embedding_provider):
    db, created_video_ids = test_db
    vid = f"auto-index-vid-{uuid.uuid4()}"
    created_video_ids.append(vid)

    # Video completed with transcript but no vector chunks in DB
    v = Video(id=vid, url="https://example.com/auto", platform="youtube", status=VideoStatus.COMPLETED)
    t = Transcript(id=str(uuid.uuid4()), video_id=vid, text="Reinforcement learning uses reward signals.", language="en")
    seg = TranscriptSegment(id=str(uuid.uuid4()), transcript_id=t.id, start_time=0.0, end_time=12.0, text="Reinforcement learning uses reward signals.", sequence=0)
    db.add_all([v, t, seg])
    db.commit()

    with patch("app.api.routes.chat.chat_service.llm_fn", return_value="RL uses rewards."):
        # Before request, chunks are not indexed
        store = VectorStore()
        assert not store.is_video_indexed(db, vid)

        response = client.post(
            f"/api/v1/videos/{vid}/chat",
            json={"question": "Reinforcement learning uses reward signals."},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == vid
    assert "RL uses rewards." in data["answer"]
    # Verify chunks are now persisted in DB
    assert store.is_video_indexed(db, vid)


def test_chat_no_relevant_chunks(test_db, mock_default_embedding_provider):
    db, created_video_ids = test_db
    vid = f"no-rel-vid-{uuid.uuid4()}"
    created_video_ids.append(vid)

    v = Video(id=vid, url="https://example.com/norel", platform="youtube", status=VideoStatus.COMPLETED)
    t = Transcript(id=str(uuid.uuid4()), video_id=vid, text="Content about cooking pasta.", language="en")
    seg = TranscriptSegment(id=str(uuid.uuid4()), transcript_id=t.id, start_time=0.0, end_time=10.0, text="Content about cooking pasta.", sequence=0)
    db.add_all([v, t, seg])
    db.commit()

    store = VectorStore()
    provider = mock_default_embedding_provider
    chunk = TranscriptChunk(chunk_id=str(uuid.uuid4()), video_id=vid, start_time=0.0, end_time=10.0, text="Content about cooking pasta.", sequence=0, token_count=4)
    store.store_chunks(db, vid, [chunk], provider.embed_texts([chunk.text]), provider.model_name)

    # Mock retriever to return no chunks (simulating threshold filtering)
    with patch("app.api.routes.chat.chat_service.retriever.retrieve", return_value=[]):
        response = client.post(
            f"/api/v1/videos/{vid}/chat",
            json={"question": "Explain astrophysics cosmology models."},
        )

    assert response.status_code == 200
    data = response.json()
    assert "does not contain sufficient information" in data["answer"]
    assert data["sources"] == []


def test_chat_llm_failure_handling(test_db, mock_default_embedding_provider):
    db, created_video_ids = test_db
    vid = f"llm-fail-vid-{uuid.uuid4()}"
    created_video_ids.append(vid)

    v = Video(id=vid, url="https://example.com/fail", platform="youtube", status=VideoStatus.COMPLETED)
    t = Transcript(id=str(uuid.uuid4()), video_id=vid, text="Some text.", language="en")
    seg = TranscriptSegment(id=str(uuid.uuid4()), transcript_id=t.id, start_time=0.0, end_time=5.0, text="Some text.", sequence=0)
    db.add_all([v, t, seg])
    db.commit()

    provider = mock_default_embedding_provider
    store = VectorStore()
    chunk = TranscriptChunk(chunk_id=str(uuid.uuid4()), video_id=vid, start_time=0.0, end_time=5.0, text="Some text.", sequence=0, token_count=2)
    store.store_chunks(db, vid, [chunk], provider.embed_texts([chunk.text]), provider.model_name)

    # Simulate LLM raising an exception when chunks ARE retrieved
    with patch("app.api.routes.chat.chat_service.llm_fn", side_effect=RuntimeError("Google AI unreachable")):
        response = client.post(
            f"/api/v1/videos/{vid}/chat",
            json={"question": "Some text."},
        )

    # Must return 502 without leaking secrets or stack trace
    assert response.status_code == 502
    assert "error occurred while communicating with the AI service" in response.json()["detail"]


def test_chat_video_isolation(test_db, mock_default_embedding_provider):
    db, created_video_ids = test_db
    vid_1 = f"iso-vid-1-{uuid.uuid4()}"
    vid_2 = f"iso-vid-2-{uuid.uuid4()}"
    created_video_ids.extend([vid_1, vid_2])

    v1 = Video(id=vid_1, url="https://example.com/1", platform="youtube", status=VideoStatus.COMPLETED)
    v2 = Video(id=vid_2, url="https://example.com/2", platform="youtube", status=VideoStatus.COMPLETED)
    t1 = Transcript(id=str(uuid.uuid4()), video_id=vid_1, text="Secret code for Video 1 is ALPHA.", language="en")
    t2 = Transcript(id=str(uuid.uuid4()), video_id=vid_2, text="Secret code for Video 2 is BRAVO.", language="en")
    seg1 = TranscriptSegment(id=str(uuid.uuid4()), transcript_id=t1.id, start_time=0.0, end_time=5.0, text="Secret code for Video 1 is ALPHA.", sequence=0)
    seg2 = TranscriptSegment(id=str(uuid.uuid4()), transcript_id=t2.id, start_time=0.0, end_time=5.0, text="Secret code for Video 2 is BRAVO.", sequence=0)
    db.add_all([v1, v2, t1, t2, seg1, seg2])
    db.commit()

    provider = mock_default_embedding_provider
    store = VectorStore()
    chunk1 = TranscriptChunk(chunk_id=str(uuid.uuid4()), video_id=vid_1, start_time=0.0, end_time=5.0, text=t1.text, sequence=0, token_count=7)
    chunk2 = TranscriptChunk(chunk_id=str(uuid.uuid4()), video_id=vid_2, start_time=0.0, end_time=5.0, text=t2.text, sequence=0, token_count=7)
    store.store_chunks(db, vid_1, [chunk1], provider.embed_texts([chunk1.text]), provider.model_name)
    store.store_chunks(db, vid_2, [chunk2], provider.embed_texts([chunk2.text]), provider.model_name)

    captured_contexts = []

    def fake_llm(q, ctx):
        captured_contexts.append(ctx)
        return "Answer based on context."

    with patch("app.api.routes.chat.chat_service.llm_fn", side_effect=fake_llm):
        # Ask Video 1
        res1 = client.post(f"/api/v1/videos/{vid_1}/chat", json={"question": "Secret code for Video 1 is ALPHA."})
        assert res1.status_code == 200
        assert "ALPHA" in captured_contexts[-1]
        assert "BRAVO" not in captured_contexts[-1]

        # Ask Video 2
        res2 = client.post(f"/api/v1/videos/{vid_2}/chat", json={"question": "Secret code for Video 2 is BRAVO."})
        assert res2.status_code == 200
        assert "BRAVO" in captured_contexts[-1]
        assert "ALPHA" not in captured_contexts[-1]
