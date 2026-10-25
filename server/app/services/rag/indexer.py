import logging
from typing import List, Optional, Any
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.transcript import Transcript, TranscriptSegment
from app.services.rag.chunker import chunk_transcript, TranscriptChunk
from app.services.rag.embeddings import BaseEmbeddingProvider, get_embedding_provider
from app.services.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def index_video_transcript(
    db: Session,
    video_id: str,
    segments: Optional[List[Any]] = None,
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    vector_store: Optional[VectorStore] = None,
    force_reindex: bool = False,
) -> int:
    """
    Indexes the transcript of a video into the vector database.
    - Validates transcript.
    - Deduplication: checks if already indexed for this video and model, avoiding duplicate embeddings.
    - Generates timestamp-preserving chunks.
    - Generates embeddings in batches.
    - Stores chunks and embeddings in pgvector atomically.
    - Returns the count of indexed chunks.
    """
    store = vector_store or VectorStore()
    provider = embedding_provider or get_embedding_provider()

    # Deduplication check: skip if already indexed with this model
    if not force_reindex and store.is_video_indexed(db, video_id, embedding_model=provider.model_name):
        logger.info(f"Video {video_id} already has vector index for {provider.model_name}. Skipping re-indexing.")
        return store.get_chunk_count(db, video_id)

    # Retrieve segments from DB if not provided
    if segments is None:
        transcript = db.query(Transcript).filter(Transcript.video_id == video_id).first()
        if not transcript:
            logger.warning(f"No transcript found for video {video_id}.")
            return 0

        segments = (
            db.query(TranscriptSegment)
            .filter(TranscriptSegment.transcript_id == transcript.id)
            .order_by(TranscriptSegment.sequence)
            .all()
        )

    if not segments:
        logger.warning(f"No transcript segments found to index for video {video_id}.")
        return 0

    # 1. Chunk transcript segments
    chunks: List[TranscriptChunk] = chunk_transcript(
        segments=segments,
        video_id=video_id,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    if not chunks:
        logger.warning(f"Zero chunks generated for video {video_id}.")
        return 0

    # 2. Generate embeddings
    texts = [c.text for c in chunks]
    embeddings = provider.embed_texts(texts)

    # 3. Atomically persist to vector database
    stored = store.store_chunks(
        db=db,
        video_id=video_id,
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=provider.model_name,
    )

    logger.info(f"Indexed {len(stored)} chunks for video {video_id}")
    return len(stored)
