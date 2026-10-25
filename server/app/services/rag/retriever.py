import logging
from typing import List, Optional
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.rag.vector_store import VectorStore
from app.services.rag.embeddings import BaseEmbeddingProvider, get_embedding_provider

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    chunk_id: str
    video_id: str
    start_time: float
    end_time: float
    text: str
    sequence: int
    similarity: float

    model_config = ConfigDict(from_attributes=True)


class Retriever:
    """
    Coordinates semantic retrieval: embeds user question, queries vector store,
    and enforces video-scoped filtering and relevance thresholds.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.embedding_provider = embedding_provider or get_embedding_provider()

    def retrieve(
        self,
        db: Session,
        video_id: str,
        question: str,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant transcript chunks for the given video and question.
        """
        if not video_id or not video_id.strip():
            raise ValueError("video_id must not be empty.")

        if not question or not question.strip():
            return []

        retrieval_limit = limit if limit is not None else settings.RETRIEVAL_LIMIT
        relevance_threshold = threshold if threshold is not None else settings.RETRIEVAL_THRESHOLD

        # 1. Generate query embedding
        query_vector = self.embedding_provider.embed_query(question.strip())

        # 2. Search vector store (strictly isolated to video_id)
        raw_results = self.vector_store.search_similar_chunks(
            db=db,
            video_id=video_id,
            query_vector=query_vector,
            limit=retrieval_limit,
            relevance_threshold=relevance_threshold,
        )

        # 3. Format and deduplicate chunks
        retrieved: List[RetrievedChunk] = []
        seen_ids = set()

        for chunk_model, score in raw_results:
            if chunk_model.id not in seen_ids:
                seen_ids.add(chunk_model.id)
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=chunk_model.id,
                        video_id=chunk_model.video_id,
                        start_time=chunk_model.start_time,
                        end_time=chunk_model.end_time,
                        text=chunk_model.text,
                        sequence=chunk_model.sequence,
                        similarity=round(score, 4),
                    )
                )

        logger.info(
            f"Retrieved {len(retrieved)} chunks for video {video_id} (threshold={relevance_threshold}, limit={retrieval_limit})"
        )
        return retrieved
