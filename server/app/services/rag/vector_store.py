import logging
from typing import List, Tuple, Optional
from sqlalchemy.orm import Session
from app.models.chunk import VideoChunk
from app.services.rag.chunker import TranscriptChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Encapsulates vector database operations and similarity search using pgvector."""

    def is_video_indexed(self, db: Session, video_id: str, embedding_model: Optional[str] = None) -> bool:
        """
        Check if chunks for this video are already indexed in the vector database.
        """
        query = db.query(VideoChunk).filter(VideoChunk.video_id == video_id)
        if embedding_model:
            query = query.filter(VideoChunk.embedding_model == embedding_model)
        count = query.count()
        return count > 0

    def get_chunk_count(self, db: Session, video_id: str) -> int:
        """Return the number of stored chunks for a video."""
        return db.query(VideoChunk).filter(VideoChunk.video_id == video_id).count()

    def delete_video_chunks(self, db: Session, video_id: str) -> int:
        """Delete all chunks associated with a video."""
        deleted = db.query(VideoChunk).filter(VideoChunk.video_id == video_id).delete()
        db.commit()
        return deleted

    def store_chunks(
        self,
        db: Session,
        video_id: str,
        chunks: List[TranscriptChunk],
        embeddings: List[List[float]],
        embedding_model: str,
    ) -> List[VideoChunk]:
        """
        Atomically store chunks and their vector embeddings in the database.
        Removes any previous chunks for the video to ensure clean state and prevent duplication.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        if not chunks:
            return []

        try:
            # Clean up prior records to prevent orphan/partial states
            db.query(VideoChunk).filter(VideoChunk.video_id == video_id).delete()

            db_chunks: List[VideoChunk] = []
            for chunk, emb in zip(chunks, embeddings):
                db_chunk = VideoChunk(
                    id=chunk.chunk_id,
                    video_id=video_id,
                    text=chunk.text,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    sequence=chunk.sequence,
                    embedding=emb,
                    embedding_model=embedding_model,
                )
                db_chunks.append(db_chunk)

            db.add_all(db_chunks)
            db.commit()
            logger.info(f"Successfully stored {len(db_chunks)} chunks for video {video_id}")
            return db_chunks

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to store chunks for video {video_id}: {e}")
            raise

    def search_similar_chunks(
        self,
        db: Session,
        video_id: str,
        query_vector: List[float],
        limit: int = 5,
        relevance_threshold: float = 0.0,
    ) -> List[Tuple[VideoChunk, float]]:
        """
        Perform semantic similarity search restricted to the specified video.

        - Scoped strictly to video_id.
        - Computes cosine distance and converts to similarity score (1 - distance).
        - Filters by relevance_threshold.
        - Returns list of (VideoChunk, similarity_score) tuples ordered by similarity descending.
        """
        distance_expr = VideoChunk.embedding.cosine_distance(query_vector)
        similarity_expr = 1.0 - distance_expr

        # Mandatory video_id filter enforces complete video isolation
        query = (
            db.query(VideoChunk, similarity_expr.label("similarity"))
            .filter(VideoChunk.video_id == video_id)
            .order_by(distance_expr.asc())
        )

        rows = query.all()
        matched: List[Tuple[VideoChunk, float]] = []

        for chunk, sim in rows:
            score = float(sim) if sim is not None else 0.0
            if score >= relevance_threshold:
                matched.append((chunk, score))
            if len(matched) >= limit:
                break

        return matched
