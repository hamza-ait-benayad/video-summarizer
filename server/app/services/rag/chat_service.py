import logging
from typing import Dict, Any, Optional, Callable
from sqlalchemy.orm import Session

from app.services.rag.retriever import Retriever
from app.services.llm import generate_rag_answer

logger = logging.getLogger(__name__)


def format_seconds_to_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ChatService:
    """
    Coordinates semantic retrieval and LLM answer generation for a video.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.retriever = retriever or Retriever()
        self.llm_fn = llm_fn or generate_rag_answer

    def answer_question(
        self,
        db: Session,
        video_id: str,
        question: str,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks and generate a grounded answer using the LLM.
        """
        # 1. Retrieve relevant chunks scoped strictly to this video
        retrieved_chunks = self.retriever.retrieve(
            db=db,
            video_id=video_id,
            question=question,
            limit=limit,
            threshold=threshold,
        )

        # 2. Check if relevant context is available
        if not retrieved_chunks:
            return {
                "video_id": video_id,
                "answer": "The video transcript does not contain sufficient information to answer this question.",
                "sources": [],
            }

        # 3. Construct context with timestamp headers
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            start_ts = format_seconds_to_timestamp(chunk.start_time)
            end_ts = format_seconds_to_timestamp(chunk.end_time)
            context_parts.append(
                f"[Excerpt {i} | Time: {start_ts} - {end_ts} ({chunk.start_time:.1f}s - {chunk.end_time:.1f}s)]\n{chunk.text}"
            )
        context_text = "\n\n".join(context_parts)

        # 4. Invoke LLM with strict instructions
        try:
            answer = self.llm_fn(question, context_text)
        except Exception as e:
            logger.error(f"Error calling LLM provider: {e}")
            raise

        # 5. Build and return structured response
        sources = [
            {
                "chunk_id": chunk.chunk_id,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "text": chunk.text,
            }
            for chunk in retrieved_chunks
        ]

        return {
            "video_id": video_id,
            "answer": answer,
            "sources": sources,
        }
