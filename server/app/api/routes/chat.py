import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.video import Video, VideoStatus
from app.models.transcript import Transcript
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag.chat_service import ChatService
from app.services.rag.vector_store import VectorStore
from app.services.rag.indexer import index_video_transcript

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/videos", tags=["chat"])
chat_service = ChatService()
vector_store = VectorStore()


@router.post("/{video_id}/chat", response_model=ChatResponse)
def chat_with_video(
    video_id: str,
    chat_in: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Chat with video using Retrieval-Augmented Generation (RAG).
    Restricted strictly to the requested video ID.
    """
    # 1. Validate question
    cleaned_question = chat_in.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or contain only whitespace.",
        )

    # 2. Validate video existence
    db_video = db.query(Video).filter(Video.id == video_id).first()
    if not db_video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found.",
        )

    # 3. Check video processing status
    if db_video.status in (
        VideoStatus.PENDING,
        VideoStatus.FETCHING,
        VideoStatus.TRANSCRIBING,
        VideoStatus.SUMMARIZING,
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Video is still processing. Please try again once processing is complete.",
        )

    if db_video.status == VideoStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video processing failed. Transcript is not available.",
        )

    # 4. Check transcript availability
    transcript = db.query(Transcript).filter(Transcript.video_id == video_id).first()
    if not transcript or not transcript.text or not transcript.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video does not have an available transcript.",
        )

    # 5. Check vector index, auto-index if unindexed
    if not vector_store.is_video_indexed(db, video_id):
        try:
            indexed_count = index_video_transcript(db=db, video_id=video_id)
            if indexed_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Transcript does not contain enough content to index.",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error auto-indexing video {video_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to index video transcript for chat.",
            )

    # 6. Execute RAG chat workflow
    try:
        result = chat_service.answer_question(
            db=db,
            video_id=video_id,
            question=cleaned_question,
        )
        return result
    except HTTPException:
        raise
    except ValueError as ve:
        logger.warning(f"Validation error during chat: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except TimeoutError:
        logger.error(f"Timeout during LLM/Embedding request for video {video_id}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The AI service timed out while generating an answer.",
        )
    except Exception as e:
        logger.error(f"Unexpected error during chat with video {video_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"An error occurred while communicating with the AI service: {str(e)}",
        )
