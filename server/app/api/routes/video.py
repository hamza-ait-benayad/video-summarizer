from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models.video import Video, VideoStatus
from app.models.transcript import Transcript, TranscriptSegment
from app.models.summary import Summary
from app.schemas.video import (
    VideoCreate, VideoCreateResponse, VideoDetailResponse,
    VideoTranscriptResponse, VideoSummaryResponse
)
from app.services.video.detector import detect_platform
from app.services.video.service import VideoService
from app.services.video.exceptions import (
    VideoProcessingError,
    TranscriptUnavailableError,
)
from app.services.llm import generate_summary
from app.services.rag.indexer import index_video_transcript

router = APIRouter(prefix="/api/v1/videos", tags=["videos"])
video_service = VideoService()


def process_video_background(video_id: str, url_str: str):
    db = SessionLocal()
    try:
        db_video = db.query(Video).filter(Video.id == video_id).first()
        if not db_video:
            return

        db_video.status = VideoStatus.FETCHING
        db.commit()

        try:
            adapter = video_service.get_adapter(url_str)

            # 1. Metadata — persisted immediately so it survives transcript failure
            try:
                metadata = adapter.get_metadata(url_str)
                db_video.title = metadata.title
                db_video.duration = metadata.duration_seconds
                db_video.thumbnail_url = metadata.thumbnail_url
                db.commit()
            except VideoProcessingError as meta_err:
                print(f"Metadata fetch warning for {video_id} [{meta_err.code}]: {meta_err.message}")

            # 2. Transcript — independent of metadata
            from app.services.video.normaliser import normalise_transcript
            raw_transcript = adapter.get_transcript(url_str)
            transcript = normalise_transcript(raw_transcript)

            if not transcript.segments:
                raise TranscriptUnavailableError(
                    f"No usable transcript segments for video {video_id}"
                )

            if db_video.duration is None and transcript.segments:
                db_video.duration = int(transcript.segments[-1].end)

            db_video.status = VideoStatus.TRANSCRIBING
            db.commit()

            full_text = " ".join(seg.text for seg in transcript.segments)

            db_transcript = Transcript(
                video_id=db_video.id,
                text=full_text,
                language=transcript.language or "unknown",
                provider=db_video.platform,
            )
            db.add(db_transcript)
            db.flush()

            for i, seg in enumerate(transcript.segments):
                db_seg = TranscriptSegment(
                    transcript_id=db_transcript.id,
                    start_time=seg.start,
                    end_time=seg.end,
                    text=seg.text,
                    sequence=i,
                )
                db.add(db_seg)
            db.commit()

            # 3. Vector chunks + embeddings for RAG
            try:
                indexed_chunks = index_video_transcript(db=db, video_id=db_video.id, force_reindex=True)
                print(f"RAG Indexing: {indexed_chunks} chunks indexed for video {db_video.id}")
            except Exception as index_err:
                print(f"RAG Indexing error for {video_id}: {index_err}")

            # 4. LLM summarisation
            db_video.status = VideoStatus.SUMMARIZING
            db.commit()

            try:
                summary_data = generate_summary(full_text)
                db_summary = Summary(
                    video_id=db_video.id,
                    summary=summary_data.get("summary", ""),
                    short_summary=summary_data.get("short_summary", ""),
                    key_points=summary_data.get("key_points", []),
                )
                db.add(db_summary)
                db.commit()
            except Exception as llm_error:
                print(f"LLM Summarisation failed for {video_id}: {llm_error}")

            db_video.status = VideoStatus.COMPLETED
            db.commit()

        except VideoProcessingError as vpe:
            print(f"Video {video_id} failed [{vpe.code}]: {vpe.message}")
            db_video.status = VideoStatus.FAILED
            db.commit()
        except Exception as exc:
            print(f"Video {video_id} unexpected error: {exc}")
            db_video.status = VideoStatus.FAILED
            db.commit()
    finally:
        db.close()


@router.post("", response_model=VideoCreateResponse)
def create_video(video_in: VideoCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    url_str = str(video_in.url)
    platform = detect_platform(url_str)

    db_video = Video(
        url=url_str,
        platform=platform,
        status=VideoStatus.PENDING
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)

    background_tasks.add_task(process_video_background, db_video.id, url_str)

    return {"id": db_video.id, "status": "processing"}


@router.get("/{video_id}", response_model=VideoDetailResponse)
def get_video(video_id: str, db: Session = Depends(get_db)):
    db_video = db.query(Video).filter(Video.id == video_id).first()
    if db_video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    return {
        "id": db_video.id,
        "status": db_video.status.value.lower() if db_video.status else "pending",
        "video": {
            "title": db_video.title,
            "platform": db_video.platform,
            "thumbnail_url": db_video.thumbnail_url,
            "duration_seconds": db_video.duration,
        }
    }


@router.get("/{video_id}/summary", response_model=VideoSummaryResponse)
def get_video_summary(video_id: str, db: Session = Depends(get_db)):
    db_video = db.query(Video).filter(Video.id == video_id).first()
    if db_video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    db_summary = db.query(Summary).filter(Summary.video_id == video_id).first()
    if db_summary is None:
        raise HTTPException(
            status_code=404,
            detail="Summary not available yet. The video may still be processing."
        )

    return {
        "video_id": video_id,
        "summary": db_summary.summary,
        "short_summary": db_summary.short_summary,
        "key_points": db_summary.key_points,
    }


@router.get("/{video_id}/transcript", response_model=VideoTranscriptResponse)
def get_video_transcript(video_id: str, db: Session = Depends(get_db)):
    db_video = db.query(Video).filter(Video.id == video_id).first()
    if db_video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    transcript = db.query(Transcript).filter(Transcript.video_id == video_id).first()
    if not transcript:
        return {"video_id": video_id, "language": "unknown", "segments": []}

    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.transcript_id == transcript.id)
        .order_by(TranscriptSegment.sequence)
        .all()
    )

    return {
        "video_id": video_id,
        "language": transcript.language,
        "segments": segments,
    }
