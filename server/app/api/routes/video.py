from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import video
from app.models.transcript import Transcript, TranscriptSegment
from app.schemas import VideoCreate, VideoResponse
from app.services.video.detector import detect_platform
from app.services.video.service import VideoService

router = APIRouter(prefix="/api/v1/videos", tags=["videos"])
video_service = VideoService()

@router.post("", response_model=VideoResponse)
def create_video(video_in: VideoCreate, db: Session = Depends(get_db)):
    url_str = str(video_in.url)
    platform = detect_platform(url_str)
    
    db_video = video.Video(
        url=url_str,
        platform=platform
    )
    db.add(db_video)
    db.commit()
    
    result = video_service.process_video(url_str)
    
    metadata = result.get("metadata", {})
    transcript_data = result.get("transcript", [])
    
    db_video.title = metadata.get("title")
    db_video.duration = metadata.get("duration")
    db_video.thumbnail_url = metadata.get("thumbnail")
    db_video.status = video.VideoStatus.COMPLETED
    
    if transcript_data:
        full_text = " ".join([seg["text"] for seg in transcript_data])
        db_transcript = Transcript(
            video_id=db_video.id,
            text=full_text,
            language="en",
            provider=db_video.platform
        )
        db.add(db_transcript)
        db.flush()
        
        for i, seg in enumerate(transcript_data):
            db_seg = TranscriptSegment(
                transcript_id=db_transcript.id,
                start_time=seg["start"],
                end_time=seg["start"] + seg.get("duration", 0.0),
                text=seg["text"],
                sequence=i
            )
            db.add(db_seg)

    db.commit()
    db.refresh(db_video)
    
    return db_video

@router.get("/{video_id}", response_model=VideoResponse)
def get_video(video_id: str, db: Session = Depends(get_db)):
    db_video = db.query(video.Video).filter(video.Video.id == video_id).first()
    if db_video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return db_video

    
