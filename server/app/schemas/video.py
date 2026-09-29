from typing import Optional, List, Dict, Any
from pydantic import BaseModel, HttpUrl
from app.models.video import VideoStatus

class VideoCreate(BaseModel):
    url: HttpUrl

class TranscriptSegmentResponse(BaseModel):
    start_time: float
    end_time: float
    text: str
    sequence: int

    class Config:
        from_attributes = True

class TranscriptResponse(BaseModel):
    id: str
    text: str
    language: str
    provider: str
    segments: List[TranscriptSegmentResponse] = []

    class Config:
        from_attributes = True

class VideoResponse(BaseModel):
    id: str
    url: str
    platform: str
    status: VideoStatus
    title: Optional[str] = None
    duration: Optional[int] = None
    thumbnail_url: Optional[str] = None
    transcripts: List[TranscriptResponse] = []

    class Config:
        orm_mode = True
        from_attributes = True
