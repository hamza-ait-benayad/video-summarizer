from typing import Optional, List
from pydantic import BaseModel, HttpUrl, ConfigDict

class VideoCreate(BaseModel):
    url: HttpUrl

class VideoCreateResponse(BaseModel):
    id: str
    status: str

    model_config = ConfigDict(from_attributes=True)

class VideoDetail(BaseModel):
    title: Optional[str] = None
    platform: str
    thumbnail_url: Optional[str] = None
    duration_seconds: Optional[int] = None

class VideoDetailResponse(BaseModel):
    id: str
    status: str
    video: VideoDetail

class VideoSummaryResponse(BaseModel):
    video_id: str
    summary: str
    short_summary: str
    key_points: List[str]

class TranscriptSegmentResponse(BaseModel):
    start_time: float
    end_time: float
    text: str

    model_config = ConfigDict(from_attributes=True)

class VideoTranscriptResponse(BaseModel):
    video_id: str
    language: str
    segments: List[TranscriptSegmentResponse] = []
