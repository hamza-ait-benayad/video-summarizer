from datetime import datetime
import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, Float, Integer, Text
from sqlalchemy.orm import relationship
from app.models.base import Base

class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String, ForeignKey("videos.id"), index=True)
    text = Column(Text)
    language = Column(String, default="en")
    provider = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="transcripts")
    segments = relationship("TranscriptSegment", back_populates="transcript", cascade="all, delete-orphan")

class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    transcript_id = Column(String, ForeignKey("transcripts.id"), index=True)
    start_time = Column(Float)
    end_time = Column(Float)
    text = Column(Text)
    sequence = Column(Integer)

    transcript = relationship("Transcript", back_populates="segments")
