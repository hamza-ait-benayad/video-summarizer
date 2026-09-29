import enum
from datetime import datetime
import uuid
from sqlalchemy import Column, Integer, String, Enum, DateTime
from sqlalchemy.orm import relationship
from app.models.base import Base

class VideoStatus(str, enum.Enum):
    PENDING = "PENDING"
    FETCHING = "FETCHING"
    TRANSCRIBING = "TRANSCRIBING"
    SUMMARIZING = "SUMMARIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    url = Column(String, index=True)
    platform = Column(String)
    title = Column(String, nullable=True)
    duration = Column(Integer, nullable=True)
    thumbnail_url = Column(String, nullable=True)
    status = Column(Enum(VideoStatus), default=VideoStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    transcripts = relationship("Transcript", back_populates="video", cascade="all, delete-orphan")
