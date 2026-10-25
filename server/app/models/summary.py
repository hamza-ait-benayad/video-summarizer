import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from app.models.base import Base

class Summary(Base):
    __tablename__ = "summaries"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String, ForeignKey("videos.id"), index=True, unique=True)
    summary = Column(Text)
    short_summary = Column(Text)
    key_points = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="summary")
