from typing import List
from pydantic import BaseModel, Field, ConfigDict


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask about the video content.",
        json_schema_extra={"example": "What is supervised learning?"}
    )


class ChatSource(BaseModel):
    chunk_id: str
    start_time: float
    end_time: float
    text: str

    model_config = ConfigDict(from_attributes=True)


class ChatResponse(BaseModel):
    video_id: str
    answer: str
    sources: List[ChatSource] = []
