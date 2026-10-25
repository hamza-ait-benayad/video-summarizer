from fastapi import FastAPI
import uvicorn
from sqlalchemy import text
from app.database import engine
from app.models import base
from app.api.routes import video, chat

# Ensure pgvector extension exists before creating tables
try:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
except Exception as ext_err:
    print(f"Warning: Could not create vector extension automatically: {ext_err}")

base.Base.metadata.create_all(bind=engine)

from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

app = FastAPI(title="Video Summarizer & Chat with Video API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video.router)
app.include_router(chat.router)

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8004, reload=True)
