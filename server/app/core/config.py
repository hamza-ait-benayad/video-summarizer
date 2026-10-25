from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+psycopg2://vidmind:vidmind@localhost/vidmind"
    GEMINI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None

    # Embedding settings
    EMBEDDING_PROVIDER: str = "gemini"  # "gemini" or "deterministic"
    EMBEDDING_MODEL: str = "gemini-embedding-001"  # "gemini-embedding-001", "gemini-embedding-2", or "gemini-embedding-2-preview"
    EMBEDDING_DIMENSION: int = 768
    
    # Chunking settings
    CHUNK_SIZE: int = 500  # Target tokens per chunk
    CHUNK_OVERLAP: int = 50  # Overlap tokens between adjacent chunks
    
    # Retrieval settings
    RETRIEVAL_LIMIT: int = 5
    RETRIEVAL_THRESHOLD: float = 0.0  # Cosine similarity minimum threshold (default 0.0 for non-negative similarity)
    
    # LLM generation settings
    LLM_MODEL: str = "gemini-3.8-flash"

    ALLOWED_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
