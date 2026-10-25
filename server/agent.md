# Video Summarizer / Transcripter Server

This project is a FastAPI-based backend service designed to process videos, generate transcripts, and create AI-driven summaries.

## Core Architecture

- **Framework**: FastAPI
- **Database ORM**: SQLAlchemy
- **Background Tasks**: FastAPI's built-in `BackgroundTasks` for asynchronous processing
- **Local Server**: Uvicorn (running on port 8005)

## API Endpoints (`/api/v1/videos`)

The main logic resides in `app/api/routes/video.py`. It exposes the following endpoints:

### 1. Submit Video
- **Endpoint**: `POST /api/v1/videos`
- **Description**: Accepts a video URL, detects the platform, saves the initial video record to the database, and dispatches a background task (`process_video_background`) to process the video.
- **Returns**: `{"id": "uuid", "status": "processing"}`

### 2. Get Video Details
- **Endpoint**: `GET /api/v1/videos/{video_id}`
- **Description**: Retrieves the processing status (`PENDING`, `FETCHING`, `TRANSCRIBING`, `SUMMARIZING`, `COMPLETED`, `FAILED`) and metadata (title, platform, duration, thumbnail) for a given video ID.

### 3. Get Video Summary
- **Endpoint**: `GET /api/v1/videos/{video_id}/summary`
- **Description**: Retrieves the generated summary content (full summary, short summary, and key points) for a processed video.

### 4. Get Video Transcript
- **Endpoint**: `GET /api/v1/videos/{video_id}/transcript`
- **Description**: Returns the full transcript of a video, broken down into time-stamped segments.

### 5. Chat with Video (RAG)
- **Endpoint**: `POST /api/v1/videos/{video_id}/chat`
- **Description**: Queries video content using semantic retrieval over transcript chunks indexed in pgvector. Grounded response generated with source timestamps.
- **Request Body**: `{"question": "What is the main topic?"}`
- **Returns**: `{"video_id": "...", "answer": "...", "sources": [{"chunk_id": "...", "start_time": 0.0, "end_time": 15.0, "text": "..."}]}`

## Processing Pipeline (`process_video_background`)

When a video is submitted, the background task executes the following flow:
1. **Fetching**: Updates status to `FETCHING`. Calls `VideoService` to download/extract video metadata and raw transcript data.
2. **Transcribing & RAG Indexing**: Updates status to `TRANSCRIBING`. Saves the full text and individual transcript segments (`start_time`, `end_time`, `text`) to the database. Chunks segments token-aware and persists embeddings into the `video_chunks` pgvector table.
3. **Summarizing**: Updates status to `SUMMARIZING`. Invokes the LLM service (`app.services.llm.generate_summary`) to generate a summary, short summary, and key points from the full transcript text.
4. **Completion**: Updates status to `COMPLETED` (or `FAILED` if an exception occurs).

## Database Models

Located in `app/models/`:
- **Video**: Stores URL, platform, status, and basic metadata.
- **Transcript & TranscriptSegment**: Stores the raw text and sequential time-stamped segments for a video.
- **Summary**: Stores the LLM-generated summaries (short summary, full summary, key points) associated with a video.
- **VideoChunk**: Stores timestamp-preserving transcript chunks and 768-dimensional pgvector embeddings.

## Services

Located in `app/services/`:
- **Video Detector/Service** (`app/services/video/`): Detects the platform (e.g., YouTube) and extracts video information/transcripts.
- **LLM Service** (`app/services/llm.py`): Interfaces with a language model to analyze transcripts and generate summaries/grounded answers.
- **RAG Services** (`app/services/rag/`):
  - `chunker.py`: Token-aware segment chunking preserving start/end timestamps.
  - `embeddings.py`: Generates chunk and query embeddings with provider abstractions (Gemini, Deterministic).
  - `vector_store.py`: Vector operations, atomic persistence, and cosine similarity search using pgvector.
  - `retriever.py`: Video-scoped semantic retriever enforcing relevance thresholds and isolation.
  - `chat_service.py`: Coordinates retriever and LLM grounded question answering.
  - `indexer.py`: Deduplicated indexing pipeline for video transcripts.
