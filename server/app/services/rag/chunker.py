import uuid
from typing import List, Dict, Any, Union
from pydantic import BaseModel

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_AVAILABLE = False
    _ENCODER = None

from app.core.config import settings


class TranscriptChunk(BaseModel):
    chunk_id: str
    video_id: str
    start_time: float
    end_time: float
    text: str
    sequence: int
    token_count: int


def count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken (cl100k_base).
    Falls back to a whitespace-based estimation (~1.3 tokens per word) if tiktoken is unavailable.
    """
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE and _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    # Documented fallback approximation: words * 1.3
    words = text.split()
    return max(1, int(len(words) * 1.3))


class NormalizedSegment(BaseModel):
    start_time: float
    end_time: float
    text: str
    token_count: int = 0


def normalize_segment(seg: Union[Dict[str, Any], Any]) -> NormalizedSegment:
    """Normalize input segment dictionary or ORM object into a standard dataclass."""
    if isinstance(seg, dict):
        start = float(seg.get("start_time", seg.get("start", 0.0)))
        end = seg.get("end_time")
        if end is None:
            duration = seg.get("duration", 0.0)
            end = start + float(duration)
        else:
            end = float(end)
        text = str(seg.get("text", "")).strip()
    else:
        start = float(getattr(seg, "start_time", getattr(seg, "start", 0.0)))
        end = getattr(seg, "end_time", None)
        if end is None:
            duration = getattr(seg, "duration", 0.0)
            end = start + float(duration)
        else:
            end = float(end)
        text = str(getattr(seg, "text", "")).strip()

    tokens = count_tokens(text)
    return NormalizedSegment(start_time=start, end_time=max(start, end), text=text, token_count=tokens)


def split_long_segment(seg: NormalizedSegment, max_tokens: int) -> List[NormalizedSegment]:
    """
    Split an unusually long segment into smaller parts while preserving/interpolating timestamps.
    """
    if seg.token_count <= max_tokens:
        return [seg]

    # Split by sentences or spaces
    words = seg.text.split()
    if not words:
        return []

    parts = []
    current_words: List[str] = []
    total_words = len(words)
    total_duration = max(0.1, seg.end_time - seg.start_time)

    start_idx = 0
    for i, word in enumerate(words):
        current_words.append(word)
        candidate_text = " ".join(current_words)
        if count_tokens(candidate_text) >= max_tokens:
            end_idx = i + 1
            part_start = seg.start_time + (start_idx / total_words) * total_duration
            part_end = seg.start_time + (end_idx / total_words) * total_duration
            parts.append(NormalizedSegment(
                start_time=round(part_start, 2),
                end_time=round(part_end, 2),
                text=candidate_text,
                token_count=count_tokens(candidate_text)
            ))
            current_words = []
            start_idx = end_idx

    if current_words:
        part_start = seg.start_time + (start_idx / total_words) * total_duration
        part_end = seg.end_time
        remaining_text = " ".join(current_words)
        parts.append(NormalizedSegment(
            start_time=round(part_start, 2),
            end_time=round(part_end, 2),
            text=remaining_text,
            token_count=count_tokens(remaining_text)
        ))

    return parts


def chunk_transcript(
    segments: List[Union[Dict[str, Any], Any]],
    video_id: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[TranscriptChunk]:
    """
    Groups neighboring transcript segments into larger chunks suitable for semantic retrieval.

    - Preserves timestamps: start_time is taken from the first segment in the chunk,
      and end_time is taken from the last segment in the chunk.
    - Handles empty transcripts and unusually long individual segments.
    - Configurable chunk size and overlap in tokens.
    - Produces deterministic chunk IDs scoped to the video.
    """
    if not segments:
        return []

    # 1. Normalize and clean segments
    normalized: List[NormalizedSegment] = []
    for raw in segments:
        norm = normalize_segment(raw)
        if norm.text:  # Skip empty text
            # Handle unusually long segment
            if norm.token_count > chunk_size:
                sub_segments = split_long_segment(norm, chunk_size)
                normalized.extend(sub_segments)
            else:
                normalized.append(norm)

    if not normalized:
        return []

    chunks: List[TranscriptChunk] = []
    seq = 0
    n = len(normalized)
    start_idx = 0

    while start_idx < n:
        chunk_segs: List[NormalizedSegment] = []
        accumulated_tokens = 0
        end_idx = start_idx

        # Accumulate segments until reaching chunk_size
        while end_idx < n:
            seg = normalized[end_idx]
            # If chunk is already not empty and adding seg exceeds chunk_size, stop
            if chunk_segs and (accumulated_tokens + seg.token_count > chunk_size):
                break
            chunk_segs.append(seg)
            accumulated_tokens += seg.token_count
            end_idx += 1

        if not chunk_segs:
            # Safeguard to ensure progress
            chunk_segs.append(normalized[start_idx])
            end_idx = start_idx + 1

        combined_text = " ".join(s.text for s in chunk_segs)
        chunk_start_time = chunk_segs[0].start_time
        chunk_end_time = chunk_segs[-1].end_time

        # Generate deterministic UUID based on video_id and sequence number
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"video:{video_id}:chunk:{seq}"))

        chunk = TranscriptChunk(
            chunk_id=chunk_id,
            video_id=video_id,
            start_time=round(chunk_start_time, 2),
            end_time=round(chunk_end_time, 2),
            text=combined_text,
            sequence=seq,
            token_count=count_tokens(combined_text)
        )
        chunks.append(chunk)
        seq += 1

        if end_idx >= n:
            break

        # Calculate next start_idx using chunk_overlap tokens
        if chunk_overlap > 0:
            overlap_tokens = 0
            new_start = end_idx
            # Walk backwards from end_idx to include overlap
            for back_idx in range(end_idx - 1, start_idx, -1):
                overlap_tokens += normalized[back_idx].token_count
                if overlap_tokens <= chunk_overlap:
                    new_start = back_idx
                else:
                    break
            # Ensure progress: new_start must be strictly greater than start_idx
            start_idx = max(start_idx + 1, new_start)
        else:
            start_idx = end_idx

    return chunks
