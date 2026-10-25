"""
Transcript normalisation service.

Takes a raw Transcript (as returned by an adapter) and produces a clean,
chronologically ordered Transcript. Normalisation is conservative:

- Whitespace is collapsed and stripped.
- Segments with no usable text are dropped.
- Segments with negative or non-finite timestamps are dropped.
- Segments are sorted by start time.
- When end <= start the segment is extended by its duration is set to
  0.5 s rather than dropped so that single-word captions survive.
- Punctuation and original language are preserved.
- Segment text is not otherwise rewritten.
"""

from __future__ import annotations

import math
import re

from app.services.video.models import Transcript, TranscriptSegment

# Collapse any sequence of whitespace characters (including non-breaking
# spaces, tabs, newlines) to a single ASCII space.
_WHITESPACE_RE = re.compile(r"\s+")

# Minimum duration assigned to a segment whose end timestamp is not
# greater than its start.
_MIN_SEGMENT_DURATION = 0.5


def normalise_transcript(transcript: Transcript) -> Transcript:
    """
    Return a new Transcript with normalised segments.

    The original Transcript is not mutated.
    """
    normalised: list[TranscriptSegment] = []

    for seg in transcript.segments:
        # ------------------------------------------------------------------
        # 1. Validate timestamps: must be finite, non-negative numbers.
        # ------------------------------------------------------------------
        start = seg.start
        end = seg.end

        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        if start < 0 or end < 0:
            continue

        # ------------------------------------------------------------------
        # 2. Fix inverted or zero-length timestamps.
        # ------------------------------------------------------------------
        if end <= start:
            end = start + _MIN_SEGMENT_DURATION

        # ------------------------------------------------------------------
        # 3. Normalise text.
        # ------------------------------------------------------------------
        text = _WHITESPACE_RE.sub(" ", seg.text).strip()
        if not text:
            continue

        normalised.append(
            TranscriptSegment(start=start, end=end, text=text)
        )

    # ------------------------------------------------------------------
    # 4. Sort by start time to guarantee chronological order.
    #    Stable sort preserves the provider's ordering for ties.
    # ------------------------------------------------------------------
    normalised.sort(key=lambda s: s.start)

    return Transcript(
        segments=normalised,
        language=transcript.language,
        source_type=transcript.source_type,
    )
