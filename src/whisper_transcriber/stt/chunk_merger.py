from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEDUP_TOLERANCE = 0.5
BOUNDARY_THRESHOLD = 1.0


@dataclass
class MergedSegment:
    start: float
    end: float
    text: str


class ChunkMerger:
    def __init__(self) -> None:
        self.last_end_time: float = 0.0
        self._pending: MergedSegment | None = None

    def merge(
        self,
        segments: list[Any],
        chunk_offset: float,
        safe_duration: float | None = None,
        chunk_duration: float = 30.0,
    ) -> list[MergedSegment]:
        accepted: list[MergedSegment] = []
        is_flush = safe_duration is None

        if self._pending is not None:
            self._resolve_pending(segments, chunk_offset, accepted)

        for segment in segments:
            abs_start = segment.start + chunk_offset
            abs_end = segment.end + chunk_offset
            text = segment.text.strip()

            if not text:
                continue

            if (
                hasattr(segment, "no_speech_prob")
                and hasattr(segment, "avg_logprob")
                and segment.no_speech_prob > 0.6
                and segment.avg_logprob < -0.8
            ):
                logger.debug("Filtered low-confidence: '%s'", text[:60])
                continue

            if not is_flush and segment.start >= safe_duration:
                continue

            if abs_end <= self.last_end_time + 0.1:
                continue

            if abs_start < self.last_end_time - DEDUP_TOLERANCE:
                continue

            if not is_flush and segment.end >= chunk_duration - BOUNDARY_THRESHOLD:
                self._pending = MergedSegment(start=abs_start, end=abs_end, text=text)
                logger.debug("Held back boundary segment: '%s' [%.1f-%.1f]", text[:40], abs_start, abs_end)
                continue

            accepted.append(MergedSegment(start=abs_start, end=abs_end, text=text))
            self.last_end_time = abs_end

        if is_flush and self._pending is not None:
            accepted.insert(0, self._pending)
            self.last_end_time = max(self.last_end_time, self._pending.end)
            self._pending = None

        return accepted

    def _resolve_pending(
        self, segments: list[Any], chunk_offset: float, accepted: list[MergedSegment],
    ) -> None:
        pending = self._pending
        self._pending = None
        if pending is None:
            return

        continuation = self._find_continuation(pending, segments, chunk_offset)

        if continuation is not None and continuation.end > pending.end + 0.5:
            merged_text = _text_merge(pending.text, continuation.text)
            result = MergedSegment(start=pending.start, end=continuation.end, text=merged_text)
            logger.info("Merged boundary segment: '%s' [%.1f-%.1f]", merged_text[:60], result.start, result.end)
        else:
            result = pending
            logger.debug("Committed held segment as-is: '%s'", pending.text[:40])

        accepted.append(result)
        self.last_end_time = result.end

    @staticmethod
    def _find_continuation(
        pending: MergedSegment, segments: list[Any], chunk_offset: float,
    ) -> MergedSegment | None:
        best = None
        for seg in segments:
            abs_start = seg.start + chunk_offset
            abs_end = seg.end + chunk_offset
            text = seg.text.strip()
            if not text:
                continue
            if (
                abs_start <= pending.end + 1.0
                and abs_start >= pending.start - 1.0
                and (best is None or abs_end > best.end)
            ):
                best = MergedSegment(start=abs_start, end=abs_end, text=text)
        return best

    def reset(self) -> None:
        self.last_end_time = 0.0
        self._pending = None


def _text_merge(text_a: str, text_b: str) -> str:
    words_a = text_a.split()
    words_b = text_b.split()
    for length in range(min(len(words_a), len(words_b)), 0, -1):
        if words_a[-length:] == words_b[:length]:
            return " ".join(words_a + words_b[length:])
    return text_a + " " + text_b
