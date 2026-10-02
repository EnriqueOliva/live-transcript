from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from whisper_transcriber.audio.quiet import find_quiet_point
from whisper_transcriber.audio.timeline import SAMPLE_RATE, samples_to_seconds, seconds_to_samples

WINDOW_EDGE_SECONDS = 0.2
TAIL_GUARD_SECONDS = 0.4
CLEAR_GAP_SECONDS = 0.2
LEAN_BEFORE_SECONDS = 0.15
LEAN_AFTER_SECONDS = 0.0


@dataclass(frozen=True)
class TimedWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Handoff:
    committed: list[TimedWord]
    boundary_sample: int

    @property
    def text(self) -> str:
        return "".join(word.text for word in self.committed).strip()


def choose_handoff(
    words: list[TimedWord],
    cut_sample: int,
    next_start_sample: int,
    audio: np.ndarray,
    audio_start_sample: int,
) -> Handoff:
    if not words:
        return Handoff(committed=[], boundary_sample=next_start_sample)
    window_start = samples_to_seconds(next_start_sample) + WINDOW_EDGE_SECONDS
    window_end = samples_to_seconds(cut_sample) - TAIL_GUARD_SECONDS
    best_index: int | None = None
    best_gap = 0.0
    for index in range(len(words) - 1):
        midpoint = (words[index].end + words[index + 1].start) / 2
        gap = words[index + 1].start - words[index].end
        if window_start <= midpoint <= window_end and (best_index is None or gap >= best_gap):
            best_index = index
            best_gap = gap

    if best_index is not None:
        midpoint = (words[best_index].end + words[best_index + 1].start) / 2
        boundary_seconds = midpoint
        if best_gap < CLEAR_GAP_SECONDS:
            refined = find_quiet_point(
                audio,
                audio_start_sample,
                seconds_to_samples(midpoint - LEAN_BEFORE_SECONDS),
                seconds_to_samples(midpoint + LEAN_AFTER_SECONDS),
            )
            if refined is not None:
                boundary_seconds = refined / SAMPLE_RATE
        boundary = _clamp(seconds_to_samples(boundary_seconds), next_start_sample, cut_sample)
        return Handoff(committed=words[: best_index + 1], boundary_sample=boundary)
    elif all(word.end <= window_start for word in words):
        return Handoff(committed=list(words), boundary_sample=next_start_sample)
    else:
        spanning_index = next(index for index, word in enumerate(words) if word.end > window_start)
        spanning = words[spanning_index]
        if spanning.start - LEAN_BEFORE_SECONDS >= samples_to_seconds(next_start_sample):
            boundary = _clamp(
                seconds_to_samples(spanning.start - LEAN_BEFORE_SECONDS), next_start_sample, cut_sample,
            )
            return Handoff(committed=words[:spanning_index], boundary_sample=boundary)
        else:
            return Handoff(committed=words[: spanning_index + 1], boundary_sample=next_start_sample)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))
