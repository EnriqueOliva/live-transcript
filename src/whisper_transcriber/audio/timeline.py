from __future__ import annotations

SAMPLE_RATE = 16000
INT16_FULL_SCALE = 32768.0
INT16_MAXIMUM = 32767


def seconds_to_samples(seconds: float) -> int:
    return round(seconds * SAMPLE_RATE)


def samples_to_seconds(samples: int) -> float:
    return samples / SAMPLE_RATE
