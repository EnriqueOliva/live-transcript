from __future__ import annotations

import logging
from math import gcd

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

TARGET_RATE = 16000
NORMALIZE_DIVISOR = 32768.0


def convert_to_whisper_format(
    raw_bytes: bytes,
    source_rate: int,
    source_channels: int,
) -> np.ndarray:
    pcm = np.frombuffer(raw_bytes, dtype=np.int16)
    audio = pcm.astype(np.float32) / NORMALIZE_DIVISOR

    if source_channels > 1:
        audio = audio.reshape(-1, source_channels).mean(axis=1)

    if source_rate != TARGET_RATE:
        divisor = gcd(source_rate, TARGET_RATE)
        up = TARGET_RATE // divisor
        down = source_rate // divisor
        audio = resample_poly(audio, up, down).astype(np.float32)

    return audio
