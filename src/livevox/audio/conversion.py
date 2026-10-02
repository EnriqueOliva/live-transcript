from __future__ import annotations

import av
import numpy as np

from livevox.audio.timeline import INT16_FULL_SCALE, SAMPLE_RATE

BYTES_PER_SAMPLE = 2
EMPTY = np.zeros(0, dtype=np.float32)


class StreamConverter:
    def __init__(self, sample_rate: int, channels: int) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_bytes = channels * BYTES_PER_SAMPLE
        self._remainder = b""
        if sample_rate == SAMPLE_RATE:
            self._resampler: av.AudioResampler | None = None
        else:
            self._resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def convert(self, raw_bytes: bytes) -> np.ndarray:
        data = self._remainder + raw_bytes
        usable_length = len(data) - len(data) % self._frame_bytes
        self._remainder = data[usable_length:]
        if usable_length == 0:
            return EMPTY
        interleaved = np.frombuffer(data[:usable_length], dtype=np.int16).astype(np.float32) / INT16_FULL_SCALE
        mono = interleaved.reshape(-1, self._channels).mean(axis=1, dtype=np.float32)
        return self._resample(mono)

    def flush(self) -> np.ndarray:
        if self._resampler is None:
            return EMPTY
        else:
            return _frames_to_array(self._resampler.resample(None))

    def _resample(self, mono: np.ndarray) -> np.ndarray:
        if self._resampler is None:
            return np.ascontiguousarray(mono, dtype=np.float32)
        else:
            frame = av.AudioFrame.from_ndarray(mono.reshape(1, -1), format="flt", layout="mono")
            frame.sample_rate = self._sample_rate
            return _frames_to_array(self._resampler.resample(frame))


def _frames_to_array(frames: list[av.AudioFrame]) -> np.ndarray:
    if not frames:
        return EMPTY
    else:
        return np.concatenate([frame.to_ndarray().reshape(-1) for frame in frames]).astype(np.float32, copy=False)
