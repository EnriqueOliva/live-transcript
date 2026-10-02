from __future__ import annotations

import numpy as np

from whisper_transcriber.audio.timeline import seconds_to_samples

DEFAULT_MAXIMUM_SKEW_SECONDS = 0.5
EMPTY = np.zeros(0, dtype=np.float32)


class _SourceBuffer:
    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._length = 0

    def __len__(self) -> int:
        return self._length

    def push(self, samples: np.ndarray) -> None:
        if samples.size:
            self._chunks.append(np.asarray(samples, dtype=np.float32))
            self._length += samples.size

    def take(self, count: int) -> np.ndarray:
        if count <= 0:
            return EMPTY
        joined = np.concatenate(self._chunks) if len(self._chunks) > 1 else self._chunks[0]
        taken = joined[:count]
        rest = joined[count:]
        self._chunks = [rest] if rest.size else []
        self._length = rest.size
        return taken


class SourceMixer:
    def __init__(self, maximum_skew_seconds: float = DEFAULT_MAXIMUM_SKEW_SECONDS) -> None:
        self._maximum_skew = seconds_to_samples(maximum_skew_seconds)
        self._sources: dict[str, _SourceBuffer] = {}

    @property
    def source_names(self) -> list[str]:
        return list(self._sources)

    def add_source(self, name: str) -> None:
        if name not in self._sources:
            self._sources[name] = _SourceBuffer()

    def push(self, name: str, samples: np.ndarray) -> None:
        self.add_source(name)
        self._sources[name].push(samples)

    def pending_samples(self) -> int:
        return max((len(buffer) for buffer in self._sources.values()), default=0)

    def pull(self) -> np.ndarray:
        if not self._sources:
            return EMPTY
        common = min(len(buffer) for buffer in self._sources.values())
        leading = max(len(buffer) for buffer in self._sources.values())
        excess = leading - common - self._maximum_skew
        count = common + max(0, excess)
        return self._mix(count)

    def flush(self) -> np.ndarray:
        return self._mix(self.pending_samples())

    def _mix(self, count: int) -> np.ndarray:
        if count <= 0:
            return EMPTY
        mixed = np.zeros(count, dtype=np.float32)
        for buffer in self._sources.values():
            taken = buffer.take(min(count, len(buffer)))
            mixed[: taken.size] += taken
        if len(self._sources) > 1:
            np.clip(mixed, -1.0, 1.0, out=mixed)
        return mixed
