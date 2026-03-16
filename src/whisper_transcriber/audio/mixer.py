from __future__ import annotations

import logging
import queue
import threading

import numpy as np

logger = logging.getLogger(__name__)

MIX_TIMEOUT = 0.02


class StreamMixer:
    def __init__(
        self,
        loopback_queue: queue.SimpleQueue,
        mic_queue: queue.SimpleQueue,
        output_queue: queue.SimpleQueue,
        stop_event: threading.Event,
    ) -> None:
        self._loopback_q = loopback_queue
        self._mic_q = mic_queue
        self._output_q = output_queue
        self._stop_event = stop_event

    def run(self) -> None:
        logger.info("Stream mixer started")
        while not self._stop_event.is_set():
            loopback_chunks = self._drain_all(self._loopback_q)
            mic_chunks = self._drain_all(self._mic_q)

            if not loopback_chunks and not mic_chunks:
                self._stop_event.wait(timeout=MIX_TIMEOUT)
                continue

            loopback_data = b"".join(loopback_chunks) if loopback_chunks else None
            mic_data = b"".join(mic_chunks) if mic_chunks else None

            if loopback_data is not None and mic_data is not None:
                self._output_q.put_nowait(self._mix(loopback_data, mic_data))
            elif loopback_data is not None:
                self._output_q.put_nowait(loopback_data)
            elif mic_data is not None:
                self._output_q.put_nowait(mic_data)

        logger.info("Stream mixer stopped")

    @staticmethod
    def _drain_all(q: queue.SimpleQueue) -> list[bytes]:
        chunks = []
        while True:
            try:
                chunks.append(q.get_nowait())
            except Exception:
                break
        return chunks

    @staticmethod
    def _mix(a: bytes, b: bytes) -> bytes:
        arr_a = np.frombuffer(a, dtype=np.int16).astype(np.float32)
        arr_b = np.frombuffer(b, dtype=np.int16).astype(np.float32)
        max_len = max(len(arr_a), len(arr_b))
        padded_a = np.zeros(max_len, dtype=np.float32)
        padded_b = np.zeros(max_len, dtype=np.float32)
        padded_a[: len(arr_a)] = arr_a
        padded_b[: len(arr_b)] = arr_b
        mixed = padded_a + padded_b
        mixed = np.clip(mixed, -32768, 32767)
        return mixed.astype(np.int16).tobytes()
