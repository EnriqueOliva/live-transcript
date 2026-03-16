from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from whisper_transcriber.audio.resampler import convert_to_whisper_format

if TYPE_CHECKING:
    from whisper_transcriber.ui.signals import WorkerSignals

logger = logging.getLogger(__name__)

SILENCE_RMS_THRESHOLD = 50.0
HEARTBEAT_INTERVAL = 15.0
LEVEL_EMIT_INTERVAL = 0.1
NUM_BANDS = 24


def _compute_band_levels(raw_bytes: bytes, num_bands: int) -> list[float]:
    pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if len(pcm) < 512:
        return [0.0] * num_bands
    spectrum = np.abs(np.fft.rfft(pcm[-2048:]))
    band_size = max(1, len(spectrum) // num_bands)
    levels = []
    for i in range(num_bands):
        start = i * band_size
        band = spectrum[start : start + band_size]
        val = float(np.mean(band)) if len(band) > 0 else 0.0
        levels.append(min(1.0, val * 0.15))
    return levels


class AudioAccumulator:
    def __init__(
        self,
        raw_queue: queue.SimpleQueue,
        chunk_queue: queue.Queue,
        stop_event: threading.Event,
        signals: WorkerSignals,
        source_rate: int,
        source_channels: int,
        chunk_duration: float = 30.0,
        overlap_duration: float = 5.0,
    ) -> None:
        self._raw_queue = raw_queue
        self._chunk_queue = chunk_queue
        self._stop_event = stop_event
        self._signals = signals
        self._source_rate = source_rate
        self._source_channels = source_channels

        bytes_per_second = source_rate * source_channels * 2
        self._chunk_size = int(chunk_duration * bytes_per_second)
        self._overlap_size = int(overlap_duration * bytes_per_second)
        self._step_size = self._chunk_size - self._overlap_size
        self._buffer = bytearray()
        self._chunk_index = 0
        self._global_byte_offset = 0
        self._silence_start: float | None = None
        self._last_heartbeat: float = 0.0
        self._last_level_emit: float = 0.0

    def _is_silent(self, raw_bytes: bytes) -> bool:
        pcm = np.frombuffer(raw_bytes, dtype=np.int16)
        rms = np.sqrt(np.mean(pcm.astype(np.float32) ** 2))
        return rms < SILENCE_RMS_THRESHOLD

    def _compute_global_offset(self) -> float:
        bytes_per_second = self._source_rate * self._source_channels * 2
        return self._global_byte_offset / bytes_per_second

    def _emit_heartbeat(self) -> None:
        now = time.monotonic()
        if self._silence_start is None:
            self._silence_start = now
        if now - self._last_heartbeat >= HEARTBEAT_INTERVAL:
            elapsed = int(now - self._silence_start)
            minutes, seconds = divmod(elapsed, 60)
            self._signals.simple_status.emit(f"Listening... no speech for {minutes}m {seconds:02d}s")
            self._last_heartbeat = now

    def _emit_audio_levels(self, raw_data: bytes) -> None:
        now = time.monotonic()
        if now - self._last_level_emit >= LEVEL_EMIT_INTERVAL:
            levels = _compute_band_levels(raw_data, NUM_BANDS)
            self._signals.audio_levels.emit(levels)
            self._last_level_emit = now

    def run(self) -> None:
        logger.info("Accumulator thread started")
        while not self._stop_event.is_set():
            try:
                raw_data = self._raw_queue.get(timeout=0.5)
                self._buffer.extend(raw_data)
                self._emit_audio_levels(raw_data)
            except Exception:
                continue

            if len(self._buffer) < self._chunk_size:
                continue

            chunk_bytes = bytes(self._buffer[: self._chunk_size])
            self._buffer = self._buffer[self._step_size :]
            global_offset = self._compute_global_offset()
            self._global_byte_offset += self._step_size

            if self._is_silent(chunk_bytes):
                self._emit_heartbeat()
                self._chunk_index += 1
                continue

            self._silence_start = None
            self._signals.simple_status.emit(f"Processing audio chunk {self._chunk_index}...")
            audio_f32 = convert_to_whisper_format(chunk_bytes, self._source_rate, self._source_channels)

            try:
                self._chunk_queue.put((self._chunk_index, audio_f32, global_offset), timeout=10.0)
            except queue.Full:
                logger.warning("Chunk queue full, dropping chunk %d", self._chunk_index)

            self._chunk_index += 1

        self._flush_remaining()
        logger.info("Accumulator thread stopped")

    def _flush_remaining(self) -> None:
        min_bytes = self._source_rate * self._source_channels * 2 * 2
        if len(self._buffer) > min_bytes and not self._is_silent(bytes(self._buffer)):
            audio_f32 = convert_to_whisper_format(bytes(self._buffer), self._source_rate, self._source_channels)
            global_offset = self._compute_global_offset()
            try:
                self._chunk_queue.put((self._chunk_index, audio_f32, global_offset), timeout=5.0)
            except queue.Full:
                pass
