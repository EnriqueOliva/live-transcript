from __future__ import annotations

import logging
import os
import struct
import time
from pathlib import Path
from typing import BinaryIO

import numpy as np

from livevox.audio.timeline import INT16_FULL_SCALE, INT16_MAXIMUM, SAMPLE_RATE

logger = logging.getLogger(__name__)

HEADER_SIZE = 44
RIFF_SIZE_OFFSET = 4
DATA_SIZE_OFFSET = 40
BITS_PER_SAMPLE = 16
BYTES_PER_SAMPLE = 2
MONO = 1
PCM_FORMAT = 1
FMT_CHUNK_SIZE = 16
RIFF_HEADER_REMAINDER = HEADER_SIZE - 8
HEADER_REFRESH_SECONDS = 1.0
DURABLE_SYNC_SECONDS = 5.0
INT16_MINIMUM = -32768


def float_to_int16(samples: np.ndarray) -> np.ndarray:
    scaled = np.rint(np.asarray(samples, dtype=np.float32) * INT16_FULL_SCALE)
    converted: np.ndarray = np.clip(scaled, INT16_MINIMUM, INT16_MAXIMUM).astype(np.int16)
    return converted


def _build_header(data_size: int, sample_rate: int) -> bytes:
    byte_rate = sample_rate * MONO * BYTES_PER_SAMPLE
    block_align = MONO * BYTES_PER_SAMPLE
    return b"".join([
        b"RIFF",
        struct.pack("<I", RIFF_HEADER_REMAINDER + data_size),
        b"WAVE",
        b"fmt ",
        struct.pack("<IHHIIHH", FMT_CHUNK_SIZE, PCM_FORMAT, MONO, sample_rate, byte_rate, block_align, BITS_PER_SAMPLE),
        b"data",
        struct.pack("<I", data_size),
    ])


class WavRecorder:
    def __init__(self, path: Path, sample_rate: int = SAMPLE_RATE) -> None:
        self._path = path
        self._sample_rate = sample_rate
        self._handle: BinaryIO | None = None
        self._data_size = 0
        self._last_header_refresh = 0.0
        self._last_durable_sync = 0.0
        self._has_error = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def samples_written(self) -> int:
        return self._data_size // BYTES_PER_SAMPLE

    @property
    def has_error(self) -> bool:
        return self._has_error

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._path, "w+b")  # noqa: SIM115
        self._handle.write(_build_header(0, self._sample_rate))
        self._handle.flush()
        now = time.monotonic()
        self._last_header_refresh = now
        self._last_durable_sync = now
        logger.info("Recording audio to %s", self._path)

    def write(self, samples: np.ndarray) -> None:
        if self._handle is None or samples.size == 0:
            return
        payload = float_to_int16(samples).tobytes()
        try:
            self._handle.write(payload)
            self._data_size += len(payload)
            now = time.monotonic()
            if now - self._last_header_refresh >= HEADER_REFRESH_SECONDS:
                self._refresh_header()
                self._last_header_refresh = now
            if now - self._last_durable_sync >= DURABLE_SYNC_SECONDS:
                os.fsync(self._handle.fileno())
                self._last_durable_sync = now
        except OSError:
            if not self._has_error:
                logger.exception("Failed to write audio recording")
            self._has_error = True

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._refresh_header()
            os.fsync(self._handle.fileno())
        except OSError:
            logger.exception("Failed to finalize audio recording")
            self._has_error = True
        finally:
            self._handle.close()
            self._handle = None
        logger.info("Recording closed: %.1f s of audio", self.samples_written / self._sample_rate)

    def _refresh_header(self) -> None:
        if self._handle is None:
            return
        end_position = self._handle.tell()
        self._handle.seek(RIFF_SIZE_OFFSET)
        self._handle.write(struct.pack("<I", RIFF_HEADER_REMAINDER + self._data_size))
        self._handle.seek(DATA_SIZE_OFFSET)
        self._handle.write(struct.pack("<I", self._data_size))
        self._handle.seek(end_position)
        self._handle.flush()
