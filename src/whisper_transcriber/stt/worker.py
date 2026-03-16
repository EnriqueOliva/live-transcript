from __future__ import annotations

import gc
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from whisper_transcriber.stt.chunk_merger import ChunkMerger
from whisper_transcriber.stt.whisper_engine import WhisperEngine

if TYPE_CHECKING:
    from whisper_transcriber.io.transcript_writer import TranscriptWriter
    from whisper_transcriber.ui.signals import WorkerSignals

logger = logging.getLogger(__name__)

BACKPRESSURE_THRESHOLD = 3
SILENCE_RMS_F32 = 0.001


class TranscriptionWorker:
    def __init__(
        self,
        chunk_queue: queue.Queue,
        stop_event: threading.Event,
        engine: WhisperEngine,
        merger: ChunkMerger,
        signals: WorkerSignals,
        writer: TranscriptWriter,
        safe_duration: float,
        chunk_duration: float = 30.0,
    ) -> None:
        self._chunk_queue = chunk_queue
        self._stop_event = stop_event
        self._engine = engine
        self._merger = merger
        self._signals = signals
        self._writer = writer
        self._safe_duration = safe_duration
        self._chunk_duration = chunk_duration
        self._consecutive_silent = 0

    def run(self) -> None:
        logger.info("Transcription worker started")
        if not self._engine.is_loaded:
            try:
                self._signals.status_update.emit("Loading model...", "", "")
                self._signals.simple_status.emit("Loading transcription model... this may take a moment")
                self._engine.load_model()
                self._signals.status_update.emit("Model loaded", "", "")
                self._signals.simple_status.emit("Model ready, listening for audio")
            except Exception:
                logger.exception("Failed to load model")
                self._signals.error_occurred.emit("Failed to load transcription model")
                return

        self._signals.status_update.emit("Recording", "", "")

        while not self._stop_event.is_set():
            try:
                chunk_data = self._chunk_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            chunk_index, audio_array, chunk_offset = chunk_data

            while self._chunk_queue.qsize() > BACKPRESSURE_THRESHOLD:
                try:
                    chunk_index, audio_array, chunk_offset = self._chunk_queue.get_nowait()
                    logger.warning("Skipping to chunk %d (backpressure)", chunk_index)
                except queue.Empty:
                    break

            rms = float(np.sqrt(np.mean(audio_array**2)))
            if rms < SILENCE_RMS_F32:
                self._consecutive_silent += 1
                if self._consecutive_silent == 1 or self._consecutive_silent % 10 == 0:
                    logger.debug("Silent chunk %d (consecutive: %d)", chunk_index, self._consecutive_silent)
                continue
            if self._consecutive_silent > 1:
                logger.debug("Speech resumed after %d silent chunks", self._consecutive_silent)
            self._consecutive_silent = 0

            self._process_chunk(chunk_index, audio_array, chunk_offset, flush=False)

        self._drain_remaining()
        logger.info("Transcription worker stopped")

    def _process_chunk(
        self, chunk_index: int, audio_array: np.ndarray, chunk_offset: float, flush: bool = False,
    ) -> None:
        try:
            t0 = time.monotonic()
            segments, info = self._engine.transcribe(audio_array)
            elapsed = time.monotonic() - t0

            logger.info(
                "Chunk %d: %.2fs inference, %d segments, speech: %.1f/%.1fs",
                chunk_index, elapsed, len(segments), info.duration_after_vad, info.duration,
            )
            self._signals.simple_status.emit(
                f"Transcribed chunk {chunk_index} ({elapsed:.1f}s inference, {len(segments)} segments)"
            )

            if not segments:
                return

            safe = None if flush else self._safe_duration
            merged = self._merger.merge(
                segments, chunk_offset, safe_duration=safe, chunk_duration=self._chunk_duration,
            )
            for seg in merged:
                self._writer.write_segment(seg.text, seg.start, seg.end)
                self._signals.transcript_segment.emit(seg.text, seg.start, seg.end)

            self._signals.status_update.emit("Recording", "", f"Pending: {self._chunk_queue.qsize()}")

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error("CUDA OOM, skipping chunk %d", chunk_index)
                gc.collect()
                WhisperEngine._cuda_empty_cache()
            else:
                logger.exception("Transcription error at chunk %d", chunk_index)
        except Exception:
            logger.exception("Unexpected error at chunk %d", chunk_index)

    def _drain_remaining(self) -> None:
        remaining = []
        while not self._chunk_queue.empty():
            try:
                remaining.append(self._chunk_queue.get_nowait())
            except queue.Empty:
                break
        for i, (chunk_index, audio_array, chunk_offset) in enumerate(remaining):
            is_last = i == len(remaining) - 1
            self._process_chunk(chunk_index, audio_array, chunk_offset, flush=is_last)
