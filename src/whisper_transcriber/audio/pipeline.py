from __future__ import annotations

import logging
import queue
import time
from typing import TYPE_CHECKING

import numpy as np
import pyaudiowpatch as pyaudio

from whisper_transcriber.audio.capture import (
    CaptureClosed,
    CaptureData,
    CaptureFinished,
    CaptureFormat,
    CaptureNotice,
)
from whisper_transcriber.audio.conversion import StreamConverter
from whisper_transcriber.audio.mixer import SourceMixer
from whisper_transcriber.session.messages import EndOfStream, Notice, PipelineStatistics

if TYPE_CHECKING:
    from whisper_transcriber.io.recording import WavRecorder
    from whisper_transcriber.session.events import SessionEvents
    from whisper_transcriber.stt.segmenter import Piece, SpeechSegmenter

logger = logging.getLogger(__name__)

RAW_QUEUE_POLL_SECONDS = 0.05
MAXIMUM_MESSAGES_PER_BATCH = 500
LEVEL_EMIT_INTERVAL = 0.1
OVERFLOW_NOTICE_INTERVAL = 10.0
NUM_BANDS = 24
SPECTRUM_SAMPLES = 1024
LEVEL_SCALE = 0.15
INPUT_OVERFLOW_FLAG = int(pyaudio.paInputOverflow)


def compute_band_levels(samples: np.ndarray, num_bands: int = NUM_BANDS) -> list[float]:
    if samples.size < SPECTRUM_SAMPLES // 2:
        return [0.0] * num_bands
    spectrum = np.abs(np.fft.rfft(samples[-SPECTRUM_SAMPLES:]))
    band_size = max(1, len(spectrum) // num_bands)
    levels = []
    for band_index in range(num_bands):
        band = spectrum[band_index * band_size : (band_index + 1) * band_size]
        value = float(np.mean(band)) if len(band) > 0 else 0.0
        levels.append(min(1.0, value * LEVEL_SCALE))
    return levels


class AudioPipeline:
    def __init__(
        self,
        raw_queue: queue.SimpleQueue,
        piece_queue: queue.Queue,
        segmenter: SpeechSegmenter,
        recorder: WavRecorder | None,
        events: SessionEvents,
        statistics: PipelineStatistics,
    ) -> None:
        self._raw_queue = raw_queue
        self._piece_queue = piece_queue
        self._segmenter = segmenter
        self._recorder = recorder
        self._events = events
        self._statistics = statistics
        self._converters: dict[tuple[str, int], StreamConverter] = {}
        self._mixer = SourceMixer()
        self._last_level_emit = 0.0
        self._last_overflow_notice = 0.0
        self._recent = np.zeros(SPECTRUM_SAMPLES, dtype=np.float32)

    def run(self) -> None:
        logger.info("Audio pipeline started")
        finished = False
        try:
            while not finished:
                for message in self._next_batch():
                    try:
                        finished = self._handle(message) or finished
                    except Exception:
                        logger.exception("Audio pipeline could not handle %s", type(message).__name__)
                self._consume_safely(self._mixer.pull())
        finally:
            self._finish()
        logger.info("Audio pipeline stopped")

    def _consume_safely(self, samples: np.ndarray) -> None:
        try:
            self._consume(samples)
        except Exception:
            logger.exception("Audio pipeline failed to process %d samples", samples.size)
            self._events.error("Audio processing failed, the recording is kept. See the log for details.")

    def _next_batch(self) -> list[object]:
        batch: list[object] = []
        try:
            batch.append(self._raw_queue.get(timeout=RAW_QUEUE_POLL_SECONDS))
        except queue.Empty:
            return batch
        while len(batch) < MAXIMUM_MESSAGES_PER_BATCH:
            try:
                batch.append(self._raw_queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _handle(self, message: object) -> bool:
        if isinstance(message, CaptureData):
            converter = self._converters.get((message.source, message.generation))
            if converter is None:
                logger.error("Audio from %s#%d arrived before its format", message.source, message.generation)
            else:
                self._mixer.push(message.source, converter.convert(message.data))
            if message.status & INPUT_OVERFLOW_FLAG:
                self._on_overflow(message.source)
            return False
        elif isinstance(message, CaptureFormat):
            self._converters[(message.source, message.generation)] = StreamConverter(
                message.sample_rate, message.channels,
            )
            self._mixer.add_source(message.source)
            if message.device_name not in self._statistics.devices:
                self._statistics.devices.append(message.device_name)
            return False
        elif isinstance(message, CaptureClosed):
            converter = self._converters.pop((message.source, message.generation), None)
            if converter is not None:
                self._mixer.push(message.source, converter.flush())
            return False
        elif isinstance(message, CaptureNotice):
            if message.device_switch:
                self._statistics.device_switches += 1
            self._post_notice(message.text)
            return False
        elif isinstance(message, CaptureFinished):
            return True
        else:
            logger.error("Unknown capture message %r", message)
            return False

    def _on_overflow(self, source: str) -> None:
        self._statistics.overflow_events += 1
        now = time.monotonic()
        if now - self._last_overflow_notice >= OVERFLOW_NOTICE_INTERVAL:
            self._last_overflow_notice = now
            self._post_notice(f"Windows reported an audio overflow on {source}, a few milliseconds may be missing")

    def _post_notice(self, text: str) -> None:
        self._piece_queue.put(Notice(sample=self._statistics.captured_samples, text=text))
        self._events.notice(text)

    def _consume(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        if self._recorder is not None:
            self._recorder.write(samples)
        self._statistics.captured_samples += samples.size
        self._emit_levels(samples)
        self._enqueue(self._segmenter.feed(samples))

    def _enqueue(self, pieces: list[Piece]) -> None:
        for piece in pieces:
            self._piece_queue.put(piece)

    def _emit_levels(self, samples: np.ndarray) -> None:
        if samples.size >= SPECTRUM_SAMPLES:
            self._recent = samples[-SPECTRUM_SAMPLES:].copy()
        else:
            self._recent = np.concatenate([self._recent[samples.size :], samples])
        now = time.monotonic()
        if now - self._last_level_emit >= LEVEL_EMIT_INTERVAL:
            self._last_level_emit = now
            self._events.audio_levels(compute_band_levels(self._recent))

    def _finish(self) -> None:
        try:
            for key in list(self._converters):
                converter = self._converters.pop(key)
                self._mixer.push(key[0], converter.flush())
            self._consume_safely(self._mixer.flush())
            self._enqueue(self._segmenter.finish())
        except Exception:
            logger.exception("Audio pipeline could not flush the end of the stream")
        finally:
            if self._recorder is not None:
                self._recorder.close()
                self._statistics.recorded_samples = self._recorder.samples_written
                self._statistics.recording_failed = self._recorder.has_error
                self._statistics.recording_path = self._recorder.path
            self._piece_queue.put(EndOfStream(self._statistics))
