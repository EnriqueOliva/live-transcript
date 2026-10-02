from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

import numpy as np

from livevox.audio.timeline import SAMPLE_RATE
from livevox.io.transcript_writer import TranscriptWriter
from livevox.session.events import LoggingEvents, SessionEvents
from livevox.session.messages import EndOfStream, PipelineStatistics
from livevox.session.report import SessionReport
from livevox.stt.segmenter import SegmenterConfig, SpeechSegmenter
from livevox.stt.vad import StreamingVad
from livevox.stt.whisper_engine import WhisperEngine
from livevox.stt.worker import DecodingOptions, TranscriptionWorker

logger = logging.getLogger(__name__)

FEED_BLOCK_SAMPLES = SAMPLE_RATE
REPORT_FILE_NAME = "session_report.txt"


def decode_media(path: Path) -> np.ndarray:
    from faster_whisper.audio import decode_audio

    return np.asarray(decode_audio(str(path), sampling_rate=SAMPLE_RATE), dtype=np.float32)


def transcribe_samples(
    samples: np.ndarray,
    engine: WhisperEngine,
    output_dir: Path,
    language: str,
    events: SessionEvents | None = None,
    config: SegmenterConfig | None = None,
    decoding: DecodingOptions | None = None,
) -> SessionReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = TranscriptWriter(output_dir)
    writer.open()
    piece_queue: queue.Queue = queue.Queue()
    segmenter = SpeechSegmenter(StreamingVad(), config)
    statistics = PipelineStatistics()
    worker = TranscriptionWorker(
        piece_queue,
        engine,
        writer,
        events or LoggingEvents(),
        language=language,
        captured_samples=lambda: statistics.captured_samples,
        report_path=output_dir / REPORT_FILE_NAME,
        partials=False,
        decoding=decoding,
    )
    thread = threading.Thread(target=worker.run, name="Transcription")
    thread.start()
    for start in range(0, samples.size, FEED_BLOCK_SAMPLES):
        block = samples[start : start + FEED_BLOCK_SAMPLES]
        statistics.captured_samples += block.size
        for piece in segmenter.feed(block):
            piece_queue.put(piece)
    for piece in segmenter.finish():
        piece_queue.put(piece)
    piece_queue.put(EndOfStream(statistics))
    thread.join()
    return worker.report


def transcribe_file(media_path: Path, model: str, language: str, output_dir: Path, compute_type: str = "auto") -> int:
    samples = decode_media(media_path)
    logger.info("Decoded %s: %.1fs of audio", media_path, samples.size / SAMPLE_RATE)
    engine = WhisperEngine(model, compute_type_setting=compute_type)
    report = transcribe_samples(samples, engine, output_dir, language)
    print(report.to_text())
    print(f"Transcript written to {output_dir}")
    return 0 if report.is_complete else 1
