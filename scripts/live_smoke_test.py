from __future__ import annotations

import argparse
import queue
import tempfile
import threading
import time
from pathlib import Path

from livevox.audio.capture import CaptureManager
from livevox.audio.pipeline import AudioPipeline
from livevox.io.recording import WavRecorder
from livevox.io.transcript_writer import TranscriptWriter
from livevox.session.events import LoggingEvents
from livevox.session.messages import PipelineStatistics
from livevox.stt.cuda_runtime import register_library_directories
from livevox.stt.segmenter import SpeechSegmenter
from livevox.stt.vad import StreamingVad
from livevox.stt.whisper_engine import WhisperEngine
from livevox.stt.worker import TranscriptionWorker

READY_TIMEOUT = 10.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture whatever is playing for a while and audit the session")
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--language", default="es")
    parser.add_argument("--microphone", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    register_library_directories()
    output_dir = arguments.output or Path(tempfile.mkdtemp(prefix="livevox-smoke-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = TranscriptWriter(output_dir)
    writer.open()
    recorder = WavRecorder(output_dir / "recording.wav")
    recorder.open()
    raw_queue: queue.SimpleQueue = queue.SimpleQueue()
    piece_queue: queue.Queue = queue.Queue()
    statistics = PipelineStatistics(recording_path=recorder.path)
    segmenter = SpeechSegmenter(StreamingVad())
    events = LoggingEvents()
    capture = CaptureManager(raw_queue, record_microphone=arguments.microphone)
    pipeline = AudioPipeline(raw_queue, piece_queue, segmenter, recorder, events, statistics)
    worker = TranscriptionWorker(
        piece_queue, WhisperEngine(arguments.model), writer, events, arguments.language,
        captured_samples=lambda: statistics.captured_samples,
        snapshot_source=segmenter.snapshot,
        report_path=output_dir / "session_report.txt",
    )
    threads = [
        threading.Thread(target=capture.run, name="CaptureSupervisor"),
        threading.Thread(target=pipeline.run, name="AudioPipeline"),
        threading.Thread(target=worker.run, name="Transcription"),
    ]
    for thread in threads:
        thread.start()
    if not capture.wait_until_ready(READY_TIMEOUT) or capture.startup_error:
        print(f"Capture failed to start: {capture.startup_error}")
    started = time.monotonic()
    time.sleep(arguments.seconds)
    capture.request_stop()
    stop_requested = time.monotonic()
    for thread in threads:
        thread.join()
    print(f"Captured for {stop_requested - started:.1f}s, finishing took {time.monotonic() - stop_requested:.1f}s")
    print(worker.report.to_text())
    print(f"Session folder: {output_dir}")
    return 0 if worker.report.is_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
