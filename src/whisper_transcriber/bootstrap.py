from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

from whisper_transcriber.audio.capture import CaptureManager
from whisper_transcriber.audio.pipeline import AudioPipeline
from whisper_transcriber.config.settings import AppSettings
from whisper_transcriber.io.paths import TRANSCRIPTS_DIR, create_session_paths, ensure_dirs
from whisper_transcriber.io.recording import WavRecorder
from whisper_transcriber.io.transcript_writer import TranscriptWriter
from whisper_transcriber.logging import log_setup
from whisper_transcriber.logging.log_bridge import GuiBridge
from whisper_transcriber.session.messages import PipelineStatistics
from whisper_transcriber.stt.segmenter import SpeechSegmenter
from whisper_transcriber.stt.vad import StreamingVad
from whisper_transcriber.stt.whisper_engine import WhisperEngine
from whisper_transcriber.stt.worker import TranscriptionWorker
from whisper_transcriber.ui.signals import QtSessionEvents, WorkerSignals

logger = logging.getLogger(__name__)

CAPTURE_READY_TIMEOUT = 10.0
THREAD_JOIN_TIMEOUT = 5.0
RECORDING_FILE_NAME = "recording.wav"
REPORT_FILE_NAME = "session_report.txt"


@dataclass
class _Session:
    directory: Path
    capture: CaptureManager
    threads: list[threading.Thread]


class Application:
    def __init__(self) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self._settings = AppSettings.load()
        self._gui_bridge = GuiBridge()
        ensure_dirs()
        log_setup.setup(gui_handler=self._gui_bridge)
        self._worker_signals = WorkerSignals()
        self._events = QtSessionEvents(self._worker_signals)
        self._session: _Session | None = None
        self._last_session_dir: Path | None = None

    @property
    def settings(self) -> AppSettings:
        return self._settings

    @property
    def worker_signals(self) -> WorkerSignals:
        return self._worker_signals

    @property
    def gui_bridge(self) -> GuiBridge:
        return self._gui_bridge

    @property
    def is_busy(self) -> bool:
        return self._session is not None

    def start_session(self, model: str, language: str) -> None:
        if self._session is not None:
            return
        session_dir = create_session_paths()
        self._last_session_dir = session_dir
        writer = TranscriptWriter(session_dir)
        writer.open()
        recorder = WavRecorder(session_dir / RECORDING_FILE_NAME)
        recorder.open()

        raw_queue: queue.SimpleQueue = queue.SimpleQueue()
        piece_queue: queue.Queue = queue.Queue()
        statistics = PipelineStatistics(recording_path=recorder.path)
        segmenter = SpeechSegmenter(StreamingVad())
        capture = CaptureManager(raw_queue, record_microphone=self._settings.record_mic)
        pipeline = AudioPipeline(raw_queue, piece_queue, segmenter, recorder, self._events, statistics)
        engine = WhisperEngine(
            model,
            compute_type_setting=self._settings.compute_type,
            initial_prompt=self._settings.initial_prompt,
            hotwords=self._settings.hotwords,
        )
        worker = TranscriptionWorker(
            piece_queue,
            engine,
            writer,
            self._events,
            language=language,
            captured_samples=lambda: statistics.captured_samples,
            snapshot_source=segmenter.snapshot,
            report_path=session_dir / REPORT_FILE_NAME,
        )
        threads = [
            threading.Thread(target=capture.run, daemon=True, name="CaptureSupervisor"),
            threading.Thread(target=pipeline.run, daemon=True, name="AudioPipeline"),
            threading.Thread(target=worker.run, daemon=True, name="Transcription"),
        ]
        self._session = _Session(directory=session_dir, capture=capture, threads=threads)
        for thread in threads:
            thread.start()
        logger.info("Session started in %s", session_dir)

        if not capture.wait_until_ready(CAPTURE_READY_TIMEOUT):
            self._events.error("The audio device did not respond")
        elif capture.startup_error is not None:
            self._events.error(capture.startup_error)
        else:
            self._events.status("Recording", "")

    def stop_session(self) -> None:
        if self._session is not None:
            logger.info("Stopping capture, the remaining audio will be transcribed")
            self._session.capture.request_stop()

    def finalize_session(self) -> None:
        session = self._session
        if session is None:
            return
        session.capture.request_stop()
        for thread in session.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=THREAD_JOIN_TIMEOUT)
                if thread.is_alive():
                    logger.warning("Thread %s still running after the session finished", thread.name)
        self._session = None
        logger.info("Session finalized")

    def open_session_folder(self) -> None:
        target = self._last_session_dir if self._last_session_dir and self._last_session_dir.exists() else TRANSCRIPTS_DIR
        target.mkdir(parents=True, exist_ok=True)
        os.startfile(target)

    def shutdown(self) -> None:
        if self._session is not None:
            self.finalize_session()
        self._settings.save()
        log_setup.shutdown()
