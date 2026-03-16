from __future__ import annotations

import logging
import os
import queue
import threading
from pathlib import Path

import pyaudiowpatch as pyaudio

from whisper_transcriber.audio.accumulator import AudioAccumulator
from whisper_transcriber.audio.capture import AudioCapture
from whisper_transcriber.audio.devices import get_default_input_device, resolve_device
from whisper_transcriber.audio.mixer import StreamMixer
from whisper_transcriber.config.settings import AppSettings
from whisper_transcriber.io.paths import MODELS_DIR, create_session_paths, ensure_dirs
from whisper_transcriber.io.transcript_writer import TranscriptWriter
from whisper_transcriber.logging import log_setup
from whisper_transcriber.logging.log_bridge import GuiBridge
from whisper_transcriber.stt.chunk_merger import ChunkMerger
from whisper_transcriber.stt.whisper_engine import WhisperEngine
from whisper_transcriber.stt.worker import TranscriptionWorker
from whisper_transcriber.ui.signals import WorkerSignals

logger = logging.getLogger(__name__)

CHUNK_QUEUE_MAXSIZE = 4
THREAD_JOIN_TIMEOUT = 10.0


class Application:
    def __init__(self) -> None:
        os.environ["HF_HOME"] = str(MODELS_DIR)
        self._settings = AppSettings.load()
        self._gui_bridge = GuiBridge()
        ensure_dirs()
        log_setup.setup(gui_handler=self._gui_bridge)
        self._worker_signals = WorkerSignals()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._writer: TranscriptWriter | None = None
        self._transcript_dir: Path | None = None
        self._pa: pyaudio.PyAudio | None = None
        self._is_running = False

    @property
    def settings(self) -> AppSettings:
        return self._settings

    @property
    def worker_signals(self) -> WorkerSignals:
        return self._worker_signals

    @property
    def gui_bridge(self) -> GuiBridge:
        return self._gui_bridge

    def start_session(self, model: str, language: str) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._stop_event.clear()

        transcript_dir, log_dir = create_session_paths()
        self._transcript_dir = transcript_dir
        log_setup.add_session_handler(log_dir)

        self._pa = pyaudio.PyAudio()
        device = resolve_device(self._pa, self._settings.audio_device)
        if device is None:
            self._cleanup_failed_start()
            self._worker_signals.error_occurred.emit("No audio loopback device found")
            return

        raw_q: queue.SimpleQueue = queue.SimpleQueue()
        accum_q: queue.SimpleQueue = raw_q
        chunk_q: queue.Queue = queue.Queue(maxsize=CHUNK_QUEUE_MAXSIZE)

        loopback = AudioCapture(raw_q, self._stop_event, device, label="loopback")
        if not loopback.open(self._pa):
            self._cleanup_failed_start()
            self._worker_signals.error_occurred.emit("Failed to open loopback audio device")
            return

        self._threads = [
            threading.Thread(target=loopback.wait_until_stopped, daemon=True, name="AudioCapture"),
        ]

        if self._settings.record_mic:
            mic_device = get_default_input_device(self._pa)
            if mic_device:
                mic_q: queue.SimpleQueue = queue.SimpleQueue()
                mic_cap = AudioCapture(mic_q, self._stop_event, mic_device, label="mic")
                if mic_cap.open(self._pa):
                    accum_q = queue.SimpleQueue()
                    mixer = StreamMixer(raw_q, mic_q, accum_q, self._stop_event)
                    self._threads.append(
                        threading.Thread(target=mic_cap.wait_until_stopped, daemon=True, name="MicCapture")
                    )
                    self._threads.append(threading.Thread(target=mixer.run, daemon=True, name="Mixer"))
                else:
                    logger.warning("Mic open failed, proceeding with loopback only")

        self._writer = TranscriptWriter(transcript_dir)
        self._writer.open()

        accumulator = AudioAccumulator(
            accum_q, chunk_q, self._stop_event, self._worker_signals,
            source_rate=loopback.sample_rate, source_channels=loopback.channels,
            chunk_duration=self._settings.chunk_duration,
            overlap_duration=self._settings.overlap_seconds,
        )
        engine = WhisperEngine(
            model_size=model, language=language,
            compute_type=self._settings.compute_type, model_dir=MODELS_DIR,
            initial_prompt=self._settings.initial_prompt, hotwords=self._settings.hotwords,
        )
        safe_duration = self._settings.chunk_duration - self._settings.overlap_seconds
        worker = TranscriptionWorker(
            chunk_q, self._stop_event, engine, ChunkMerger(),
            self._worker_signals, self._writer,
            safe_duration=safe_duration, chunk_duration=self._settings.chunk_duration,
        )

        self._threads.append(threading.Thread(target=accumulator.run, daemon=True, name="Accumulator"))
        self._threads.append(threading.Thread(target=worker.run, daemon=True, name="Transcription"))
        for t in self._threads:
            t.start()
        logger.info("Session started in %s", transcript_dir)

    def _cleanup_failed_start(self) -> None:
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        log_setup.remove_session_handler()
        self._is_running = False

    def open_session_folder(self) -> None:
        if self._transcript_dir and self._transcript_dir.exists():
            os.startfile(self._transcript_dir)

    def stop_session(self) -> None:
        if not self._is_running:
            return
        logger.info("Stopping session...")
        self._stop_event.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=THREAD_JOIN_TIMEOUT)
                if t.is_alive():
                    logger.warning("Thread %s did not exit within timeout", t.name)
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        all_stopped = all(not t.is_alive() for t in self._threads)
        if self._pa is not None and all_stopped:
            self._pa.terminate()
            self._pa = None
        elif self._pa is not None:
            logger.warning("Skipping pa.terminate() - threads still alive")
        log_setup.remove_session_handler()
        self._threads.clear()
        self._is_running = False
        logger.info("Session stopped")

    def shutdown(self) -> None:
        if self._is_running:
            self.stop_session()
        self._settings.save()
        log_setup.shutdown()
