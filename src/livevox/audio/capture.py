from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import pyaudiowpatch as pyaudio

from livevox.audio.devices import (
    DefaultOutputWatcher,
    find_render_device,
    get_default_input_device,
    get_default_loopback,
)

logger = logging.getLogger(__name__)

FRAMES_PER_BUFFER = 2048
KEEP_ALIVE_FRAMES_PER_BUFFER = 2048
STREAM_FORMAT = pyaudio.paInt16
BYTES_PER_SAMPLE = 2
SUPERVISOR_POLL_SECONDS = 0.25
ENDPOINT_POLL_SECONDS = 1.0
REOPEN_RETRY_SECONDS = 1.0
LOOPBACK_SOURCE = "loopback"
MICROPHONE_SOURCE = "microphone"
MONO = 1
STEREO = 2


@dataclass(frozen=True)
class CaptureFormat:
    source: str
    generation: int
    sample_rate: int
    channels: int
    device_name: str


@dataclass(frozen=True)
class CaptureData:
    source: str
    generation: int
    data: bytes
    status: int


@dataclass(frozen=True)
class CaptureClosed:
    source: str
    generation: int


@dataclass(frozen=True)
class CaptureNotice:
    text: str
    device_switch: bool = False


@dataclass(frozen=True)
class CaptureFinished:
    pass


class OutputWatcher(Protocol):
    def current_id(self) -> str | None: ...

    def close(self) -> None: ...


class _InputStream:
    def __init__(self, raw_queue: queue.SimpleQueue, source: str, generation: int, device: dict) -> None:
        self._raw_queue = raw_queue
        self._source = source
        self._generation = generation
        self._device = device
        self._stream: Any = None
        self.sample_rate = int(device["defaultSampleRate"])
        self.channels = int(device["maxInputChannels"])

    @property
    def source(self) -> str:
        return self._source

    @property
    def device_name(self) -> str:
        return str(self._device["name"])

    def _callback(self, in_data: bytes | None, frame_count: int, time_info: dict, status: int) -> tuple[None, int]:
        if in_data:
            self._raw_queue.put(CaptureData(self._source, self._generation, in_data, status))
        return (None, pyaudio.paContinue)

    def open(self, port_audio: Any) -> bool:
        for channels in dict.fromkeys((self.channels, MONO)):
            try:
                self._stream = port_audio.open(
                    format=STREAM_FORMAT,
                    channels=channels,
                    rate=self.sample_rate,
                    frames_per_buffer=FRAMES_PER_BUFFER,
                    input=True,
                    input_device_index=self._device["index"],
                    stream_callback=self._callback,
                )
                self.channels = channels
                self._raw_queue.put(CaptureFormat(
                    self._source, self._generation, self.sample_rate, self.channels, self.device_name,
                ))
                logger.info(
                    "%s capture opened: %s (%d Hz, %d ch)", self._source, self.device_name, self.sample_rate, channels,
                )
                return True
            except Exception:
                logger.warning("%s could not open with %d channel(s)", self._source, channels, exc_info=True)
        return False

    def is_active(self) -> bool:
        try:
            return self._stream is not None and bool(self._stream.is_active())
        except Exception:
            return False

    def close(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop_stream()
        except Exception:
            logger.warning("%s stream did not stop cleanly", self._source, exc_info=True)
        try:
            self._stream.close()
        except Exception:
            logger.warning("%s stream did not close cleanly", self._source, exc_info=True)
        self._stream = None
        self._raw_queue.put(CaptureClosed(self._source, self._generation))
        logger.info("%s stream closed", self._source)


class _KeepAliveStream:
    def __init__(self, device: dict) -> None:
        self._device = device
        self._channels = min(STEREO, int(device["maxOutputChannels"]))
        self._stream: Any = None
        self._silence = b""

    def _callback(self, in_data: bytes | None, frame_count: int, time_info: dict, status: int) -> tuple[bytes, int]:
        required = frame_count * self._channels * BYTES_PER_SAMPLE
        if len(self._silence) != required:
            self._silence = bytes(required)
        return (self._silence, pyaudio.paContinue)

    def open(self, port_audio: Any) -> None:
        try:
            self._stream = port_audio.open(
                format=STREAM_FORMAT,
                channels=self._channels,
                rate=int(self._device["defaultSampleRate"]),
                frames_per_buffer=KEEP_ALIVE_FRAMES_PER_BUFFER,
                output=True,
                output_device_index=self._device["index"],
                stream_callback=self._callback,
            )
            logger.info("Keep-alive silence playing on %s", self._device["name"])
        except Exception:
            logger.warning("Could not open the keep-alive stream on %s", self._device["name"], exc_info=True)
            self._stream = None

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                logger.warning("Keep-alive stream did not close cleanly", exc_info=True)
            self._stream = None


class CaptureManager:
    def __init__(
        self,
        raw_queue: queue.SimpleQueue,
        record_microphone: bool,
        port_audio_factory: Callable[[], Any] = pyaudio.PyAudio,
        watcher_factory: Callable[[], OutputWatcher] = DefaultOutputWatcher,
        keep_alive: bool = True,
    ) -> None:
        self._raw_queue = raw_queue
        self._record_microphone = record_microphone
        self._port_audio_factory = port_audio_factory
        self._watcher_factory = watcher_factory
        self._keep_alive_enabled = keep_alive
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._startup_error: str | None = None
        self._port_audio: Any = None
        self._generation = 0
        self._loopback: _InputStream | None = None
        self._microphone: _InputStream | None = None
        self._keep_alive: _KeepAliveStream | None = None

    @property
    def startup_error(self) -> str | None:
        return self._startup_error

    def wait_until_ready(self, timeout: float) -> bool:
        return self._ready_event.wait(timeout)

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        watcher: OutputWatcher | None = None
        try:
            watcher = self._watcher_factory()
            self._port_audio = self._port_audio_factory()
            if not self._open_all():
                self._startup_error = "No audio output device available for loopback capture"
            self._ready_event.set()
            self._supervise(watcher)
        except Exception:
            logger.exception("Audio capture supervisor failed")
            if self._startup_error is None and not self._ready_event.is_set():
                self._startup_error = "Audio capture could not start"
            self._raw_queue.put(CaptureNotice("Audio capture stopped unexpectedly, see the log"))
        finally:
            self._ready_event.set()
            self._close_all()
            self._terminate_port_audio()
            if watcher is not None:
                watcher.close()
            self._raw_queue.put(CaptureFinished())
            logger.info("Audio capture finished")

    def _supervise(self, watcher: OutputWatcher) -> None:
        last_endpoint = watcher.current_id()
        last_endpoint_poll = time.monotonic()
        last_retry = time.monotonic()
        while not self._stop_event.wait(SUPERVISOR_POLL_SECONDS):
            now = time.monotonic()
            reason: str | None = None
            if self._loopback is None:
                if now - last_retry >= REOPEN_RETRY_SECONDS:
                    last_retry = now
                    reason = "waiting for an output device"
            elif not self._loopback.is_active():
                reason = "the output device stopped"
            elif self._microphone is not None and not self._microphone.is_active():
                reason = "the microphone stopped"
            elif now - last_endpoint_poll >= ENDPOINT_POLL_SECONDS:
                last_endpoint_poll = now
                current_endpoint = watcher.current_id()
                if current_endpoint != last_endpoint:
                    reason = "the default output device changed"
                    last_endpoint = current_endpoint
            if reason is not None:
                self._restart(reason)

    def _restart(self, reason: str) -> None:
        had_loopback = self._loopback is not None
        logger.warning("Restarting audio capture: %s", reason)
        self._close_all()
        self._terminate_port_audio()
        self._port_audio = self._port_audio_factory()
        if self._open_all():
            assert self._loopback is not None
            self._raw_queue.put(CaptureNotice(
                f"Audio capture reconnected ({reason}), now capturing {self._loopback.device_name}",
                device_switch=had_loopback,
            ))

    def _open_all(self) -> bool:
        self._generation += 1
        loopback_device = get_default_loopback(self._port_audio)
        if loopback_device is None:
            return False
        loopback = _InputStream(self._raw_queue, LOOPBACK_SOURCE, self._generation, loopback_device)
        if not loopback.open(self._port_audio):
            return False
        self._loopback = loopback
        if self._keep_alive_enabled:
            render_device = find_render_device(self._port_audio, loopback_device)
            if render_device is not None:
                self._keep_alive = _KeepAliveStream(render_device)
                self._keep_alive.open(self._port_audio)
            else:
                logger.warning("No render device matches %s, silence keep-alive disabled", loopback_device["name"])
        if self._record_microphone:
            microphone_device = get_default_input_device(self._port_audio)
            if microphone_device is not None:
                microphone = _InputStream(self._raw_queue, MICROPHONE_SOURCE, self._generation, microphone_device)
                if microphone.open(self._port_audio):
                    self._microphone = microphone
                else:
                    self._raw_queue.put(CaptureNotice("The microphone could not be opened, capturing system audio only"))
            else:
                self._raw_queue.put(CaptureNotice("No microphone found, capturing system audio only"))
        return True

    def _close_all(self) -> None:
        for stream in (self._loopback, self._microphone):
            if stream is not None:
                stream.close()
        if self._keep_alive is not None:
            self._keep_alive.close()
        self._loopback = None
        self._microphone = None
        self._keep_alive = None

    def _terminate_port_audio(self) -> None:
        if self._port_audio is not None:
            try:
                self._port_audio.terminate()
            except Exception:
                logger.warning("PortAudio did not terminate cleanly", exc_info=True)
            self._port_audio = None
