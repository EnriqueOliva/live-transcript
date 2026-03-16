from __future__ import annotations

import logging
import queue
import threading

import pyaudiowpatch as pyaudio

logger = logging.getLogger(__name__)

FRAMES_PER_BUFFER = 512
STREAM_FORMAT = pyaudio.paInt16


class AudioCapture:
    def __init__(
        self,
        raw_queue: queue.SimpleQueue,
        stop_event: threading.Event,
        device_info: dict,
        label: str = "audio",
    ) -> None:
        self._raw_queue = raw_queue
        self._stop_event = stop_event
        self._device_info = device_info
        self._label = label
        self.sample_rate: int = int(device_info["defaultSampleRate"])
        self.channels: int = device_info["maxInputChannels"]
        self.device_index: int = device_info["index"]
        self._stream: pyaudio.Stream | None = None
        self._pa: pyaudio.PyAudio | None = None

    def _callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: dict,
        status: int,
    ) -> tuple[None, int]:
        if in_data is not None:
            self._raw_queue.put_nowait(in_data)
        return (None, pyaudio.paContinue)

    def open(self, pa: pyaudio.PyAudio) -> bool:
        try:
            self._pa = pa
            self._stream = pa.open(
                format=STREAM_FORMAT,
                channels=self.channels,
                rate=self.sample_rate,
                frames_per_buffer=FRAMES_PER_BUFFER,
                input=True,
                input_device_index=self.device_index,
                stream_callback=self._callback,
            )
            logger.info(
                "%s capture opened: device=%d rate=%d ch=%d",
                self._label, self.device_index, self.sample_rate, self.channels,
            )
            return True
        except Exception:
            if self.channels > 1:
                logger.warning("%s failed with %d ch, retrying with 1 ch", self._label, self.channels)
                self.channels = 1
                return self.open(pa)
            logger.exception("Failed to open %s stream for device %d", self._label, self.device_index)
            return False

    def wait_until_stopped(self) -> None:
        if self._stream is None:
            return
        try:
            while not self._stop_event.is_set():
                if not self._stream.is_active():
                    logger.error("%s stream became inactive", self._label)
                    break
                self._stop_event.wait(timeout=0.5)
        finally:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
            logger.info("%s stream closed", self._label)
