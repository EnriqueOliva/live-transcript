import queue
import threading
from itertools import pairwise

import numpy as np

from tests.helpers import EnergyClassifier, RecordingEvents
from whisper_transcriber.audio.capture import (
    CaptureClosed,
    CaptureData,
    CaptureFinished,
    CaptureFormat,
    CaptureNotice,
)
from whisper_transcriber.audio.pipeline import INPUT_OVERFLOW_FLAG, AudioPipeline, compute_band_levels
from whisper_transcriber.io.recording import WavRecorder
from whisper_transcriber.session.messages import EndOfStream, Notice, PipelineStatistics
from whisper_transcriber.stt.segmenter import Piece, SpeechSegmenter

CAPTURE_RATE = 48000


def capture_audio(duration, rate=CAPTURE_RATE, channels=2):
    frames = int(duration * rate)
    time_axis = np.arange(frames) / rate
    envelope = (np.sin(2 * np.pi * 0.25 * time_axis) > -0.2).astype(np.float64)
    mono = 0.3 * envelope * np.sin(2 * np.pi * 300 * time_axis)
    return (np.repeat(mono[:, None], channels, axis=1).reshape(-1) * 32767).astype(np.int16)


def data_messages(source, generation, interleaved, channels, frames_per_block=2048, status=0):
    block = frames_per_block * channels
    return [
        CaptureData(source, generation, interleaved[start : start + block].tobytes(), status)
        for start in range(0, interleaved.size, block)
    ]


def run_pipeline(messages, tmp_path, segmenter=None, recorder=True):
    raw_queue = queue.SimpleQueue()
    piece_queue = queue.Queue()
    statistics = PipelineStatistics()
    events = RecordingEvents()
    wav = None
    if recorder:
        wav = WavRecorder(tmp_path / "recording.wav")
        wav.open()
    pipeline = AudioPipeline(raw_queue, piece_queue, segmenter or SpeechSegmenter(EnergyClassifier()), wav,
                             events, statistics)
    thread = threading.Thread(target=pipeline.run)
    thread.start()
    for message in messages:
        raw_queue.put(message)
    thread.join(timeout=60)
    assert not thread.is_alive()
    items = []
    while not piece_queue.empty():
        items.append(piece_queue.get_nowait())
    return items, statistics, events


class TestEndToEndAccounting:
    def test_every_captured_frame_reaches_the_recording_and_the_pieces(self, tmp_path):
        interleaved = capture_audio(31.3)
        messages = [CaptureFormat("loopback", 1, CAPTURE_RATE, 2, "Speakers")]
        messages += data_messages("loopback", 1, interleaved, 2, frames_per_block=1999)
        messages += [CaptureClosed("loopback", 1), CaptureFinished()]
        items, statistics, events = run_pipeline(messages, tmp_path)
        pieces = [item for item in items if isinstance(item, Piece)]
        expected = round(31.3 * 16000)
        assert abs(statistics.captured_samples - expected) <= 1
        assert statistics.recorded_samples == statistics.captured_samples
        assert pieces[0].start_sample == 0
        assert pieces[-1].end_sample == statistics.captured_samples
        for previous, current in pairwise(pieces):
            assert current.start_sample + current.overlap_samples == previous.end_sample
        assert isinstance(items[-1], EndOfStream)
        assert statistics.devices == ["Speakers"]
        assert events.levels > 0

    def test_device_switch_keeps_both_devices_audio(self, tmp_path):
        first = capture_audio(4.0, rate=48000)
        second = capture_audio(5.0, rate=44100)
        messages = [CaptureFormat("loopback", 1, 48000, 2, "Speakers")]
        messages += data_messages("loopback", 1, first, 2)
        messages += [CaptureClosed("loopback", 1)]
        messages += [CaptureNotice("Audio capture reconnected", device_switch=True)]
        messages += [CaptureFormat("loopback", 2, 44100, 2, "Headphones")]
        messages += data_messages("loopback", 2, second, 2)
        messages += [CaptureFinished()]
        items, statistics, events = run_pipeline(messages, tmp_path)
        assert abs(statistics.captured_samples - 9 * 16000) <= 2
        assert statistics.device_switches == 1
        assert statistics.devices == ["Speakers", "Headphones"]
        assert any(isinstance(item, Notice) for item in items)
        assert events.notices == ["Audio capture reconnected"]

    def test_microphone_and_loopback_are_both_kept(self, tmp_path):
        loopback = capture_audio(3.0)
        microphone = capture_audio(3.0, rate=16000, channels=1)
        messages = [CaptureFormat("loopback", 1, CAPTURE_RATE, 2, "Speakers"),
                    CaptureFormat("microphone", 1, 16000, 1, "Mic")]
        loopback_blocks = data_messages("loopback", 1, loopback, 2)
        microphone_blocks = data_messages("microphone", 1, microphone, 1, frames_per_block=700)
        for index in range(max(len(loopback_blocks), len(microphone_blocks))):
            if index < len(loopback_blocks):
                messages.append(loopback_blocks[index])
            if index < len(microphone_blocks):
                messages.append(microphone_blocks[index])
        messages += [CaptureFinished()]
        _, statistics, _ = run_pipeline(messages, tmp_path)
        assert abs(statistics.captured_samples - 3 * 16000) <= 2

    def test_overflow_flags_are_counted_and_reported(self, tmp_path):
        interleaved = capture_audio(1.0)
        messages = [CaptureFormat("loopback", 1, CAPTURE_RATE, 2, "Speakers")]
        messages += data_messages("loopback", 1, interleaved, 2, status=INPUT_OVERFLOW_FLAG)
        messages += [CaptureFinished()]
        _, statistics, events = run_pipeline(messages, tmp_path)
        assert statistics.overflow_events == len(messages) - 2
        assert len(events.notices) == 1

    def test_audio_without_a_known_format_does_not_crash_the_pipeline(self, tmp_path):
        messages = [CaptureData("loopback", 9, b"\x00\x01" * 100, 0), CaptureFinished()]
        items, _, _ = run_pipeline(messages, tmp_path)
        assert isinstance(items[-1], EndOfStream)


class TestFailureIsolation:
    def test_end_of_stream_is_always_sent_even_if_segmentation_breaks(self, tmp_path):
        class BrokenSegmenter:
            def feed(self, samples):
                raise RuntimeError("boom")

            def finish(self):
                raise RuntimeError("boom")

        interleaved = capture_audio(1.0)
        messages = [CaptureFormat("loopback", 1, CAPTURE_RATE, 2, "Speakers")]
        messages += data_messages("loopback", 1, interleaved, 2)
        messages += [CaptureFinished()]
        items, statistics, events = run_pipeline(messages, tmp_path, segmenter=BrokenSegmenter())
        assert isinstance(items[-1], EndOfStream)
        assert events.errors
        assert statistics.recorded_samples > 0


class TestLevels:
    def test_levels_are_bounded(self):
        levels = compute_band_levels(np.ones(2048, dtype=np.float32))
        assert len(levels) == 24
        assert all(0.0 <= level <= 1.0 for level in levels)

    def test_short_input_is_silent(self):
        assert compute_band_levels(np.zeros(10, dtype=np.float32)) == [0.0] * 24
