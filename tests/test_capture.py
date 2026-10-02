import queue
import threading
import time

import pyaudiowpatch as pyaudio
import pytest

from livevox.audio import capture
from livevox.audio.capture import (
    CaptureClosed,
    CaptureData,
    CaptureFinished,
    CaptureFormat,
    CaptureManager,
    CaptureNotice,
)

LOOPBACK = {"index": 5, "name": "Speakers [Loopback]", "defaultSampleRate": 48000.0, "maxInputChannels": 2,
            "maxOutputChannels": 0, "isLoopbackDevice": True}
RENDER = {"index": 4, "name": "Speakers", "defaultSampleRate": 48000.0, "maxInputChannels": 0,
          "maxOutputChannels": 2, "isLoopbackDevice": False}
HEADPHONES_LOOPBACK = {"index": 7, "name": "Headphones [Loopback]", "defaultSampleRate": 44100.0,
                       "maxInputChannels": 2, "maxOutputChannels": 0, "isLoopbackDevice": True}
MICROPHONE = {"index": 2, "name": "Mic", "defaultSampleRate": 16000.0, "maxInputChannels": 1,
              "maxOutputChannels": 0, "isLoopbackDevice": False}


class FakeStream:
    def __init__(self, keyword_arguments):
        self.keyword_arguments = keyword_arguments
        self.callback = keyword_arguments["stream_callback"]
        self.active = True
        self.closed = False

    def is_active(self):
        return self.active

    def stop_stream(self):
        self.active = False

    def close(self):
        self.closed = True


class FakePortAudio:
    def __init__(self, world):
        self.world = world
        self.streams = []
        world.instances.append(self)

    def get_default_wasapi_loopback(self):
        if self.world.loopback is None:
            raise LookupError("none")
        return self.world.loopback

    def get_host_api_info_by_type(self, host_type):
        return {"index": 0, "deviceCount": 1}

    def get_device_info_by_host_api_device_index(self, host_index, device_index):
        return RENDER

    def get_default_input_device_info(self):
        return MICROPHONE

    def open(self, **keyword_arguments):
        if self.world.fail_microphone and keyword_arguments.get("input_device_index") == MICROPHONE["index"]:
            raise OSError("mic busy")
        stream = FakeStream(keyword_arguments)
        self.streams.append(stream)
        return stream

    def terminate(self):
        self.world.terminated += 1


class FakeWatcher:
    def __init__(self, world):
        self.world = world

    def current_id(self):
        return self.world.endpoint

    def close(self):
        self.world.watcher_closed = True


class World:
    def __init__(self):
        self.loopback = LOOPBACK
        self.endpoint = "speakers"
        self.instances = []
        self.terminated = 0
        self.fail_microphone = False
        self.watcher_closed = False


@pytest.fixture(autouse=True)
def fast_supervisor(monkeypatch):
    monkeypatch.setattr(capture, "SUPERVISOR_POLL_SECONDS", 0.01)
    monkeypatch.setattr(capture, "ENDPOINT_POLL_SECONDS", 0.02)
    monkeypatch.setattr(capture, "REOPEN_RETRY_SECONDS", 0.02)


def start(world, record_microphone=False):
    raw_queue = queue.SimpleQueue()
    manager = CaptureManager(raw_queue, record_microphone, port_audio_factory=lambda: FakePortAudio(world),
                             watcher_factory=lambda: FakeWatcher(world))
    thread = threading.Thread(target=manager.run, daemon=True)
    thread.start()
    assert manager.wait_until_ready(5)
    return manager, raw_queue, thread


def stop(manager, thread):
    manager.request_stop()
    thread.join(5)
    assert not thread.is_alive()


def drain(raw_queue):
    items = []
    while True:
        try:
            items.append(raw_queue.get_nowait())
        except queue.Empty:
            return items


def wait_for(condition, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def input_streams(port_audio):
    return [stream for stream in port_audio.streams if stream.keyword_arguments.get("input")]


class TestCapture:
    def test_callback_audio_reaches_the_queue_and_stop_closes_in_order(self):
        world = World()
        manager, raw_queue, thread = start(world)
        loopback_stream = input_streams(world.instances[0])[0]
        loopback_stream.callback(b"\x01\x00" * 4, 2, {}, pyaudio.paInputOverflow)
        stop(manager, thread)
        items = drain(raw_queue)
        assert isinstance(items[0], CaptureFormat)
        assert items[0].sample_rate == 48000
        data = [item for item in items if isinstance(item, CaptureData)]
        assert data[0].data == b"\x01\x00" * 4
        assert data[0].status == pyaudio.paInputOverflow
        assert isinstance(items[-2], CaptureClosed)
        assert isinstance(items[-1], CaptureFinished)
        assert loopback_stream.closed
        assert world.terminated == 1
        assert world.watcher_closed

    def test_keep_alive_plays_silence_on_the_matching_output(self):
        world = World()
        manager, _, thread = start(world)
        output_streams = [stream for stream in world.instances[0].streams if stream.keyword_arguments.get("output")]
        assert len(output_streams) == 1
        assert output_streams[0].keyword_arguments["output_device_index"] == RENDER["index"]
        silence, flag = output_streams[0].callback(None, 100, {}, 0)
        assert silence == bytes(100 * 2 * 2)
        assert flag == pyaudio.paContinue
        stop(manager, thread)

    def test_default_output_change_moves_the_capture(self):
        world = World()
        manager, raw_queue, thread = start(world)
        world.loopback = HEADPHONES_LOOPBACK
        world.endpoint = "headphones"
        assert wait_for(lambda: len(world.instances) >= 2)
        stop(manager, thread)
        items = drain(raw_queue)
        formats = [item for item in items if isinstance(item, CaptureFormat)]
        assert [item.device_name for item in formats] == ["Speakers [Loopback]", "Headphones [Loopback]"]
        assert formats[1].generation > formats[0].generation
        notices = [item for item in items if isinstance(item, CaptureNotice)]
        assert notices and notices[0].device_switch
        closed_index = next(index for index, item in enumerate(items) if isinstance(item, CaptureClosed))
        second_format_index = items.index(formats[1])
        assert closed_index < second_format_index

    def test_dead_stream_is_reopened(self):
        world = World()
        manager, raw_queue, thread = start(world)
        input_streams(world.instances[0])[0].active = False
        assert wait_for(lambda: len(world.instances) >= 2)
        stop(manager, thread)
        assert len([item for item in drain(raw_queue) if isinstance(item, CaptureFormat)]) >= 2

    def test_missing_output_device_is_reported_and_retried(self):
        world = World()
        world.loopback = None
        manager, raw_queue, thread = start(world)
        assert manager.startup_error is not None
        world.loopback = LOOPBACK
        assert wait_for(lambda: any(input_streams(instance) for instance in world.instances))
        stop(manager, thread)
        assert any(isinstance(item, CaptureFormat) for item in drain(raw_queue))

    def test_microphone_failure_falls_back_to_system_audio(self):
        world = World()
        world.fail_microphone = True
        manager, raw_queue, thread = start(world, record_microphone=True)
        stop(manager, thread)
        items = drain(raw_queue)
        assert [item.source for item in items if isinstance(item, CaptureFormat)] == ["loopback"]
        assert any(isinstance(item, CaptureNotice) and "microphone" in item.text for item in items)

    def test_microphone_is_opened_when_requested(self):
        world = World()
        manager, raw_queue, thread = start(world, record_microphone=True)
        stop(manager, thread)
        sources = [item.source for item in drain(raw_queue) if isinstance(item, CaptureFormat)]
        assert sources == ["loopback", "microphone"]
