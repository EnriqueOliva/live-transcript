import wave

import numpy as np

from whisper_transcriber.io import recording
from whisper_transcriber.io.recording import WavRecorder, float_to_int16


def read_wav(path):
    with wave.open(str(path)) as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


class TestWavRecorder:
    def test_round_trip_keeps_every_sample(self, tmp_path):
        recorder = WavRecorder(tmp_path / "recording.wav")
        recorder.open()
        audio = np.linspace(-0.5, 0.5, 48000, dtype=np.float32)
        for start in range(0, audio.size, 777):
            recorder.write(audio[start : start + 777])
        recorder.close()
        samples = read_wav(tmp_path / "recording.wav")
        np.testing.assert_array_equal(samples, float_to_int16(audio))
        assert recorder.samples_written == audio.size

    def test_file_is_readable_while_still_recording(self, tmp_path, monkeypatch):
        monkeypatch.setattr(recording, "HEADER_REFRESH_SECONDS", 0.0)
        recorder = WavRecorder(tmp_path / "recording.wav")
        recorder.open()
        recorder.write(np.full(16000, 0.25, dtype=np.float32))
        samples = read_wav(tmp_path / "recording.wav")
        assert samples.size == 16000
        recorder.close()

    def test_out_of_range_samples_are_clipped(self):
        converted = float_to_int16(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32))
        assert converted.tolist() == [-32768, -32768, 0, 32767, 32767]

    def test_writes_before_open_are_ignored(self, tmp_path):
        recorder = WavRecorder(tmp_path / "recording.wav")
        recorder.write(np.ones(10, dtype=np.float32))
        assert recorder.samples_written == 0
