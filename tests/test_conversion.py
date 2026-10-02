import numpy as np
import pytest

from whisper_transcriber.audio.conversion import StreamConverter
from whisper_transcriber.audio.timeline import SAMPLE_RATE


def capture_bytes(duration, rate, channels, seed=0):
    generator = np.random.default_rng(seed)
    frames = int(duration * rate)
    time_axis = np.arange(frames) / rate
    left = 0.4 * np.sin(2 * np.pi * 440 * time_axis) + 0.05 * generator.standard_normal(frames)
    right = 0.4 * np.sin(2 * np.pi * 660 * time_axis)
    columns = [left, right][:channels]
    interleaved = np.column_stack(columns).reshape(-1)
    return (np.clip(interleaved, -1, 1) * 32767).astype(np.int16).tobytes()


def convert_in_blocks(raw, rate, channels, block_bytes):
    converter = StreamConverter(rate, channels)
    parts = [converter.convert(raw[start : start + block_bytes]) for start in range(0, len(raw), block_bytes)]
    parts.append(converter.flush())
    return np.concatenate(parts)


class TestStreamingConversion:
    @pytest.mark.parametrize(("rate", "channels"), [(48000, 2), (44100, 2), (44100, 1), (16000, 1), (16000, 2), (32000, 2)])
    def test_block_size_never_changes_the_output(self, rate, channels):
        raw = capture_bytes(3.0, rate, channels)
        whole = convert_in_blocks(raw, rate, channels, len(raw))
        for block_bytes in (4096, 1000, 2 * channels * 441, 7):
            np.testing.assert_array_equal(convert_in_blocks(raw, rate, channels, block_bytes), whole)

    @pytest.mark.parametrize(("rate", "channels"), [(48000, 2), (44100, 2), (96000, 2), (22050, 1)])
    def test_every_second_of_capture_becomes_a_second_of_audio(self, rate, channels):
        duration = 7.0
        output = convert_in_blocks(capture_bytes(duration, rate, channels), rate, channels, 8192)
        assert abs(output.size - duration * SAMPLE_RATE) <= 1
        assert output.dtype == np.float32

    def test_stereo_is_averaged_not_summed(self):
        frames = 16000
        left = np.full(frames, 16384, dtype=np.int16)
        right = np.full(frames, 8192, dtype=np.int16)
        raw = np.column_stack([left, right]).reshape(-1).tobytes()
        output = convert_in_blocks(raw, SAMPLE_RATE, 2, len(raw))
        np.testing.assert_allclose(output, (0.5 + 0.25) / 2, rtol=1e-6)

    def test_a_split_frame_is_carried_to_the_next_block(self):
        converter = StreamConverter(SAMPLE_RATE, 2)
        raw = capture_bytes(0.1, SAMPLE_RATE, 2)
        first = converter.convert(raw[:5])
        second = converter.convert(raw[5:])
        assert first.size == 1
        assert first.size + second.size == len(raw) // 4

    def test_resampled_signal_keeps_its_content(self):
        rate = 48000
        time_axis = np.arange(rate * 2) / rate
        signal = (0.5 * np.sin(2 * np.pi * 1000 * time_axis) * 32767).astype(np.int16)
        output = convert_in_blocks(signal.tobytes(), rate, 1, 4096)
        spectrum = np.abs(np.fft.rfft(output[1000:-1000]))
        peak_frequency = np.argmax(spectrum) * SAMPLE_RATE / (output.size - 2000)
        assert abs(peak_frequency - 1000) < 5
