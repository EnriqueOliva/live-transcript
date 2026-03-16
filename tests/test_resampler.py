import numpy as np

from whisper_transcriber.audio.resampler import convert_to_whisper_format


class TestStereoToMono:
    def test_stereo_to_mono_halves_samples(self, sample_audio_int16_stereo):
        raw_bytes, rate, channels = sample_audio_int16_stereo
        result = convert_to_whisper_format(raw_bytes, rate, channels)
        expected_samples = int(rate * 1.0 * 16000 / rate)
        assert abs(len(result) - expected_samples) <= 1


class TestResampleRatio:
    def test_48k_to_16k_ratio(self):
        sample_rate = 48000
        duration = 1.0
        samples = int(sample_rate * duration)
        mono = np.zeros(samples, dtype=np.int16)
        raw_bytes = mono.tobytes()
        result = convert_to_whisper_format(raw_bytes, sample_rate, 1)
        expected = int(samples * 16000 / sample_rate)
        assert abs(len(result) - expected) <= 1


class TestOutputFormat:
    def test_output_dtype_float32(self, sample_audio_int16_stereo):
        raw_bytes, rate, channels = sample_audio_int16_stereo
        result = convert_to_whisper_format(raw_bytes, rate, channels)
        assert result.dtype == np.float32

    def test_output_range_normalized(self, sample_audio_int16_stereo):
        raw_bytes, rate, channels = sample_audio_int16_stereo
        result = convert_to_whisper_format(raw_bytes, rate, channels)
        assert result.min() >= -1.1
        assert result.max() <= 1.1


class TestPassthrough:
    def test_already_16k_mono_passthrough(self):
        sample_rate = 16000
        samples = 16000
        tone = (np.sin(np.linspace(0, 2 * np.pi * 440, samples)) * 16000).astype(np.int16)
        raw_bytes = tone.tobytes()
        result = convert_to_whisper_format(raw_bytes, sample_rate, 1)
        assert len(result) == samples
        assert result.dtype == np.float32
