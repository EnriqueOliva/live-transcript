import numpy as np

from livevox.audio.mixer import SourceMixer
from livevox.audio.timeline import SAMPLE_RATE


def drain(mixer):
    parts = [mixer.pull()]
    parts.append(mixer.flush())
    return np.concatenate(parts)


class TestSingleSource:
    def test_passes_audio_through_untouched(self):
        mixer = SourceMixer()
        audio = np.linspace(-0.9, 0.9, 5000, dtype=np.float32)
        mixer.push("loopback", audio[:1234])
        first = mixer.pull()
        mixer.push("loopback", audio[1234:])
        second = mixer.pull()
        np.testing.assert_array_equal(np.concatenate([first, second]), audio)
        assert mixer.flush().size == 0


class TestTwoSources:
    def test_aligned_sources_are_summed(self):
        mixer = SourceMixer()
        mixer.push("loopback", np.full(100, 0.1, dtype=np.float32))
        mixer.push("microphone", np.full(100, 0.2, dtype=np.float32))
        np.testing.assert_allclose(mixer.pull(), 0.3, rtol=1e-6)

    def test_waits_for_the_slower_source_within_the_skew(self):
        mixer = SourceMixer(maximum_skew_seconds=0.5)
        mixer.push("loopback", np.full(SAMPLE_RATE // 4, 0.1, dtype=np.float32))
        mixer.add_source("microphone")
        assert mixer.pull().size == 0

    def test_never_waits_beyond_the_skew(self):
        mixer = SourceMixer(maximum_skew_seconds=0.5)
        mixer.add_source("microphone")
        mixer.push("loopback", np.full(SAMPLE_RATE, 0.1, dtype=np.float32))
        assert mixer.pull().size == SAMPLE_RATE // 2

    def test_no_sample_from_either_source_is_lost(self):
        generator = np.random.default_rng(3)
        mixer = SourceMixer(maximum_skew_seconds=0.25)
        totals = {"loopback": 0.0, "microphone": 0.0}
        outputs = []
        for _ in range(400):
            name = "loopback" if generator.random() < 0.7 else "microphone"
            block = generator.uniform(0.0, 0.001, int(generator.integers(1, 3000))).astype(np.float32)
            totals[name] += float(block.astype(np.float64).sum())
            mixer.push(name, block)
            outputs.append(mixer.pull())
        outputs.append(mixer.flush())
        mixed = np.concatenate(outputs).astype(np.float64)
        assert abs(mixed.sum() - sum(totals.values())) < 1e-3

    def test_loud_overlap_is_clipped_into_range(self):
        mixer = SourceMixer()
        mixer.push("loopback", np.full(10, 0.8, dtype=np.float32))
        mixer.push("microphone", np.full(10, 0.8, dtype=np.float32))
        assert np.all(drain(mixer) <= 1.0)
