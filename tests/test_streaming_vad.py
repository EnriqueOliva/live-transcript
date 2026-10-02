import numpy as np

from whisper_transcriber.stt.vad import FRAME_SAMPLES, StreamingVad


def speech_like_audio(seconds_count=12, seed=1):
    generator = np.random.default_rng(seed)
    sample_count = seconds_count * 16000 // FRAME_SAMPLES * FRAME_SAMPLES
    time_axis = np.arange(sample_count) / 16000
    envelope = (np.sin(2 * np.pi * 0.5 * time_axis) > 0).astype(np.float32)
    voiced = 0.3 * np.sin(2 * np.pi * 180 * time_axis) * (1 + 0.5 * np.sin(2 * np.pi * 4 * time_axis))
    return (envelope * voiced + 0.01 * generator.standard_normal(sample_count)).astype(np.float32)


class TestStreamingVad:
    def test_streaming_matches_the_batch_model_exactly(self):
        from faster_whisper.vad import get_vad_model

        audio = speech_like_audio()
        batch = np.asarray(get_vad_model()(audio)).reshape(-1)
        streaming = StreamingVad()
        frames = audio.reshape(-1, FRAME_SAMPLES)
        outputs = []
        position = 0
        sizes = [1, 5, 2, 37, 3, 100]
        index = 0
        while position < frames.shape[0]:
            count = sizes[index % len(sizes)]
            index += 1
            outputs.append(streaming(frames[position : position + count]))
            position += count
        np.testing.assert_allclose(np.concatenate(outputs), batch, atol=1e-6)

    def test_empty_input_returns_no_probabilities(self):
        assert StreamingVad()(np.zeros((0, FRAME_SAMPLES), dtype=np.float32)).size == 0

    def test_probabilities_are_bounded(self):
        probabilities = StreamingVad()(speech_like_audio(4).reshape(-1, FRAME_SAMPLES))
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
