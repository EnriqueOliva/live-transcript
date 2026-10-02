from types import SimpleNamespace

import numpy as np
import pytest

from livevox.stt.whisper_engine import WhisperEngine


def word(start, end, text):
    return SimpleNamespace(start=start, end=end, word=text)


def segment(start, end, text, words=None, no_speech_prob=0.01, avg_logprob=-0.2):
    return SimpleNamespace(start=start, end=end, text=text, words=words, no_speech_prob=no_speech_prob,
                           avg_logprob=avg_logprob)


class FakeModel:
    def __init__(self, segments):
        self.segments = segments
        self.kwargs = None
        self.yielded = 0

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs

        def generate():
            for item in self.segments:
                self.yielded += 1
                yield item

        return generate(), SimpleNamespace(language="es", language_probability=0.97)


def engine_with(model, device="cpu", fail_on=()):
    created = []

    def factory(name, device_name, compute_type):
        created.append((device_name, compute_type))
        if device_name in fail_on:
            raise RuntimeError("load failed")
        return model

    engine = WhisperEngine("turbo", model_factory=factory, device_resolver=lambda setting: (device, "float16"))
    engine.load()
    return engine, created


AUDIO = np.zeros(16000 * 10, dtype=np.float32)


class TestDecodingOptions:
    def test_nothing_inside_faster_whisper_may_discard_text(self):
        model = FakeModel([segment(0.0, 2.0, " hola")])
        engine, _ = engine_with(model)
        engine.transcribe(AUDIO, "es", word_timestamps=True)
        assert model.kwargs["no_speech_threshold"] is None
        assert model.kwargs["vad_filter"] is False
        assert model.kwargs["without_timestamps"] is True
        assert model.kwargs["condition_on_previous_text"] is False
        assert model.kwargs["beam_size"] == 5
        assert len(model.kwargs["temperature"]) > 1

    def test_partial_decoding_is_greedy(self):
        model = FakeModel([segment(0.0, 2.0, " hola")])
        engine, _ = engine_with(model)
        engine.transcribe(AUDIO, "es", fast=True)
        assert model.kwargs["beam_size"] == 1
        assert model.kwargs["temperature"] == 0.0

    def test_prompts_are_combined(self):
        model = FakeModel([segment(0.0, 2.0, " hola")])
        engine = WhisperEngine("turbo", initial_prompt="Dasbanq", model_factory=lambda *_: model,
                               device_resolver=lambda setting: ("cpu", "int8"))
        engine.load()
        engine.transcribe(AUDIO, "es", prompt="palabras previas")
        assert model.kwargs["initial_prompt"] == "Dasbanq palabras previas"


class TestContinuationControl:
    def test_continues_while_speech_remains_after_the_last_word(self):
        model = FakeModel([
            segment(0.0, 4.8, " perfecto", [word(0.5, 4.8, " perfecto")]),
            segment(4.8, 9.0, " o sea esta subida", [word(4.9, 9.0, " subida")]),
            segment(9.0, 9.02, " Gracias.", [word(9.0, 9.02, " Gracias.")]),
        ])
        engine, _ = engine_with(model)
        result = engine.transcribe(AUDIO, "es", word_timestamps=True, speech_end_seconds=9.1)
        assert result.text == "perfecto o sea esta subida"
        assert result.uncertain_text == ""
        assert model.yielded == 2

    def test_stops_once_the_words_reach_the_end_of_speech(self):
        model = FakeModel([
            segment(0.0, 8.64, " complicado", [word(1.0, 8.64, " complicado")]),
            segment(8.64, 8.68, " Gracias.", [word(8.64, 8.68, " Gracias.")]),
        ])
        engine, _ = engine_with(model)
        result = engine.transcribe(AUDIO, "es", word_timestamps=True, speech_end_seconds=8.7)
        assert result.text == "complicado"
        assert model.yielded == 1

    def test_text_too_long_for_its_audio_is_discarded(self):
        model = FakeModel([
            segment(0.0, 3.0, " uno", [word(0.0, 3.0, " uno")]),
            segment(3.0, 3.05, " Gracias.", [word(3.0, 3.05, " Gracias.")]),
            segment(3.05, 3.6, " otra", [word(3.05, 3.6, " otra")]),
        ])
        engine, _ = engine_with(model)
        result = engine.transcribe(AUDIO, "es", word_timestamps=True, speech_end_seconds=6.0)
        assert result.text == "uno"
        assert result.uncertain_text == ""
        assert model.yielded == 2

    def test_speakable_but_doubtful_continuation_is_kept_as_uncertain(self):
        model = FakeModel([
            segment(0.0, 3.0, " uno", [word(0.0, 3.0, " uno")]),
            segment(3.0, 3.6, " Gracias.", [word(3.0, 3.6, " Gracias.")], avg_logprob=-0.95),
            segment(3.6, 5.0, " y algo más", [word(3.6, 5.0, " más")], avg_logprob=-0.3),
        ])
        engine, _ = engine_with(model)
        result = engine.transcribe(AUDIO, "es", word_timestamps=True, speech_end_seconds=6.0)
        assert result.text == "uno"
        assert result.uncertain_text == "Gracias. y algo más"
        assert result.passes == 3

    def test_confident_continuation_joins_the_main_text(self):
        model = FakeModel([
            segment(0.0, 3.0, " uno", [word(0.0, 3.0, " uno")]),
            segment(3.0, 4.5, " dos tres", [word(3.0, 4.5, " tres")], avg_logprob=-0.4),
        ])
        engine, _ = engine_with(model)
        result = engine.transcribe(AUDIO, "es", word_timestamps=True, speech_end_seconds=4.6)
        assert result.text == "uno dos tres"
        assert result.uncertain_text == ""

    def test_non_speech_pieces_get_a_single_pass(self):
        model = FakeModel([
            segment(0.0, 1.0, " Gracias.", [word(0.0, 1.0, " Gracias.")]),
            segment(1.0, 2.0, " otra", [word(1.0, 2.0, " otra")]),
        ])
        engine, _ = engine_with(model)
        result = engine.transcribe(AUDIO, "es", word_timestamps=True, speech_end_seconds=None)
        assert result.text == "Gracias."
        assert model.yielded == 1

    def test_words_are_returned_with_their_times(self):
        model = FakeModel([segment(0.0, 2.0, " hola mundo", [word(0.1, 0.5, " hola"), word(0.6, 1.0, " mundo")])])
        engine, _ = engine_with(model)
        result = engine.transcribe(AUDIO, "es", word_timestamps=True, speech_end_seconds=1.0)
        assert [(item.start, item.end, item.text) for item in result.words] == [(0.1, 0.5, " hola"), (0.6, 1.0, " mundo")]

    def test_low_confidence_is_reported(self):
        model = FakeModel([segment(0.0, 2.0, " eh", no_speech_prob=0.9, avg_logprob=-1.5)])
        engine, _ = engine_with(model)
        assert engine.transcribe(AUDIO, "es").looks_like_non_speech


class TestDeviceFallback:
    def test_gpu_load_failure_falls_back_to_cpu(self):
        model = FakeModel([segment(0.0, 1.0, " hola")])
        engine, created = engine_with(model, device="cuda", fail_on=("cuda",))
        assert engine.device == "cpu"
        assert created == [("cuda", "float16"), ("cpu", "int8")]

    def test_runtime_failure_on_gpu_moves_the_session_to_cpu(self):
        model = FakeModel([segment(0.0, 1.0, " hola")])
        engine, created = engine_with(model, device="cuda")
        engine.recover_after_failure()
        assert engine.device == "cpu"
        assert created[-1] == ("cpu", "int8")

    def test_transcribing_without_a_model_fails_loudly(self):
        engine = WhisperEngine("turbo", model_factory=lambda *_: None, device_resolver=lambda setting: ("cpu", "int8"))
        with pytest.raises(RuntimeError):
            engine.transcribe(AUDIO, "es")
