from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from livevox.audio.timeline import SAMPLE_RATE
from livevox.session.events import TranscriptLine
from livevox.stt.handoff import TimedWord
from livevox.stt.whisper_engine import TranscriptionResult

SPEECH_PROBABILITY = 0.9
SILENCE_PROBABILITY = 0.05
SPEECH_ENERGY_THRESHOLD = 0.01


class EnergyClassifier:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        self.calls += 1
        energy = np.sqrt(np.mean(np.square(frames), axis=1))
        return np.where(energy > SPEECH_ENERGY_THRESHOLD, SPEECH_PROBABILITY, SILENCE_PROBABILITY).astype(np.float32)


def seconds(value: float) -> int:
    return round(value * SAMPLE_RATE)


def tone(duration: float, amplitude: float = 0.3, frequency: float = 220.0, start_phase_sample: int = 0) -> np.ndarray:
    sample_count = seconds(duration)
    time_axis = (np.arange(sample_count) + start_phase_sample) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * time_axis)).astype(np.float32)


def silence(duration: float) -> np.ndarray:
    return np.zeros(seconds(duration), dtype=np.float32)


def noise(duration: float, amplitude: float, seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return (amplitude * generator.uniform(-1.0, 1.0, seconds(duration))).astype(np.float32)


def build(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32)


def feed_in_blocks(segmenter, audio: np.ndarray, block_sizes: list[int]) -> list:
    pieces = []
    position = 0
    index = 0
    while position < audio.size:
        size = block_sizes[index % len(block_sizes)]
        index += 1
        pieces.extend(segmenter.feed(audio[position : position + size]))
        position += size
    pieces.extend(segmenter.finish())
    return pieces


def result(text: str = "hola", words: list[TimedWord] | None = None, language: str = "es",
           language_probability: float = 0.99, looks_like_non_speech: bool = False) -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        language=language,
        language_probability=language_probability,
        looks_like_non_speech=looks_like_non_speech,
        words=list(words or []),
    )


def words_for(text: str, start: float, step: float = 0.3) -> list[TimedWord]:
    timed = []
    position = start
    for word in text.split():
        timed.append(TimedWord(start=position, end=position + step * 0.8, text=f" {word}"))
        position += step
    return timed


@dataclass
class EngineCall:
    audio: np.ndarray
    language: str | None
    word_timestamps: bool
    fast: bool
    prompt: str | None
    speech_end_seconds: float | None = None


@dataclass
class FakeEngine:
    responses: list = field(default_factory=list)
    default: TranscriptionResult | None = None
    device: str = "cpu"
    compute_type: str = "int8"
    model_name: str = "fake"
    fail_load: bool = False
    is_loaded: bool = False
    calls: list[EngineCall] = field(default_factory=list)
    recoveries: int = 0

    def load(self) -> None:
        if self.fail_load:
            raise RuntimeError("cannot load")
        self.is_loaded = True

    def recover_after_failure(self) -> None:
        self.recoveries += 1

    def transcribe(self, audio: np.ndarray, language: str | None, word_timestamps: bool = False,
                   fast: bool = False, prompt: str | None = None,
                   speech_end_seconds: float | None = None) -> TranscriptionResult:
        self.calls.append(
            EngineCall(np.array(audio, copy=True), language, word_timestamps, fast, prompt, speech_end_seconds),
        )
        if self.responses:
            response = self.responses.pop(0)
        else:
            response = self.default if self.default is not None else result()
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return response(audio)
        return response


@dataclass
class RecordingEvents:
    lines: list[TranscriptLine] = field(default_factory=list)
    partials: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    finished_summaries: list[str] = field(default_factory=list)
    progress_updates: list[tuple[float, float]] = field(default_factory=list)
    levels: int = 0

    def status(self, state: str, detail: str = "") -> None:
        self.messages.append(f"status:{state}")

    def message(self, text: str) -> None:
        self.messages.append(text)

    def transcript_line(self, line: TranscriptLine) -> None:
        self.lines.append(line)

    def partial_text(self, text: str) -> None:
        self.partials.append(text)

    def progress(self, captured_seconds: float, transcribed_seconds: float) -> None:
        self.progress_updates.append((captured_seconds, transcribed_seconds))

    def audio_levels(self, levels: list[float]) -> None:
        self.levels += 1

    def notice(self, text: str) -> None:
        self.notices.append(text)

    def error(self, text: str) -> None:
        self.errors.append(text)

    def finished(self, summary: str) -> None:
        self.finished_summaries.append(summary)
