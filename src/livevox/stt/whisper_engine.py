from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from livevox.stt.cuda_runtime import CPU_COMPUTE_TYPE, compute_type_for, resolve_device
from livevox.stt.handoff import TimedWord

logger = logging.getLogger(__name__)

FINAL_BEAM_SIZE = 5
PARTIAL_BEAM_SIZE = 1
FALLBACK_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
GREEDY_TEMPERATURE = 0.0
COMPRESSION_RATIO_THRESHOLD = 2.4
LOG_PROBABILITY_THRESHOLD = -1.0
NON_SPEECH_PROBABILITY_THRESHOLD = 0.6
CONTINUATION_GUARD_SECONDS = 0.2
MAXIMUM_LETTERS_PER_SECOND = 40.0
MINIMUM_ALIGNED_SECONDS = 0.02
UNCERTAIN_CONTINUATION_LOG_PROBABILITY = -0.8

ModelFactory = Callable[[str, str, str], Any]
DeviceResolver = Callable[[str], tuple[str, str]]


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    looks_like_non_speech: bool
    words: list[TimedWord] = field(default_factory=list)
    uncertain_text: str = ""
    passes: int = 1


def _create_whisper_model(model_name: str, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


class WhisperEngine:
    def __init__(
        self,
        model_name: str,
        compute_type_setting: str = "auto",
        initial_prompt: str = "",
        hotwords: str = "",
        model_factory: ModelFactory = _create_whisper_model,
        device_resolver: DeviceResolver = resolve_device,
    ) -> None:
        self._model_name = model_name
        self._compute_type_setting = compute_type_setting
        self._initial_prompt = initial_prompt or None
        self._hotwords = hotwords or None
        self._model_factory = model_factory
        self._device_resolver = device_resolver
        self._model: Any = None
        self._device = "cpu"
        self._compute_type = CPU_COMPUTE_TYPE

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def compute_type(self) -> str:
        return self._compute_type

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        self._device, self._compute_type = self._device_resolver(self._compute_type_setting)
        logger.info("Loading model '%s' on %s (%s)", self._model_name, self._device, self._compute_type)
        try:
            self._model = self._model_factory(self._model_name, self._device, self._compute_type)
        except Exception:
            if self._device == "cuda":
                logger.exception("Loading on the GPU failed, falling back to CPU")
                self._switch_to_cpu()
            else:
                raise
        logger.info("Model loaded: %s on %s (%s)", self._model_name, self._device, self._compute_type)

    def recover_after_failure(self) -> None:
        if self._device == "cuda":
            logger.warning("Transcription failed on the GPU, switching this session to CPU")
            self._switch_to_cpu()
        else:
            gc.collect()

    def unload(self) -> None:
        self._model = None
        gc.collect()

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None,
        word_timestamps: bool = False,
        fast: bool = False,
        prompt: str | None = None,
        speech_end_seconds: float | None = None,
    ) -> TranscriptionResult:
        if self._model is None:
            raise RuntimeError("model not loaded")
        segments_iterator, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=PARTIAL_BEAM_SIZE if fast else FINAL_BEAM_SIZE,
            best_of=PARTIAL_BEAM_SIZE if fast else FINAL_BEAM_SIZE,
            temperature=GREEDY_TEMPERATURE if fast else FALLBACK_TEMPERATURES,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=LOG_PROBABILITY_THRESHOLD,
            no_speech_threshold=None,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
            word_timestamps=word_timestamps,
            initial_prompt=self._combined_prompt(prompt),
            hotwords=self._hotwords,
        )
        collected = _collect_segments(segments_iterator, word_timestamps, speech_end_seconds)
        segments = collected.certain
        text = "".join(segment.text for segment in segments).strip()
        words = [
            TimedWord(start=float(word.start), end=float(word.end), text=word.word)
            for segment in segments
            for word in (segment.words or [])
        ]
        looks_like_non_speech = any(
            segment.no_speech_prob > NON_SPEECH_PROBABILITY_THRESHOLD
            and segment.avg_logprob < LOG_PROBABILITY_THRESHOLD
            for segment in segments
        )
        return TranscriptionResult(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
            looks_like_non_speech=looks_like_non_speech,
            words=words,
            uncertain_text="".join(segment.text for segment in collected.uncertain).strip(),
            passes=collected.passes,
        )

    def _combined_prompt(self, prompt: str | None) -> str | None:
        parts = [part for part in (self._initial_prompt, prompt) if part]
        return " ".join(parts) if parts else None

    def _switch_to_cpu(self) -> None:
        self._model = None
        gc.collect()
        self._device = "cpu"
        self._compute_type = compute_type_for("cpu", self._compute_type_setting)
        self._model = self._model_factory(self._model_name, self._device, self._compute_type)


def _segment_end(segment: Any) -> float:
    if segment.words:
        return float(segment.words[-1].end)
    else:
        return float(segment.end)


def _is_speakable(segment: Any) -> bool:
    letters = sum(character.isalnum() for character in str(segment.text))
    duration = max(float(segment.end - segment.start), MINIMUM_ALIGNED_SECONDS)
    letters_per_second: float = letters / duration
    return letters_per_second <= MAXIMUM_LETTERS_PER_SECOND


@dataclass
class _CollectedSegments:
    certain: list[Any] = field(default_factory=list)
    uncertain: list[Any] = field(default_factory=list)
    passes: int = 0


def _collect_segments(
    segments_iterator: Any, word_timestamps: bool, speech_end_seconds: float | None,
) -> _CollectedSegments:
    collected = _CollectedSegments()
    for segment in segments_iterator:
        collected.passes += 1
        is_continuation = collected.passes > 1
        if is_continuation and not _is_speakable(segment):
            logger.debug("Discarded an unspeakable continuation %r over %.2fs", segment.text, segment.end - segment.start)
            break
        elif is_continuation and (collected.uncertain or segment.avg_logprob < UNCERTAIN_CONTINUATION_LOG_PROBABILITY):
            collected.uncertain.append(segment)
        else:
            collected.certain.append(segment)
        speech_remains = (
            word_timestamps
            and speech_end_seconds is not None
            and _segment_end(segment) < speech_end_seconds - CONTINUATION_GUARD_SECONDS
        )
        if not speech_remains:
            break
    return collected
