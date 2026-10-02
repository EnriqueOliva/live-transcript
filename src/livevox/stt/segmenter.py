from __future__ import annotations

import enum
import logging
import math
import threading
from dataclasses import dataclass

import numpy as np

from livevox.audio.quiet import find_quiet_point, window_starts
from livevox.audio.timeline import INT16_FULL_SCALE, samples_to_seconds, seconds_to_samples
from livevox.stt.vad import FRAME_SAMPLES, FrameClassifier

logger = logging.getLogger(__name__)

DIGITAL_SILENCE_PEAK = 4 / INT16_FULL_SCALE
INITIAL_BUFFER_CAPACITY = seconds_to_samples(32.0)
PROBABILITY_WEIGHT_IN_FORCED_CUTS = 4.0
CONFIDENT_SPEECH_PROBABILITY = 0.5


class PieceKind(enum.Enum):
    SPEECH = "speech"
    NON_SPEECH = "non_speech"


class CutReason(enum.Enum):
    PAUSE = "pause"
    FORCED = "forced"
    LEADING_NON_SPEECH = "leading_non_speech"
    NON_SPEECH_LIMIT = "non_speech_limit"
    END_OF_STREAM = "end_of_stream"


@dataclass(frozen=True)
class Piece:
    index: int
    start_sample: int
    end_sample: int
    audio: np.ndarray
    kind: PieceKind
    reason: CutReason
    overlap_samples: int
    successor_start_sample: int
    peak: float
    speech_end_sample: int | None = None

    @property
    def forced(self) -> bool:
        return self.reason is CutReason.FORCED

    @property
    def is_digital_silence(self) -> bool:
        return self.peak <= DIGITAL_SILENCE_PEAK

    @property
    def start_seconds(self) -> float:
        return samples_to_seconds(self.start_sample)

    @property
    def end_seconds(self) -> float:
        return samples_to_seconds(self.end_sample)

    @property
    def duration_seconds(self) -> float:
        return samples_to_seconds(self.end_sample - self.start_sample)


@dataclass(frozen=True)
class Snapshot:
    start_sample: int
    audio: np.ndarray

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.audio.size


@dataclass(frozen=True)
class SegmenterConfig:
    speech_onset_probability: float = 0.4
    speech_offset_probability: float = 0.25
    pause_rules: tuple[tuple[float, float], ...] = ((1.5, 0.8), (10.0, 0.5), (16.0, 0.3), (math.inf, 0.18))
    pause_cut_margin_seconds: float = 0.05
    pause_trailing_maximum_seconds: float = 0.6
    hard_maximum_seconds: float = 22.0
    forced_cut_search_seconds: float = 6.0
    forced_cut_end_margin_seconds: float = 0.3
    forced_overlap_seconds: float = 3.0
    leading_non_speech_maximum_seconds: float = 6.0
    preroll_seconds: float = 0.6
    preroll_search_seconds: float = 0.4
    non_speech_maximum_seconds: float = 20.0
    non_speech_keep_seconds: float = 1.0
    non_speech_search_seconds: float = 3.0
    quiet_window_samples: int = 128


class _SampleBuffer:
    def __init__(self) -> None:
        self._data = np.zeros(INITIAL_BUFFER_CAPACITY, dtype=np.float32)
        self._start = 0
        self._length = 0

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        return self._start + self._length

    def append(self, samples: np.ndarray) -> None:
        required = self._length + samples.size
        if required > self._data.size:
            grown = np.zeros(max(required, self._data.size * 2), dtype=np.float32)
            grown[: self._length] = self._data[: self._length]
            self._data = grown
        self._data[self._length : required] = samples
        self._length = required

    def view(self, start_sample: int, end_sample: int) -> np.ndarray:
        if start_sample < self._start or end_sample > self.end or start_sample > end_sample:
            raise ValueError(f"range {start_sample}-{end_sample} outside buffer {self._start}-{self.end}")
        return self._data[start_sample - self._start : end_sample - self._start]

    def discard_before(self, sample: int) -> None:
        drop = sample - self._start
        if drop <= 0:
            return
        remaining = self._length - drop
        self._data[:remaining] = self._data[drop : self._length]
        self._start = sample
        self._length = remaining


class SpeechSegmenter:
    def __init__(self, classifier: FrameClassifier, config: SegmenterConfig | None = None) -> None:
        self._classifier = classifier
        self._config = config or SegmenterConfig()
        self._lock = threading.Lock()
        self._buffer = _SampleBuffer()
        self._frames_done = 0
        self._labels: list[bool] = []
        self._probabilities: list[float] = []
        self._labels_origin = 0
        self._triggered = False
        self._piece_start = 0
        self._pending_overlap = 0
        self._first_speech_frame: int | None = None
        self._last_speech_frame: int | None = None
        self._next_index = 0
        self._emitted_until = 0
        self._finished = False
        self._pieces: list[Piece] = []

    @property
    def config(self) -> SegmenterConfig:
        return self._config

    @property
    def total_samples(self) -> int:
        return self._buffer.end

    @property
    def emitted_until(self) -> int:
        return self._emitted_until

    def feed(self, samples: np.ndarray) -> list[Piece]:
        with self._lock:
            if self._finished:
                raise RuntimeError("segmenter already finished")
            if samples.size:
                self._buffer.append(np.asarray(samples, dtype=np.float32))
            self._process_new_frames()
            return self._take_pieces()

    def finish(self) -> list[Piece]:
        with self._lock:
            if not self._finished:
                self._finished = True
                if self._buffer.end > self._piece_start:
                    if self._first_speech_frame is not None:
                        kind = PieceKind.SPEECH
                    else:
                        kind = PieceKind.NON_SPEECH
                    self._emit(self._buffer.end, kind, CutReason.END_OF_STREAM, self._buffer.end)
            return self._take_pieces()

    def snapshot(self) -> Snapshot | None:
        with self._lock:
            if self._first_speech_frame is None or self._buffer.end <= self._piece_start:
                return None
            else:
                audio = self._buffer.view(self._piece_start, self._buffer.end).copy()
                return Snapshot(start_sample=self._piece_start, audio=audio)

    def _take_pieces(self) -> list[Piece]:
        pieces = self._pieces
        self._pieces = []
        return pieces

    def _process_new_frames(self) -> None:
        available_frames = self._buffer.end // FRAME_SAMPLES
        new_frame_count = available_frames - self._frames_done
        if new_frame_count <= 0:
            return
        first_new = self._frames_done
        frames = self._buffer.view(first_new * FRAME_SAMPLES, available_frames * FRAME_SAMPLES)
        probabilities = self._classifier(frames.reshape(new_frame_count, FRAME_SAMPLES))
        for offset in range(new_frame_count):
            self._process_frame(first_new + offset, float(probabilities[offset]))
        self._frames_done = available_frames

    def _process_frame(self, frame_index: int, probability: float) -> None:
        config = self._config
        if self._triggered:
            if probability < config.speech_offset_probability:
                self._triggered = False
        elif probability >= config.speech_onset_probability:
            self._triggered = True
        speech = self._triggered
        self._labels.append(speech)
        self._probabilities.append(probability)
        frame_end = (frame_index + 1) * FRAME_SAMPLES

        if speech:
            self._on_speech_frame(frame_index)
        elif self._last_speech_frame is not None:
            self._on_pause_frame(frame_end)
        else:
            self._on_non_speech_frame(frame_end)

        if self._first_speech_frame is not None and frame_end - self._piece_start >= seconds_to_samples(
            config.hard_maximum_seconds
        ):
            self._force_cut(frame_end)

    def _on_speech_frame(self, frame_index: int) -> None:
        config = self._config
        if self._first_speech_frame is None:
            self._first_speech_frame = frame_index
            speech_start = frame_index * FRAME_SAMPLES
            leading = speech_start - self._piece_start
            if leading >= seconds_to_samples(config.leading_non_speech_maximum_seconds):
                upper = speech_start - seconds_to_samples(config.preroll_seconds)
                lower = upper - seconds_to_samples(config.preroll_search_seconds)
                cut = self._quiet_point(lower, upper, speech_start)
                self._emit(cut, PieceKind.NON_SPEECH, CutReason.LEADING_NON_SPEECH, cut)
        self._last_speech_frame = frame_index

    def _on_pause_frame(self, frame_end: int) -> None:
        config = self._config
        assert self._last_speech_frame is not None
        speech_end = (self._last_speech_frame + 1) * FRAME_SAMPLES
        pause = frame_end - speech_end
        content_seconds = samples_to_seconds(speech_end - self._piece_start)
        if pause >= seconds_to_samples(self._required_pause_seconds(content_seconds)):
            margin = seconds_to_samples(config.pause_cut_margin_seconds)
            lower = speech_end + margin
            upper = min(speech_end + seconds_to_samples(config.pause_trailing_maximum_seconds), frame_end - margin)
            cut = self._quiet_point(lower, upper, frame_end)
            self._emit(cut, PieceKind.SPEECH, CutReason.PAUSE, cut)

    def _on_non_speech_frame(self, frame_end: int) -> None:
        config = self._config
        if frame_end - self._piece_start >= seconds_to_samples(config.non_speech_maximum_seconds):
            upper = frame_end - seconds_to_samples(config.non_speech_keep_seconds)
            lower = upper - seconds_to_samples(config.non_speech_search_seconds)
            cut = self._quiet_point(lower, upper, frame_end)
            self._emit(cut, PieceKind.NON_SPEECH, CutReason.NON_SPEECH_LIMIT, cut)

    def _force_cut(self, frame_end: int) -> None:
        config = self._config
        upper = frame_end - seconds_to_samples(config.forced_cut_end_margin_seconds)
        lower = upper - seconds_to_samples(config.forced_cut_search_seconds)
        cut = self._quiet_point(lower, upper, frame_end, weigh_probability=True)
        overlap = seconds_to_samples(config.forced_overlap_seconds)
        next_start = max(self._piece_start + 1, cut - overlap)
        logger.info(
            "Forced cut at %.2fs after %.1fs without a usable pause",
            samples_to_seconds(cut), samples_to_seconds(cut - self._piece_start),
        )
        self._emit(cut, PieceKind.SPEECH, CutReason.FORCED, next_start)

    def _required_pause_seconds(self, content_seconds: float) -> float:
        for content_limit, required_pause in self._config.pause_rules:
            if content_seconds < content_limit:
                return required_pause
        return self._config.pause_rules[-1][1]

    def _quiet_point(self, lower: int, upper: int, frame_end: int, weigh_probability: bool = False) -> int:
        window = self._config.quiet_window_samples
        lower = max(lower, self._piece_start + 1, self._emitted_until + 1)
        upper = min(upper, frame_end)
        weights = None
        if weigh_probability:
            frame_indices = window_starts(lower, upper, window) // FRAME_SAMPLES
            probabilities = np.array([self._probability_of(int(index)) for index in frame_indices])
            weights = 1.0 + PROBABILITY_WEIGHT_IN_FORCED_CUTS * probabilities
        samples = self._buffer.view(self._buffer.start, self._buffer.end)
        quiet_point = find_quiet_point(samples, self._buffer.start, lower, upper, window, weights)
        if quiet_point is None:
            return min(lower, frame_end)
        else:
            return quiet_point

    def _probability_of(self, frame_index: int) -> float:
        position = frame_index - self._labels_origin
        if 0 <= position < len(self._probabilities):
            return self._probabilities[position]
        else:
            return 1.0

    def _emit(self, cut: int, kind: PieceKind, reason: CutReason, next_start: int) -> None:
        start = self._piece_start
        if not start < cut <= self._buffer.end:
            raise AssertionError(f"invalid cut {cut} for piece starting at {start}")
        if start + self._pending_overlap != self._emitted_until:
            raise AssertionError(
                f"coverage gap: piece starts at {start} (+{self._pending_overlap} overlap), "
                f"previous piece ended at {self._emitted_until}"
            )
        audio = self._buffer.view(start, cut).copy()
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        speech_end = self._speech_end_before(start, cut)
        self._pieces.append(Piece(
            index=self._next_index,
            start_sample=start,
            end_sample=cut,
            audio=audio,
            kind=kind,
            reason=reason,
            overlap_samples=self._pending_overlap,
            successor_start_sample=next_start,
            peak=peak,
            speech_end_sample=speech_end,
        ))
        self._next_index += 1
        self._emitted_until = cut
        self._pending_overlap = cut - next_start
        self._piece_start = next_start
        self._buffer.discard_before(next_start)
        self._trim_labels(next_start)
        self._recompute_speech_bounds()

    def _speech_end_before(self, start: int, cut: int) -> int | None:
        for offset in range(len(self._probabilities) - 1, -1, -1):
            frame_start = (self._labels_origin + offset) * FRAME_SAMPLES
            if frame_start >= cut or self._probabilities[offset] < CONFIDENT_SPEECH_PROBABILITY:
                continue
            frame_end = min(cut, frame_start + FRAME_SAMPLES)
            if frame_end > start:
                return frame_end
            else:
                return None
        return None

    def _trim_labels(self, start_sample: int) -> None:
        first_frame = start_sample // FRAME_SAMPLES
        drop = first_frame - self._labels_origin
        if drop > 0:
            del self._labels[:drop]
            del self._probabilities[:drop]
            self._labels_origin = first_frame

    def _recompute_speech_bounds(self) -> None:
        speech_frames = [self._labels_origin + offset for offset, label in enumerate(self._labels) if label]
        if speech_frames:
            self._first_speech_frame = speech_frames[0]
            self._last_speech_frame = speech_frames[-1]
        else:
            self._first_speech_frame = None
            self._last_speech_frame = None
