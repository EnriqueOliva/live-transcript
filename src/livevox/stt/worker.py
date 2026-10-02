from __future__ import annotations

import logging
import queue
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from livevox.audio.timeline import samples_to_seconds, seconds_to_samples
from livevox.session.events import LineStyle, SessionEvents, TranscriptLine
from livevox.session.messages import EndOfStream, Notice
from livevox.session.report import SessionReport
from livevox.stt.handoff import TimedWord, choose_handoff
from livevox.stt.segmenter import DIGITAL_SILENCE_PEAK, Piece, PieceKind, Snapshot
from livevox.stt.whisper_engine import TranscriptionResult, WhisperEngine

if TYPE_CHECKING:
    from livevox.io.transcript_writer import TranscriptWriter

logger = logging.getLogger(__name__)

AUTO_LANGUAGE = "Auto"
MAXIMUM_ATTEMPTS = 3
IDLE_POLL_SECONDS = 0.2
PROGRESS_INTERVAL_SECONDS = 0.5
SHORT_PIECE_SECONDS = 2.0
CONFIDENT_LANGUAGE_PROBABILITY = 0.7
PARTIAL_MINIMUM_SECONDS = 1.0
PARTIAL_REFRESH_SECONDS = 0.6
FAILURE_TEXT = "[audio not transcribed, it is kept in the session recording]"
CONTEXT_KEEP_MARGIN_SECONDS = 0.05
EMPTY_AUDIO = np.zeros(0, dtype=np.float32)


@dataclass(frozen=True)
class DecodingOptions:
    context_seconds: float = 0.0
    prompt_previous_text: bool = True
    previous_text_words: int = 40

SnapshotSource = Callable[[], Snapshot | None]
CapturedSamplesSource = Callable[[], int]


class TranscriptionWorker:
    def __init__(
        self,
        piece_queue: queue.Queue,
        engine: WhisperEngine,
        writer: TranscriptWriter,
        events: SessionEvents,
        language: str,
        captured_samples: CapturedSamplesSource,
        snapshot_source: SnapshotSource | None = None,
        report_path: Path | None = None,
        partials: bool | None = None,
        decoding: DecodingOptions | None = None,
    ) -> None:
        self._piece_queue = piece_queue
        self._engine = engine
        self._writer = writer
        self._events = events
        self._language = language
        self._captured_samples = captured_samples
        self._snapshot_source = snapshot_source
        self._report_path = report_path
        self._partials_requested = partials
        self._decoding = decoding or DecodingOptions()
        self._history: deque[tuple[int, np.ndarray]] = deque()
        self._history_samples = 0
        self._recent_words: deque[str] = deque(maxlen=self._decoding.previous_text_words)
        self._report = SessionReport(model=engine.model_name, language=language)
        self._covered_until = 0
        self._handoff_boundary: int | None = None
        self._session_language: str | None = None
        self._partial_shown = False
        self._last_partial_end = 0
        self._last_progress = 0.0

    @property
    def report(self) -> SessionReport:
        return self._report

    def run(self) -> None:
        logger.info("Transcription worker started")
        self._load_model()
        while True:
            try:
                item = self._piece_queue.get(timeout=IDLE_POLL_SECONDS)
            except queue.Empty:
                self._on_idle()
                continue
            if isinstance(item, EndOfStream):
                self._finish(item)
                break
            elif isinstance(item, Notice):
                self._write_notice(item)
            else:
                self._process_piece(item)
            self._emit_progress(force=False)
        logger.info("Transcription worker stopped")

    def _load_model(self) -> None:
        self._events.status("Loading model", "")
        self._events.message("Loading the transcription model, audio is already being captured")
        try:
            self._engine.load()
        except Exception:
            logger.exception("Failed to load model")
            self._report.model_failed = True
            self._events.error("Failed to load the transcription model. The audio keeps being recorded.")
            return
        self._report.device = f"{self._engine.device} ({self._engine.compute_type})"
        self._events.status("Recording", self._report.device)
        self._events.message(f"Model ready on {self._report.device}")

    def _partials_enabled(self) -> bool:
        if self._partials_requested is None:
            return self._engine.device == "cuda"
        else:
            return self._partials_requested

    def _process_piece(self, piece: Piece) -> None:
        self._report.pieces += 1
        if piece.forced:
            self._report.forced_cuts += 1
        effective_start = piece.start_sample
        audio = piece.audio
        if piece.overlap_samples > 0 and self._handoff_boundary is not None:
            trim = min(max(self._handoff_boundary - piece.start_sample, 0), piece.overlap_samples, audio.size)
            audio = audio[trim:]
            effective_start += trim
        self._handoff_boundary = None

        if effective_start > self._covered_until:
            gap = effective_start - self._covered_until
            logger.error("Coverage gap of %d samples before piece %d", gap, piece.index)
            self._report.gap_samples += gap

        if audio.size == 0:
            self._advance_coverage(effective_start, piece.end_sample)
        elif not self._engine.is_loaded:
            self._record_failure(piece, effective_start, notify=False)
        elif _peak(audio) <= DIGITAL_SILENCE_PEAK:
            self._report.silent_samples += self._advance_coverage(effective_start, piece.end_sample)
        else:
            self._transcribe_piece(piece, audio, effective_start)

    def _transcribe_piece(self, piece: Piece, audio: np.ndarray, effective_start: int) -> None:
        duration = samples_to_seconds(audio.size)
        language = self._language_for(duration)
        is_speech = piece.kind is PieceKind.SPEECH
        context = self._context_before(effective_start) if is_speech else EMPTY_AUDIO
        prompt = self._previous_text() if is_speech and self._decoding.prompt_previous_text else None
        decode_start = effective_start - context.size
        result = self._transcribe_with_retries(
            np.concatenate([context, audio]), language, prompt, _speech_end_seconds(piece, decode_start),
        )
        if result is not None and context.size and result.text and not result.words:
            result = self._transcribe_with_retries(audio, language, prompt, _speech_end_seconds(piece, effective_start))
            decode_start = effective_start
        if result is None:
            self._remember(effective_start, audio)
            self._record_failure(piece, effective_start, notify=True)
            return

        self._learn_language(result, language, duration)
        offset_seconds = samples_to_seconds(decode_start)
        words = [
            TimedWord(start=offset_seconds + word.start, end=offset_seconds + word.end, text=word.text)
            for word in result.words
        ]
        if decode_start < effective_start:
            boundary_seconds = samples_to_seconds(effective_start) + CONTEXT_KEEP_MARGIN_SECONDS
            words = [word for word in words if word.end > boundary_seconds]
        text = _join_words(words) if result.words else result.text
        region_end = piece.end_sample
        if piece.forced and words:
            handoff = choose_handoff(words, piece.end_sample, piece.successor_start_sample, audio, effective_start)
            text = handoff.text
            region_end = max(handoff.boundary_sample, effective_start)
            self._handoff_boundary = handoff.boundary_sample
            logger.info(
                "Forced piece %d commits %d of %d words, next piece resumes at %.2fs",
                piece.index, len(handoff.committed), len(words), samples_to_seconds(handoff.boundary_sample),
            )

        self._remember(effective_start, audio[: region_end - effective_start])
        self._report.transcribed_samples += self._advance_coverage(effective_start, region_end)
        if text:
            uncertain = piece.kind is PieceKind.NON_SPEECH or result.looks_like_non_speech
            style = LineStyle.UNCERTAIN if uncertain else LineStyle.NORMAL
            if uncertain:
                self._report.uncertain_lines += 1
            else:
                self._recent_words.extend(text.split())
            self._emit_line(TranscriptLine(
                text=text,
                start=samples_to_seconds(effective_start),
                end=samples_to_seconds(region_end),
                style=style,
            ))
        else:
            self._clear_partial()
        if result.uncertain_text:
            self._report.uncertain_lines += 1
            self._emit_line(TranscriptLine(
                text=result.uncertain_text,
                start=samples_to_seconds(region_end),
                end=samples_to_seconds(region_end),
                style=LineStyle.UNCERTAIN,
            ))

    def _context_before(self, start_sample: int) -> np.ndarray:
        wanted = seconds_to_samples(self._decoding.context_seconds)
        if wanted <= 0:
            return EMPTY_AUDIO
        parts: list[np.ndarray] = []
        collected = 0
        cursor = start_sample
        for chunk_start, chunk in reversed(self._history):
            chunk_end = chunk_start + chunk.size
            if chunk_end != cursor or collected >= wanted:
                break
            take = min(chunk.size, wanted - collected)
            parts.append(chunk[chunk.size - take :])
            collected += take
            cursor = chunk_end - take
        if not parts:
            return EMPTY_AUDIO
        else:
            return np.concatenate(parts[::-1])

    def _remember(self, start_sample: int, audio: np.ndarray) -> None:
        limit = seconds_to_samples(self._decoding.context_seconds)
        if limit <= 0 or audio.size == 0:
            return
        self._history.append((start_sample, audio))
        self._history_samples += audio.size
        while self._history and self._history_samples - self._history[0][1].size >= limit:
            self._history_samples -= self._history.popleft()[1].size

    def _previous_text(self) -> str | None:
        if not self._recent_words:
            return None
        else:
            return " ".join(self._recent_words)

    def _transcribe_with_retries(
        self, audio: np.ndarray, language: str | None, prompt: str | None, speech_end_seconds: float | None,
    ) -> TranscriptionResult | None:
        for attempt in range(1, MAXIMUM_ATTEMPTS + 1):
            try:
                return self._engine.transcribe(
                    audio, language, word_timestamps=True, prompt=prompt, speech_end_seconds=speech_end_seconds,
                )
            except Exception:
                logger.exception("Transcription attempt %d of %d failed", attempt, MAXIMUM_ATTEMPTS)
                try:
                    self._engine.recover_after_failure()
                except Exception:
                    logger.exception("Could not recover the transcription engine")
        return None

    def _record_failure(self, piece: Piece, effective_start: int, notify: bool) -> None:
        self._report.failed_pieces += 1
        self._report.failed_samples += self._advance_coverage(effective_start, piece.end_sample)
        if notify:
            self._emit_line(TranscriptLine(
                text=FAILURE_TEXT,
                start=samples_to_seconds(effective_start),
                end=samples_to_seconds(piece.end_sample),
                style=LineStyle.FAILURE,
            ))

    def _advance_coverage(self, start: int, end: int) -> int:
        newly_covered = max(0, end - max(start, self._covered_until))
        self._covered_until = max(self._covered_until, end)
        return newly_covered

    def _language_for(self, duration: float) -> str | None:
        if self._language != AUTO_LANGUAGE:
            return self._language
        elif duration < SHORT_PIECE_SECONDS and self._session_language is not None:
            return self._session_language
        else:
            return None

    def _learn_language(self, result: TranscriptionResult, requested: str | None, duration: float) -> None:
        if (
            requested is None
            and duration >= SHORT_PIECE_SECONDS
            and result.language_probability >= CONFIDENT_LANGUAGE_PROBABILITY
        ):
            self._session_language = result.language

    def _emit_line(self, line: TranscriptLine) -> None:
        self._writer.write_line(line)
        self._partial_shown = False
        self._events.transcript_line(line)

    def _clear_partial(self) -> None:
        if self._partial_shown:
            self._partial_shown = False
            self._events.partial_text("")

    def _write_notice(self, notice: Notice) -> None:
        seconds = samples_to_seconds(notice.sample)
        self._writer.write_line(TranscriptLine(text=notice.text, start=seconds, end=seconds, style=LineStyle.NOTICE))
        self._events.message(notice.text)

    def _on_idle(self) -> None:
        self._emit_progress(force=False)
        if self._engine.is_loaded and self._snapshot_source is not None and self._partials_enabled():
            self._refresh_partial()

    def _refresh_partial(self) -> None:
        assert self._snapshot_source is not None
        snapshot = self._snapshot_source()
        if snapshot is None:
            self._clear_partial()
            return
        if snapshot.audio.size < seconds_to_samples(PARTIAL_MINIMUM_SECONDS):
            return
        if snapshot.end_sample - self._last_partial_end < seconds_to_samples(PARTIAL_REFRESH_SECONDS):
            return
        if _peak(snapshot.audio) <= DIGITAL_SILENCE_PEAK:
            return
        try:
            result = self._engine.transcribe(
                snapshot.audio, self._language_for(samples_to_seconds(snapshot.audio.size)), fast=True,
            )
        except Exception:
            logger.debug("Partial transcription failed", exc_info=True)
            return
        self._last_partial_end = snapshot.end_sample
        if result.text:
            self._partial_shown = True
            self._events.partial_text(result.text)

    def _emit_progress(self, force: bool) -> None:
        now = time.monotonic()
        if force or now - self._last_progress >= PROGRESS_INTERVAL_SECONDS:
            self._last_progress = now
            self._events.progress(
                samples_to_seconds(self._captured_samples()), samples_to_seconds(self._covered_until),
            )

    def _finish(self, end: EndOfStream) -> None:
        statistics = end.statistics
        report = self._report
        report.captured_samples = statistics.captured_samples
        report.covered_samples = self._covered_until - report.gap_samples
        if self._covered_until < statistics.captured_samples:
            missing = statistics.captured_samples - self._covered_until
            logger.error("%d captured samples were never handed to the worker", missing)
            report.gap_samples += missing
        report.overflow_events = statistics.overflow_events
        report.device_switches = statistics.device_switches
        report.recorded_samples = statistics.recorded_samples
        report.recording_failed = statistics.recording_failed
        report.recording_path = str(statistics.recording_path or "")
        report.devices = list(statistics.devices)
        self._clear_partial()
        self._emit_progress(force=True)
        self._writer.close()
        if self._report_path is not None:
            try:
                self._report_path.write_text(report.to_text(), encoding="utf-8")
            except OSError:
                logger.exception("Could not write the session report")
        logger.info("Session report:\n%s", report.to_text())
        self._events.finished(report.summary())


def _speech_end_seconds(piece: Piece, decode_start: int) -> float | None:
    if piece.speech_end_sample is None:
        return None
    else:
        return samples_to_seconds(piece.speech_end_sample - decode_start)


def _join_words(words: list[TimedWord]) -> str:
    return "".join(word.text for word in words).strip()


def _peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0
