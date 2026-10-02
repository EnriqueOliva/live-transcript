import queue

import numpy as np
import pytest

from tests.helpers import (
    EnergyClassifier,
    FakeEngine,
    RecordingEvents,
    build,
    result,
    seconds,
    silence,
    tone,
    words_for,
)
from whisper_transcriber.io.transcript_writer import TranscriptWriter
from whisper_transcriber.session.events import LineStyle
from whisper_transcriber.session.messages import EndOfStream, Notice, PipelineStatistics
from whisper_transcriber.stt.handoff import TimedWord
from whisper_transcriber.stt.segmenter import CutReason, Piece, PieceKind, Snapshot, SpeechSegmenter
from whisper_transcriber.stt.worker import FAILURE_TEXT, DecodingOptions, TranscriptionWorker


def make_piece(index, start, end, audio=None, kind=PieceKind.SPEECH, reason=CutReason.PAUSE,
               overlap=0, successor=None):
    samples = audio if audio is not None else tone((end - start) / 16000)
    return Piece(
        index=index,
        start_sample=start,
        end_sample=end,
        audio=samples,
        kind=kind,
        reason=reason,
        overlap_samples=overlap,
        successor_start_sample=end if successor is None else successor,
        peak=float(np.max(np.abs(samples))) if samples.size else 0.0,
    )


def run_worker(items, engine, tmp_path, captured, language="es", decoding=None, snapshot_source=None, partials=False):
    piece_queue = queue.Queue()
    for item in items:
        piece_queue.put(item)
    piece_queue.put(EndOfStream(PipelineStatistics(captured_samples=captured)))
    writer = TranscriptWriter(tmp_path)
    writer.open()
    events = RecordingEvents()
    worker = TranscriptionWorker(
        piece_queue, engine, writer, events, language,
        captured_samples=lambda: captured,
        snapshot_source=snapshot_source,
        report_path=tmp_path / "session_report.txt",
        partials=partials,
        decoding=decoding,
    )
    worker.run()
    return worker, events


def plain_lines(tmp_path):
    return (tmp_path / "transcript.txt").read_text(encoding="utf-8").splitlines()


class TestEveryPieceIsTranscribed:
    def test_all_pieces_reach_the_transcript_in_order(self, tmp_path):
        pieces = [make_piece(index, seconds(index), seconds(index + 1)) for index in range(5)]
        engine = FakeEngine(responses=[result(f"linea {index}") for index in range(5)])
        worker, events = run_worker(pieces, engine, tmp_path, captured=seconds(5))
        assert plain_lines(tmp_path) == [f"linea {index}" for index in range(5)]
        assert [line.text for line in events.lines] == [f"linea {index}" for index in range(5)]
        assert worker.report.is_complete
        assert worker.report.coverage_percent == 100.0
        assert "100%" in events.finished_summaries[0]
        assert (tmp_path / "session_report.txt").read_text(encoding="utf-8").count("COMPLETE") == 1

    def test_every_call_requests_word_timestamps(self, tmp_path):
        engine = FakeEngine()
        run_worker([make_piece(0, 0, seconds(2))], engine, tmp_path, captured=seconds(2))
        assert all(call.word_timestamps for call in engine.calls)

    def test_digital_silence_is_accounted_without_calling_the_model(self, tmp_path):
        pieces = [make_piece(0, 0, seconds(20), audio=silence(20.0), kind=PieceKind.NON_SPEECH,
                             reason=CutReason.NON_SPEECH_LIMIT)]
        engine = FakeEngine()
        worker, _ = run_worker(pieces, engine, tmp_path, captured=seconds(20))
        assert engine.calls == []
        assert worker.report.silent_samples == seconds(20)
        assert worker.report.is_complete

    def test_non_speech_output_is_kept_and_marked(self, tmp_path):
        pieces = [make_piece(0, 0, seconds(5), kind=PieceKind.NON_SPEECH, reason=CutReason.NON_SPEECH_LIMIT)]
        engine = FakeEngine(responses=[result("Gracias.")])
        worker, events = run_worker(pieces, engine, tmp_path, captured=seconds(5))
        assert events.lines[0].style is LineStyle.UNCERTAIN
        assert plain_lines(tmp_path) == ["Gracias."]
        stamped = (tmp_path / "transcript_with_timestamps.txt").read_text(encoding="utf-8")
        assert "(?) Gracias." in stamped
        assert worker.report.uncertain_lines == 1

    def test_low_confidence_speech_is_kept_and_marked(self, tmp_path):
        engine = FakeEngine(responses=[result("algo dudoso", looks_like_non_speech=True)])
        _, events = run_worker([make_piece(0, 0, seconds(3))], engine, tmp_path, captured=seconds(3))
        assert events.lines[0].text == "algo dudoso"
        assert events.lines[0].style is LineStyle.UNCERTAIN


class TestFailures:
    def test_transient_failure_is_retried(self, tmp_path):
        engine = FakeEngine(responses=[RuntimeError("CUDA failure"), result("recuperado")])
        worker, _ = run_worker([make_piece(0, 0, seconds(2))], engine, tmp_path, captured=seconds(2))
        assert plain_lines(tmp_path) == ["recuperado"]
        assert engine.recoveries == 1
        assert worker.report.failed_pieces == 0

    def test_persistent_failure_is_written_and_the_session_continues(self, tmp_path):
        engine = FakeEngine(responses=[RuntimeError("a"), RuntimeError("b"), RuntimeError("c"), result("siguiente")])
        pieces = [make_piece(0, 0, seconds(2)), make_piece(1, seconds(2), seconds(4))]
        worker, events = run_worker(pieces, engine, tmp_path, captured=seconds(4))
        assert events.lines[0].style is LineStyle.FAILURE
        assert events.lines[0].text == FAILURE_TEXT
        assert events.lines[1].text == "siguiente"
        assert worker.report.failed_pieces == 1
        assert not worker.report.is_complete
        assert worker.report.gap_samples == 0

    def test_model_that_never_loads_is_reported(self, tmp_path):
        engine = FakeEngine(fail_load=True)
        worker, events = run_worker([make_piece(0, 0, seconds(2))], engine, tmp_path, captured=seconds(2))
        assert events.errors
        assert worker.report.model_failed
        assert worker.report.failed_pieces == 1
        assert events.finished_summaries
        assert "problems" in events.finished_summaries[0]


class TestCoverageAccounting:
    def test_a_gap_between_pieces_is_reported(self, tmp_path):
        pieces = [make_piece(0, 0, seconds(1)), make_piece(1, seconds(2), seconds(3))]
        worker, _ = run_worker(pieces, FakeEngine(), tmp_path, captured=seconds(3))
        assert worker.report.gap_samples == seconds(1)
        assert not worker.report.is_complete

    def test_audio_captured_but_never_segmented_is_reported(self, tmp_path):
        worker, _ = run_worker([make_piece(0, 0, seconds(1))], FakeEngine(), tmp_path, captured=seconds(2))
        assert worker.report.gap_samples == seconds(1)
        assert worker.report.covered_samples == seconds(1)
        assert not worker.report.is_complete

    def test_notices_go_to_the_timestamped_transcript_only(self, tmp_path):
        items = [Notice(sample=seconds(1), text="Audio capture reconnected"), make_piece(0, 0, seconds(2))]
        run_worker(items, FakeEngine(responses=[result("hola")]), tmp_path, captured=seconds(2))
        assert plain_lines(tmp_path) == ["hola"]
        stamped = (tmp_path / "transcript_with_timestamps.txt").read_text(encoding="utf-8")
        assert "[Audio capture reconnected]" in stamped


class TestForcedHandoff:
    def test_next_piece_resumes_exactly_at_the_word_boundary(self, tmp_path):
        cut = seconds(22.0)
        overlap_start = seconds(19.0)
        first_words = words_for("uno dos tres cuatro cinco seis siete ocho nueve diez once", 18.0, step=0.36)
        engine = FakeEngine(responses=[
            result("ignored", words=first_words),
            result("resto", words=[TimedWord(0.1, 0.5, " resto")]),
        ])
        first = make_piece(0, 0, cut, reason=CutReason.FORCED, successor=overlap_start)
        second = make_piece(1, overlap_start, seconds(26.0), overlap=cut - overlap_start)
        worker, events = run_worker([first, second], engine, tmp_path, captured=seconds(26.0))
        committed_text = events.lines[0].text
        boundary_seconds = events.lines[0].end
        committed = committed_text.split()
        assert committed == [word.text.strip() for word in first_words[: len(committed)]]
        trimmed_audio_length = engine.calls[1].audio.size
        assert trimmed_audio_length == seconds(26.0) - round(boundary_seconds * 16000)
        assert events.lines[1].start == pytest.approx(boundary_seconds)
        for word in first_words[len(committed) :]:
            assert word.start >= boundary_seconds - 1e-6
        assert worker.report.is_complete
        assert worker.report.forced_cuts == 1

    def test_failed_forced_piece_leaves_the_overlap_for_the_next_piece(self, tmp_path):
        cut = seconds(22.0)
        overlap_start = seconds(19.0)
        engine = FakeEngine(responses=[RuntimeError("x"), RuntimeError("y"), RuntimeError("z"), result("resto")])
        first = make_piece(0, 0, cut, reason=CutReason.FORCED, successor=overlap_start)
        second = make_piece(1, overlap_start, seconds(26.0), overlap=cut - overlap_start)
        worker, _ = run_worker([first, second], engine, tmp_path, captured=seconds(26.0))
        assert engine.calls[-1].audio.size == seconds(26.0) - overlap_start
        assert worker.report.gap_samples == 0


class TestContextAndPrompt:
    def test_previous_audio_is_given_as_context_and_its_words_are_dropped(self, tmp_path):
        first = make_piece(0, 0, seconds(4))
        second = make_piece(1, seconds(4), seconds(7))
        context_words = [TimedWord(3.0, 3.6, " viejo"), TimedWord(4.3, 4.8, " nuevo"), TimedWord(5.0, 5.5, " final")]
        engine = FakeEngine(responses=[result("primero", words=words_for("primero", 0.5)),
                                       result("ignored", words=context_words)])
        _, events = run_worker([first, second], engine, tmp_path, captured=seconds(7),
                               decoding=DecodingOptions(context_seconds=4.0))
        assert engine.calls[1].audio.size == seconds(7)
        assert events.lines[1].text == "nuevo final"

    def test_word_ending_right_after_the_boundary_is_kept(self, tmp_path):
        first = make_piece(0, 0, seconds(4))
        second = make_piece(1, seconds(4), seconds(7))
        context_words = [TimedWord(3.5, 4.06, " justo"), TimedWord(4.5, 5.0, " luego")]
        engine = FakeEngine(responses=[result("a", words=words_for("a", 0.5)),
                                       result("ignored", words=context_words)])
        _, events = run_worker([first, second], engine, tmp_path, captured=seconds(7),
                               decoding=DecodingOptions(context_seconds=2.0))
        assert events.lines[1].text == "justo luego"

    def test_previous_words_are_used_as_prompt(self, tmp_path):
        pieces = [make_piece(0, 0, seconds(2)), make_piece(1, seconds(2), seconds(4))]
        engine = FakeEngine(responses=[result("hola que tal", words=words_for("hola que tal", 0.2)),
                                       result("bien", words=words_for("bien", 2.2))])
        run_worker(pieces, engine, tmp_path, captured=seconds(4),
                   decoding=DecodingOptions(prompt_previous_text=True))
        assert engine.calls[0].prompt is None
        assert engine.calls[1].prompt == "hola que tal"


class TestLanguage:
    def test_fixed_language_is_always_used(self, tmp_path):
        engine = FakeEngine()
        run_worker([make_piece(0, 0, seconds(1))], engine, tmp_path, captured=seconds(1), language="es")
        assert engine.calls[0].language == "es"

    def test_auto_language_remembers_a_confident_detection_for_short_pieces(self, tmp_path):
        pieces = [make_piece(0, 0, seconds(5)), make_piece(1, seconds(5), seconds(6))]
        engine = FakeEngine(responses=[result("hola", language="es", language_probability=0.95), result("sí")])
        run_worker(pieces, engine, tmp_path, captured=seconds(6), language="Auto")
        assert engine.calls[0].language is None
        assert engine.calls[1].language == "es"


class TestPartials:
    def test_idle_worker_shows_the_unfinished_sentence(self, tmp_path):
        snapshot = Snapshot(start_sample=0, audio=tone(2.0))
        piece_queue = queue.Queue()
        writer = TranscriptWriter(tmp_path)
        writer.open()
        events = RecordingEvents()
        engine = FakeEngine(default=result("parcial"))
        engine.load()
        worker = TranscriptionWorker(piece_queue, engine, writer, events, "es", captured_samples=lambda: 0,
                                     snapshot_source=lambda: snapshot, partials=True)
        worker._on_idle()
        assert events.partials == ["parcial"]
        assert engine.calls[0].fast
        worker._on_idle()
        assert len(engine.calls) == 1

    def test_partials_are_off_on_cpu_by_default(self, tmp_path):
        engine = FakeEngine(device="cpu")
        engine.load()
        worker = TranscriptionWorker(queue.Queue(), engine, TranscriptWriter(tmp_path), RecordingEvents(), "es",
                                     captured_samples=lambda: 0, snapshot_source=lambda: Snapshot(0, tone(2.0)))
        worker._on_idle()
        assert engine.calls == []


class TestWithRealSegmenter:
    def test_segmenter_and_worker_cover_a_long_stream_completely(self, tmp_path):
        audio = build(tone(5.0), silence(1.0), tone(30.0), silence(2.0), tone(2.0), silence(25.0), tone(4.0))
        segmenter = SpeechSegmenter(EnergyClassifier())
        pieces = []
        for start in range(0, audio.size, 3000):
            pieces.extend(segmenter.feed(audio[start : start + 3000]))
        pieces.extend(segmenter.finish())

        def words_covering(audio_block):
            duration = audio_block.size / 16000
            return result("x", words=words_for(" ".join(["p"] * int(duration / 0.3)), 0.0))

        engine = FakeEngine(default=None)
        engine.responses = [words_covering] * len(pieces)
        worker, _ = run_worker(pieces, engine, tmp_path, captured=audio.size)
        assert worker.report.gap_samples == 0
        assert worker.report.covered_samples == audio.size
        assert worker.report.is_complete
