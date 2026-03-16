import pytest

from tests.conftest import make_segment
from whisper_transcriber.stt.chunk_merger import ChunkMerger, _text_merge


class TestSafeZone:
    def test_defers_segments_in_overlap_region(self):
        merger = ChunkMerger()
        segments = [
            make_segment(0.0, 10.0, "safe zone text"),
            make_segment(20.0, 24.0, "still safe"),
            make_segment(26.0, 28.0, "in overlap zone"),
        ]
        result = merger.merge(segments, chunk_offset=0.0, safe_duration=25.0)
        assert len(result) == 2
        assert result[0].text == "safe zone text"
        assert result[1].text == "still safe"

    def test_flush_accepts_all_segments(self):
        merger = ChunkMerger()
        segments = [
            make_segment(0.0, 5.0, "start"),
            make_segment(26.0, 28.0, "would be deferred normally"),
        ]
        result = merger.merge(segments, chunk_offset=50.0, safe_duration=None)
        assert len(result) == 2


class TestBoundaryHoldBack:
    def test_holds_back_segment_near_chunk_boundary(self):
        merger = ChunkMerger()
        segments = [
            make_segment(0.0, 20.0, "Normal segment."),
            make_segment(23.0, 30.0, "Truncated at boundary..."),
        ]
        result = merger.merge(segments, chunk_offset=0.0, safe_duration=25.0, chunk_duration=30.0)
        assert len(result) == 1
        assert result[0].text == "Normal segment."
        assert merger._pending is not None
        assert "Truncated" in merger._pending.text

    def test_resolves_pending_with_next_chunk(self):
        merger = ChunkMerger()

        chunk0 = [
            make_segment(0.0, 20.0, "First part of the lecture."),
            make_segment(23.0, 30.0, "The quick brown fox jumped"),
        ]
        r0 = merger.merge(chunk0, chunk_offset=0.0, safe_duration=25.0, chunk_duration=30.0)
        assert len(r0) == 1
        assert merger._pending is not None

        chunk1 = [
            make_segment(0.0, 7.0, "brown fox jumped over the lazy dog"),
            make_segment(10.0, 18.0, "Next part continues here."),
        ]
        r1 = merger.merge(chunk1, chunk_offset=25.0, safe_duration=25.0, chunk_duration=30.0)
        assert len(r1) >= 2
        assert "The quick brown fox jumped over the lazy dog" in r1[0].text
        assert r1[0].start == pytest.approx(23.0)
        assert r1[0].end == pytest.approx(32.0)

    def test_commits_pending_as_is_when_no_continuation(self):
        merger = ChunkMerger()

        chunk0 = [make_segment(20.0, 30.0, "Ends at boundary")]
        merger.merge(chunk0, chunk_offset=0.0, safe_duration=25.0, chunk_duration=30.0)
        assert merger._pending is not None

        chunk1 = [make_segment(10.0, 20.0, "Completely different region")]
        r1 = merger.merge(chunk1, chunk_offset=25.0, safe_duration=25.0, chunk_duration=30.0)
        assert any("Ends at boundary" in s.text for s in r1)

    def test_flush_commits_pending(self):
        merger = ChunkMerger()

        chunk0 = [make_segment(24.0, 30.0, "Held back text")]
        merger.merge(chunk0, chunk_offset=0.0, safe_duration=25.0, chunk_duration=30.0)
        assert merger._pending is not None

        flush = [make_segment(8.0, 15.0, "Final words")]
        r = merger.merge(flush, chunk_offset=25.0, safe_duration=None, chunk_duration=30.0)
        texts = [s.text for s in r]
        assert "Held back text" in texts
        assert "Final words" in texts


class TestTextMerge:
    def test_merges_overlapping_text(self):
        result = _text_merge(
            "The quick brown fox jumped",
            "brown fox jumped over the lazy dog",
        )
        assert result == "The quick brown fox jumped over the lazy dog"

    def test_no_overlap_concatenates(self):
        result = _text_merge("Hello world", "Goodbye moon")
        assert result == "Hello world Goodbye moon"

    def test_single_word_overlap(self):
        result = _text_merge("She said hello", "hello to everyone")
        assert result == "She said hello to everyone"

    def test_full_overlap(self):
        result = _text_merge("same text here", "same text here and more")
        assert result == "same text here and more"


class TestBoundaryBugRegression:
    def test_no_word_loss_at_chunk_boundary(self):
        merger = ChunkMerger()

        chunk0 = [
            make_segment(0.0, 20.0, "que legalmente fue una guerra autorizada."),
            make_segment(23.0, 30.0, "Esa misma ley también es la"),
        ]
        r0 = merger.merge(chunk0, chunk_offset=0.0, safe_duration=25.0, chunk_duration=30.0)
        assert len(r0) == 1
        assert merger._pending is not None
        assert "Esa misma ley" in merger._pending.text

        chunk1 = [
            make_segment(0.0, 10.0, "Esa misma ley también es la que se usó para autorizar el caso de Irak."),
            make_segment(10.0, 20.0, "No hay ninguna justificación."),
        ]
        r1 = merger.merge(chunk1, chunk_offset=25.0, safe_duration=25.0, chunk_duration=30.0)
        assert len(r1) >= 2
        assert "que se usó para autorizar" in r1[0].text
        assert "Esa misma ley" in r1[0].text


class TestDedup:
    def test_filters_segments_before_last_end_time(self):
        merger = ChunkMerger()
        merger.last_end_time = 6.0
        segments = [
            make_segment(0.0, 3.0, "old text"),
            make_segment(6.5, 9.0, "new text"),
        ]
        result = merger.merge(segments, chunk_offset=0.0, safe_duration=25.0)
        assert len(result) == 1
        assert result[0].text == "new text"

    def test_keeps_normal_confidence(self):
        merger = ChunkMerger()
        segments = [make_segment(0.0, 3.0, "Normal speech", no_speech_prob=0.1, avg_logprob=-0.3)]
        result = merger.merge(segments, chunk_offset=0.0, safe_duration=25.0)
        assert len(result) == 1

    def test_filters_low_confidence(self):
        merger = ChunkMerger()
        segments = [make_segment(0.0, 3.0, "Thank you for watching", no_speech_prob=0.8, avg_logprob=-1.2)]
        result = merger.merge(segments, chunk_offset=0.0, safe_duration=25.0)
        assert len(result) == 0
