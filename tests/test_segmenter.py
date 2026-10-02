from itertools import pairwise

import numpy as np
import pytest

from livevox.stt.segmenter import (
    DIGITAL_SILENCE_PEAK,
    CutReason,
    PieceKind,
    SegmenterConfig,
    SpeechSegmenter,
)
from tests.helpers import EnergyClassifier, build, feed_in_blocks, noise, seconds, silence, tone

BLOCK_PATTERNS = [[seconds(1.0)], [441, 1024, 17, 4800, 512, 3], [512], [seconds(7.3)], [100_000]]


def segment(audio, block_sizes=None, config=None):
    segmenter = SpeechSegmenter(EnergyClassifier(), config)
    return feed_in_blocks(segmenter, audio, block_sizes or [seconds(1.0)])


def assert_lossless(pieces, audio):
    assert pieces, "no pieces emitted"
    assert pieces[0].start_sample == 0
    assert pieces[0].overlap_samples == 0
    for previous, current in pairwise(pieces):
        assert current.start_sample == previous.successor_start_sample
        assert current.start_sample + current.overlap_samples == previous.end_sample
        assert current.end_sample > previous.end_sample
    assert pieces[-1].end_sample == audio.size
    for piece in pieces:
        np.testing.assert_array_equal(piece.audio, audio[piece.start_sample : piece.end_sample])
    rebuilt = np.concatenate([pieces[0].audio] + [piece.audio[piece.overlap_samples :] for piece in pieces[1:]])
    np.testing.assert_array_equal(rebuilt, audio)


def random_scene(seed):
    generator = np.random.default_rng(seed)
    parts = []
    for _ in range(generator.integers(5, 40)):
        kind = generator.integers(0, 4)
        duration = float(generator.uniform(0.01, 9.0))
        if kind == 0:
            parts.append(silence(duration))
        elif kind == 1:
            parts.append(tone(duration, amplitude=float(generator.uniform(0.02, 0.9))))
        elif kind == 2:
            parts.append(noise(duration, amplitude=float(generator.uniform(0.0, 0.02)), seed=int(seed)))
        else:
            parts.append(tone(float(generator.uniform(15.0, 40.0)), amplitude=0.4))
    return build(*parts)


class TestLosslessCoverage:
    @pytest.mark.parametrize("seed", range(25))
    def test_random_scenes_are_covered_exactly(self, seed):
        audio = random_scene(seed)
        pieces = segment(audio, BLOCK_PATTERNS[seed % len(BLOCK_PATTERNS)])
        assert_lossless(pieces, audio)

    @pytest.mark.parametrize("seed", range(6))
    def test_feed_block_sizes_do_not_change_the_pieces(self, seed):
        audio = random_scene(100 + seed)
        reference = [(piece.start_sample, piece.end_sample, piece.reason) for piece in segment(audio)]
        for pattern in BLOCK_PATTERNS:
            pieces = segment(audio, pattern)
            assert [(piece.start_sample, piece.end_sample, piece.reason) for piece in pieces] == reference

    @pytest.mark.parametrize("seed", range(10))
    def test_no_piece_exceeds_the_model_window(self, seed):
        audio = random_scene(200 + seed)
        config = SegmenterConfig()
        limit = seconds(config.hard_maximum_seconds) + 512 * 2
        for piece in segment(audio):
            assert piece.end_sample - piece.start_sample <= limit
            assert piece.duration_seconds < 30.0

    def test_tiny_stream_is_emitted_on_finish(self):
        audio = tone(0.005)
        pieces = segment(audio)
        assert_lossless(pieces, audio)
        assert pieces[0].reason is CutReason.END_OF_STREAM

    def test_empty_stream_emits_nothing(self):
        segmenter = SpeechSegmenter(EnergyClassifier())
        assert segmenter.finish() == []

    def test_finish_is_idempotent_and_feed_after_finish_fails(self):
        segmenter = SpeechSegmenter(EnergyClassifier())
        segmenter.feed(tone(1.0))
        assert len(segmenter.finish()) == 1
        assert segmenter.finish() == []
        with pytest.raises(RuntimeError):
            segmenter.feed(tone(0.1))


class TestPauseCuts:
    def test_cut_lands_inside_the_pause(self):
        audio = build(tone(4.0), silence(1.0), tone(3.0), silence(1.0))
        pieces = segment(audio)
        pause_pieces = [piece for piece in pieces if piece.reason is CutReason.PAUSE]
        assert pause_pieces
        for piece in pause_pieces:
            assert np.all(audio[piece.end_sample - 64 : piece.end_sample + 64] == 0.0)
        assert_lossless(pieces, audio)

    def test_sentence_is_emitted_shortly_after_it_ends(self):
        segmenter = SpeechSegmenter(EnergyClassifier())
        assert segmenter.feed(tone(4.0)) == []
        emitted = segmenter.feed(silence(0.6))
        assert len(emitted) == 1
        assert emitted[0].reason is CutReason.PAUSE
        assert emitted[0].kind is PieceKind.SPEECH
        assert seconds(4.0) < emitted[0].end_sample < seconds(4.6)

    def test_short_utterance_waits_for_a_longer_pause(self):
        segmenter = SpeechSegmenter(EnergyClassifier())
        segmenter.feed(tone(0.8))
        assert segmenter.feed(silence(0.6)) == []
        assert len(segmenter.feed(silence(0.4))) == 1

    def test_long_sentence_accepts_a_short_breath(self):
        segmenter = SpeechSegmenter(EnergyClassifier())
        segmenter.feed(tone(17.0))
        emitted = segmenter.feed(silence(0.25))
        assert len(emitted) == 1
        assert emitted[0].reason is CutReason.PAUSE

    def test_snapshot_holds_the_unfinished_sentence(self):
        segmenter = SpeechSegmenter(EnergyClassifier())
        segmenter.feed(silence(0.3))
        assert segmenter.snapshot() is None
        segmenter.feed(tone(2.0))
        snapshot = segmenter.snapshot()
        assert snapshot is not None
        assert snapshot.start_sample == 0
        assert snapshot.audio.size == seconds(2.3)


class TestForcedCuts:
    def test_continuous_speech_is_cut_with_an_overlap(self):
        audio = tone(50.0)
        pieces = segment(audio)
        forced = [piece for piece in pieces if piece.forced]
        assert len(forced) >= 2
        config = SegmenterConfig()
        overlap = seconds(config.forced_overlap_seconds)
        for piece in forced:
            assert piece.successor_start_sample == piece.end_sample - overlap
        for previous, current in pairwise(pieces):
            if previous.forced:
                assert current.overlap_samples == overlap
        assert_lossless(pieces, audio)

    def test_forced_cut_prefers_the_quietest_moment(self):
        speech_with_dip = build(tone(19.0), tone(0.05, amplitude=0.012), tone(10.0))
        pieces = segment(speech_with_dip)
        first = pieces[0]
        assert first.forced
        assert abs(first.end_sample - seconds(19.025)) < seconds(0.05)

    def test_cut_after_a_forced_piece_never_moves_backwards(self):
        audio = build(tone(23.0), silence(2.0), tone(3.0), silence(2.0))
        pieces = segment(audio)
        assert_lossless(pieces, audio)


class TestNonSpeech:
    def test_long_non_speech_is_emitted_in_bounded_pieces(self):
        audio = build(noise(65.0, amplitude=0.004))
        pieces = segment(audio)
        assert_lossless(pieces, audio)
        assert all(piece.kind is PieceKind.NON_SPEECH for piece in pieces)
        assert all(piece.duration_seconds <= SegmenterConfig().non_speech_maximum_seconds for piece in pieces)
        assert any(piece.reason is CutReason.NON_SPEECH_LIMIT for piece in pieces)

    def test_long_leading_silence_is_split_from_speech(self):
        audio = build(noise(9.0, amplitude=0.003), tone(3.0), silence(1.0))
        pieces = segment(audio)
        assert pieces[0].reason is CutReason.LEADING_NON_SPEECH
        assert pieces[0].kind is PieceKind.NON_SPEECH
        assert seconds(8.0) <= pieces[0].end_sample <= seconds(8.5)
        assert pieces[1].kind is PieceKind.SPEECH
        assert_lossless(pieces, audio)

    def test_short_leading_silence_stays_with_the_speech(self):
        audio = build(silence(3.0), tone(3.0), silence(1.0))
        pieces = segment(audio)
        assert pieces[0].kind is PieceKind.SPEECH
        assert pieces[0].start_sample == 0

    def test_digital_silence_is_flagged(self):
        audio = build(silence(25.0))
        pieces = segment(audio)
        assert all(piece.is_digital_silence for piece in pieces)

    def test_faint_noise_is_not_treated_as_silence(self):
        audio = build(noise(25.0, amplitude=DIGITAL_SILENCE_PEAK * 4))
        pieces = segment(audio)
        assert not any(piece.is_digital_silence for piece in pieces)

    def test_quiet_speech_below_the_vad_still_reaches_a_piece(self):
        whisper = tone(2.0, amplitude=0.005)
        audio = build(silence(1.0), whisper, silence(1.0))
        pieces = segment(audio)
        assert_lossless(pieces, audio)
        assert not pieces[0].is_digital_silence


class ScriptedClassifier:
    def __init__(self, speech_frames):
        self._speech_frames = speech_frames
        self._next_frame = 0

    def __call__(self, frames):
        count = frames.shape[0]
        indices = np.arange(self._next_frame, self._next_frame + count)
        self._next_frame += count
        return np.where(np.isin(indices, self._speech_frames), 0.9, 0.05).astype(np.float32)


class TestCutsNeverEnterSpeech:
    def test_quiet_speech_followed_by_a_noisy_pause_is_not_cut_short(self):
        speech_seconds = 4.0
        audio = build(tone(speech_seconds, amplitude=0.002), noise(1.5, amplitude=0.2, seed=4))
        speech_frames = np.arange(0, int(speech_seconds * 16000) // 512)
        segmenter = SpeechSegmenter(ScriptedClassifier(speech_frames))
        pieces = feed_in_blocks(segmenter, audio, [seconds(0.5)])
        first = pieces[0]
        assert first.reason is CutReason.PAUSE
        speech_end = len(speech_frames) * 512
        assert first.end_sample > speech_end
        assert first.speech_end_sample == speech_end
        assert_lossless(pieces, audio)
