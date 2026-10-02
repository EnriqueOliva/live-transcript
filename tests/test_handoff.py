import numpy as np
import pytest

from livevox.audio.timeline import SAMPLE_RATE
from livevox.stt.handoff import LEAN_BEFORE_SECONDS, TimedWord, choose_handoff
from tests.helpers import seconds, tone

CUT = seconds(20.0)
NEXT_START = seconds(17.0)


def speech_audio(duration=20.0):
    return tone(duration, amplitude=0.3)


def dense_words(start, end, step=0.3, gap=0.0):
    words = []
    position = start
    index = 0
    while position + step <= end:
        words.append(TimedWord(start=position, end=position + step - gap, text=f" w{index}"))
        position += step
        index += 1
    return words


def assert_no_word_lost(words, handoff, next_start=NEXT_START, cut=CUT):
    assert next_start <= handoff.boundary_sample <= cut
    committed = handoff.committed
    assert committed == words[: len(committed)]
    boundary_seconds = handoff.boundary_sample / SAMPLE_RATE
    for word in words[len(committed) :]:
        assert word.start >= boundary_seconds - 1e-9


class TestBoundaryChoice:
    def test_widest_gap_in_the_overlap_wins(self):
        words = [
            TimedWord(16.0, 16.4, " a"),
            TimedWord(16.4, 17.5, " b"),
            TimedWord(17.5, 18.0, " c"),
            TimedWord(18.6, 19.0, " d"),
            TimedWord(19.0, 19.9, " e"),
        ]
        handoff = choose_handoff(words, CUT, NEXT_START, speech_audio(), 0)
        assert [word.text for word in handoff.committed] == [" a", " b", " c"]
        assert handoff.boundary_sample == seconds(18.3)
        assert handoff.text == "a b c"
        assert_no_word_lost(words, handoff)

    def test_touching_words_are_split_at_the_quietest_point_before_the_next_word(self):
        audio = speech_audio()
        dip_start = seconds(18.42)
        audio[dip_start : dip_start + 256] = 0.0
        words = [TimedWord(17.4, 18.5, " a"), TimedWord(18.55, 19.0, " b"), TimedWord(19.0, 19.95, " c")]
        handoff = choose_handoff(words, CUT, NEXT_START, audio, 0)
        assert [word.text for word in handoff.committed] == [" a"]
        assert dip_start <= handoff.boundary_sample <= dip_start + 256
        assert_no_word_lost(words, handoff)

    def test_tail_word_near_the_cut_is_never_committed(self):
        words = dense_words(15.0, 20.0)
        handoff = choose_handoff(words, CUT, NEXT_START, speech_audio(), 0)
        assert all(word.end <= CUT / SAMPLE_RATE - 0.3 for word in handoff.committed)
        assert_no_word_lost(words, handoff)

    def test_no_words_commits_nothing(self):
        handoff = choose_handoff([], CUT, NEXT_START, speech_audio(), 0)
        assert handoff.committed == []
        assert handoff.boundary_sample == NEXT_START

    def test_words_that_end_before_the_overlap_are_all_committed(self):
        words = dense_words(1.0, 15.0)
        handoff = choose_handoff(words, CUT, NEXT_START, speech_audio(), 0)
        assert handoff.committed == words
        assert handoff.boundary_sample == NEXT_START

    def test_long_word_inside_the_overlap_is_left_for_the_next_piece(self):
        words = [TimedWord(10.0, 16.9, " a"), TimedWord(17.6, 19.95, " long")]
        handoff = choose_handoff(words, CUT, NEXT_START, speech_audio(), 0)
        assert_no_word_lost(words, handoff)
        assert handoff.boundary_sample <= seconds(17.6 - LEAN_BEFORE_SECONDS) + 1

    def test_word_starting_before_the_overlap_is_committed_rather_than_clipped(self):
        words = [TimedWord(10.0, 16.95, " a"), TimedWord(16.95, 19.95, " straddling")]
        handoff = choose_handoff(words, CUT, NEXT_START, speech_audio(), 0)
        assert [word.text for word in handoff.committed] == [" a", " straddling"]
        assert handoff.boundary_sample == NEXT_START


class TestNoWordLostProperty:
    @pytest.mark.parametrize("seed", range(200))
    def test_random_word_layouts(self, seed):
        generator = np.random.default_rng(seed)
        words = []
        position = float(generator.uniform(0.0, 18.0))
        while position < 20.0:
            duration = float(generator.uniform(0.05, 1.5))
            words.append(TimedWord(position, min(20.0, position + duration), f" w{len(words)}"))
            position += duration + float(generator.choice([0.0, 0.0, 0.05, 0.2, 0.6]))
        audio = (generator.uniform(-0.3, 0.3, seconds(20.0))).astype(np.float32)
        handoff = choose_handoff(words, CUT, NEXT_START, audio, 0)
        assert_no_word_lost(words, handoff)
