from collections import namedtuple

import numpy as np
import pytest

Segment = namedtuple("Segment", ["start", "end", "text", "no_speech_prob", "avg_logprob", "compression_ratio", "words"])


def make_segment(start, end, text, no_speech_prob=0.1, avg_logprob=-0.3, compression_ratio=1.2, words=None):
    return Segment(
        start=start,
        end=end,
        text=text,
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
        words=words,
    )


@pytest.fixture()
def sample_audio_int16_stereo():
    sample_rate = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * 440 * t)
    int16_mono = (tone * 32767).astype(np.int16)
    stereo = np.column_stack([int16_mono, int16_mono])
    return stereo.flatten().tobytes(), sample_rate, 2


@pytest.fixture()
def sample_audio_float32_mono():
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return tone, sample_rate, 1


@pytest.fixture()
def sample_segments():
    return [
        make_segment(0.0, 3.5, "Hello and welcome to the lecture."),
        make_segment(3.5, 6.2, "Today we will cover chapter five."),
        make_segment(6.8, 9.1, "Please open your textbooks."),
        make_segment(9.5, 12.0, "Let us begin with the first topic."),
    ]


@pytest.fixture()
def settings_path(tmp_path):
    return tmp_path / "config" / "settings.json"
