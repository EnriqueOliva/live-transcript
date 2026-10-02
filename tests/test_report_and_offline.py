from livevox.offline import transcribe_samples
from livevox.session.report import SessionReport
from livevox.stt.vad import FRAME_SAMPLES
from tests.helpers import FakeEngine, RecordingEvents, build, result, silence, tone, words_for


class TestReport:
    def test_complete_session(self):
        report = SessionReport(captured_samples=16000, covered_samples=16000)
        assert report.is_complete
        assert "100%" in report.summary()
        assert "COMPLETE" in report.to_text()

    def test_any_problem_makes_the_session_incomplete(self):
        for field, value in (("gap_samples", 1), ("failed_pieces", 1), ("overflow_events", 1), ("model_failed", True)):
            report = SessionReport(captured_samples=16000, covered_samples=16000, recording_path="rec.wav")
            setattr(report, field, value)
            assert not report.is_complete
            assert "problems" in report.summary()
            assert "rec.wav" in report.summary()


class EnergyVad:
    def __init__(self, *arguments):
        pass

    def __call__(self, frames):
        import numpy as np

        energy = np.sqrt(np.mean(np.square(frames.reshape(-1, FRAME_SAMPLES)), axis=1))
        return np.where(energy > 0.01, 0.9, 0.05).astype(np.float32)


class TestOfflineTranscription:
    def test_file_mode_uses_the_same_lossless_path(self, tmp_path, monkeypatch):
        import livevox.offline as offline

        monkeypatch.setattr(offline, "StreamingVad", EnergyVad)
        audio = build(tone(3.0), silence(1.0), tone(4.0), silence(1.0))
        engine = FakeEngine(responses=[
            result("uno", words=words_for("uno", 0.2)),
            result("dos", words=words_for("dos", 0.4)),
        ])
        report = transcribe_samples(audio, engine, tmp_path, "es", RecordingEvents())
        assert report.is_complete
        assert report.covered_samples == audio.size
        assert (tmp_path / "transcript.txt").read_text(encoding="utf-8").splitlines() == ["uno", "dos"]
        assert (tmp_path / "session_report.txt").exists()
