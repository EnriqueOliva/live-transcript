from PySide6.QtCore import QObject, Signal

from whisper_transcriber.session.events import TranscriptLine


class WorkerSignals(QObject):
    transcript_line = Signal(str, float, float, str)
    partial_text = Signal(str)
    status_update = Signal(str, str)
    simple_status = Signal(str)
    progress = Signal(float, float)
    audio_levels = Signal(list)
    notice = Signal(str)
    error_occurred = Signal(str)
    session_finished = Signal(str)


class QtSessionEvents:
    def __init__(self, signals: WorkerSignals) -> None:
        self._signals = signals

    def status(self, state: str, detail: str = "") -> None:
        self._signals.status_update.emit(state, detail)

    def message(self, text: str) -> None:
        self._signals.simple_status.emit(text)

    def transcript_line(self, line: TranscriptLine) -> None:
        self._signals.transcript_line.emit(line.text, line.start, line.end, line.style.value)

    def partial_text(self, text: str) -> None:
        self._signals.partial_text.emit(text)

    def progress(self, captured_seconds: float, transcribed_seconds: float) -> None:
        self._signals.progress.emit(captured_seconds, transcribed_seconds)

    def audio_levels(self, levels: list[float]) -> None:
        self._signals.audio_levels.emit(levels)

    def notice(self, text: str) -> None:
        self._signals.notice.emit(text)

    def error(self, text: str) -> None:
        self._signals.error_occurred.emit(text)

    def finished(self, summary: str) -> None:
        self._signals.session_finished.emit(summary)
