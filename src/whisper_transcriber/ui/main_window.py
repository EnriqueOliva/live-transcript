from __future__ import annotations

import enum
import logging
import winsound
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from whisper_transcriber.ui.audio_meter import AudioMeter
from whisper_transcriber.ui.log_view import LogView
from whisper_transcriber.ui.status_bar import StatusBar
from whisper_transcriber.ui.status_view import StatusView
from whisper_transcriber.ui.transcript_view import TranscriptView

if TYPE_CHECKING:
    from whisper_transcriber.config.settings import AppSettings
    from whisper_transcriber.logging.log_bridge import GuiBridge
    from whisper_transcriber.ui.signals import WorkerSignals

logger = logging.getLogger(__name__)

MODELS = ["turbo", "large-v3", "medium", "small", "base", "tiny", "distil-large-v3"]
LANGUAGES = ["Auto", "es", "en", "fr", "de", "ja", "zh", "ko", "pt", "ru"]
STOP_BUTTON_STYLE = "background-color: #c62828; color: white;"
SOUND_SUCCESS = winsound.MB_ICONASTERISK
SOUND_FAILURE = winsound.MB_ICONHAND


class SessionState(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    FINISHING = "finishing"


class MainWindow(QMainWindow):
    start_requested = Signal(str, str)
    stop_requested = Signal()
    session_ended = Signal()
    open_folder_requested = Signal()

    def __init__(self, signals: WorkerSignals, gui_bridge: GuiBridge, settings: AppSettings) -> None:
        super().__init__()
        self._settings = settings
        self._state = SessionState.IDLE
        self._close_when_finished = False
        self._had_error = False
        self.setWindowTitle("Live Transcript")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 0)

        controls = QHBoxLayout()
        self._model_combo = QComboBox()
        self._model_combo.addItems(MODELS)
        self._model_combo.setCurrentText(settings.model_size)
        self._lang_combo = QComboBox()
        self._lang_combo.addItems(LANGUAGES)
        self._lang_combo.setCurrentText(settings.language)
        self._mic_check = QCheckBox("Mic")
        self._mic_check.setChecked(settings.record_mic)
        self._start_btn = QPushButton("Start")
        self._start_btn.setFixedWidth(110)
        self._start_btn.clicked.connect(self._on_toggle)
        self._folder_btn = QPushButton("Open Folder")
        self._folder_btn.setFixedWidth(100)
        self._folder_btn.clicked.connect(self.open_folder_requested.emit)

        controls.addWidget(QLabel("Model:"))
        controls.addWidget(self._model_combo)
        controls.addWidget(QLabel("Language:"))
        controls.addWidget(self._lang_combo)
        controls.addWidget(self._mic_check)
        controls.addStretch()
        controls.addWidget(self._folder_btn)
        controls.addWidget(self._start_btn)
        root.addLayout(controls)

        self._audio_meter = AudioMeter()
        root.addWidget(self._audio_meter)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self._transcript_view = TranscriptView()
        self._status_view = StatusView()
        self._log_view = LogView()
        splitter.addWidget(self._transcript_view)
        splitter.addWidget(self._status_view)
        splitter.addWidget(self._log_view)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        root.addWidget(splitter)

        self._status_bar = StatusBar()
        root.addWidget(self._status_bar)

        signals.transcript_line.connect(self._on_transcript_line)
        signals.partial_text.connect(self._transcript_view.set_partial)
        signals.status_update.connect(self._status_bar.update_status)
        signals.progress.connect(self._status_bar.update_progress)
        signals.simple_status.connect(self._status_view.append_status)
        signals.notice.connect(self._status_view.append_status)
        signals.audio_levels.connect(self._audio_meter.update_levels)
        signals.error_occurred.connect(self._on_error)
        signals.session_finished.connect(self._on_session_finished)
        gui_bridge.signal.connect(self._log_view.append_log)

    @property
    def state(self) -> SessionState:
        return self._state

    @Slot()
    def _on_toggle(self) -> None:
        if self._state is SessionState.RUNNING:
            self._begin_finishing()
        elif self._state is SessionState.IDLE:
            self._start_session()

    def _start_session(self) -> None:
        self._settings.model_size = self._model_combo.currentText()
        self._settings.language = self._lang_combo.currentText()
        self._settings.record_mic = self._mic_check.isChecked()
        self._settings.save()
        self._set_controls_enabled(False)
        self._start_btn.setText("Stop")
        self._start_btn.setStyleSheet(STOP_BUTTON_STYLE)
        self._state = SessionState.RUNNING
        self._had_error = False
        self._transcript_view.clear_view()
        self.start_requested.emit(self._settings.model_size, self._settings.language)

    def _begin_finishing(self) -> None:
        self._state = SessionState.FINISHING
        self._start_btn.setText("Finishing...")
        self._start_btn.setStyleSheet("")
        self._start_btn.setEnabled(False)
        self._status_bar.update_status("Finishing", "")
        self._status_view.append_status("Stopping capture, transcribing the remaining audio...")
        self.stop_requested.emit()

    @Slot(str)
    def _on_session_finished(self, summary: str) -> None:
        self.session_ended.emit()
        self._state = SessionState.IDLE
        self._set_controls_enabled(True)
        self._start_btn.setEnabled(True)
        self._start_btn.setText("Start")
        self._start_btn.setStyleSheet("")
        self._status_bar.reset()
        self._audio_meter.reset()
        self._status_view.append_status(summary)
        _play(SOUND_FAILURE if self._had_error or "problems" in summary else SOUND_SUCCESS)
        if self._close_when_finished:
            self.close()

    @Slot(str, float, float, str)
    def _on_transcript_line(self, text: str, start: float, end: float, style: str) -> None:
        self._transcript_view.append_line(text, style)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        logger.error("Worker error: %s", message)
        self._had_error = True
        self._status_view.append_status(f"Error: {message}")
        _play(SOUND_FAILURE)
        if self._state is SessionState.RUNNING:
            self._begin_finishing()

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._model_combo.setEnabled(enabled)
        self._lang_combo.setEnabled(enabled)
        self._mic_check.setEnabled(enabled)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._state is SessionState.IDLE or (
            self._close_when_finished and self._confirm_quit_while_finishing()
        ):
            self._settings.save()
            event.accept()
        else:
            self._close_when_finished = True
            if self._state is SessionState.RUNNING:
                self._begin_finishing()
            self._status_view.append_status("The window closes by itself once every word is transcribed")
            event.ignore()

    def _confirm_quit_while_finishing(self) -> bool:
        answer = QMessageBox.question(
            self,
            "Transcription still finishing",
            "The last part of the audio is still being transcribed.\n\n"
            "Quit anyway? The full audio is saved as recording.wav in the session folder "
            "and can be transcribed later.",
        )
        return answer == QMessageBox.StandardButton.Yes


def _play(sound: int) -> None:
    try:
        winsound.MessageBeep(sound)
    except RuntimeError:
        logger.debug("Could not play notification sound")
