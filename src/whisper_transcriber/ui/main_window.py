from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
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
LANGUAGES = ["Auto", "en", "es", "fr", "de", "ja", "zh", "ko", "pt", "ru"]


class MainWindow(QMainWindow):
    start_requested = Signal(str, str)
    stop_requested = Signal()
    open_folder_requested = Signal()

    def __init__(self, signals: WorkerSignals, gui_bridge: GuiBridge, settings: AppSettings) -> None:
        super().__init__()
        self._settings = settings
        self._is_running = False
        self.setWindowTitle("Whisper Transcriber")
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
        self._start_btn.setFixedWidth(100)
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

        signals.transcript_segment.connect(self._on_transcript_segment)
        signals.status_update.connect(self._status_bar.update_status)
        signals.simple_status.connect(self._status_view.append_status)
        signals.audio_levels.connect(self._audio_meter.update_levels)
        signals.error_occurred.connect(self._on_error)
        gui_bridge.signal.connect(self._log_view.append_log)

    @Slot()
    def _on_toggle(self) -> None:
        if self._is_running:
            self._stop_session()
        else:
            self._start_session()

    def _start_session(self) -> None:
        self._settings.model_size = self._model_combo.currentText()
        self._settings.language = self._lang_combo.currentText()
        self._settings.record_mic = self._mic_check.isChecked()
        self._settings.save()
        self._model_combo.setEnabled(False)
        self._lang_combo.setEnabled(False)
        self._mic_check.setEnabled(False)
        self._start_btn.setText("Stop")
        self._start_btn.setStyleSheet("background-color: #c62828; color: white;")
        self._is_running = True
        self.start_requested.emit(self._settings.model_size, self._settings.language)

    def _stop_session(self) -> None:
        self._model_combo.setEnabled(True)
        self._lang_combo.setEnabled(True)
        self._mic_check.setEnabled(True)
        self._start_btn.setText("Start")
        self._start_btn.setStyleSheet("")
        self._is_running = False
        self._status_bar.reset()
        self._audio_meter.reset()
        self.stop_requested.emit()

    @Slot(str, float, float)
    def _on_transcript_segment(self, text: str, start: float, end: float) -> None:
        self._transcript_view.append_segment(text)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        logger.error("Worker error: %s", message)
        if self._is_running:
            self._stop_session()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._is_running:
            self.stop_requested.emit()
        self._settings.save()
        event.accept()
