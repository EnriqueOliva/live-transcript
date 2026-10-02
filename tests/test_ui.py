import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from livevox.config.settings import AppSettings
from livevox.logging.log_bridge import GuiBridge
from livevox.ui import main_window
from livevox.ui.main_window import MainWindow, SessionState
from livevox.ui.signals import WorkerSignals
from livevox.ui.transcript_view import TranscriptView


@pytest.fixture(scope="module")
def qt_application():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture()
def window(qt_application, tmp_path, monkeypatch):
    monkeypatch.setattr(main_window, "_play", lambda sound: None)
    settings = AppSettings()
    monkeypatch.setattr(settings, "save", lambda path=None: None)
    signals = WorkerSignals()
    created = MainWindow(signals, GuiBridge(), settings)
    requests = {"start": [], "stop": 0, "ended": 0}
    created.start_requested.connect(lambda model, language: requests["start"].append((model, language)))
    created.stop_requested.connect(lambda: requests.__setitem__("stop", requests["stop"] + 1))
    created.session_ended.connect(lambda: requests.__setitem__("ended", requests["ended"] + 1))
    yield created, signals, requests
    created.deleteLater()


class TestTranscriptView:
    def test_partial_line_is_replaced_by_the_final_text(self, qt_application):
        view = TranscriptView()
        view.append_line("primera", "normal")
        view.set_partial("segunda en cur")
        assert view.toPlainText() == "primera\nsegunda en cur"
        view.set_partial("segunda en curso")
        assert view.toPlainText() == "primera\nsegunda en curso"
        view.append_line("segunda en curso final", "normal")
        assert view.toPlainText() == "primera\nsegunda en curso final"
        assert not view.has_partial()

    def test_committed_text_never_includes_the_partial(self, qt_application):
        view = TranscriptView()
        view.append_line("uno", "normal")
        view.set_partial("dos")
        assert view.committed_text() == "uno"

    def test_partial_on_an_empty_view(self, qt_application):
        view = TranscriptView()
        view.set_partial("hola")
        view.set_partial("")
        assert view.toPlainText() == ""
        view.append_line("hola", "uncertain")
        assert view.toPlainText() == "hola"

    def test_the_view_keeps_every_line(self, qt_application):
        view = TranscriptView()
        for index in range(12_000):
            view.append_line(f"linea {index}", "normal")
        assert view.document().blockCount() == 12_000


class TestMainWindowLifecycle:
    def test_stop_waits_for_the_last_words_before_going_idle(self, window):
        created, signals, requests = window
        created._on_toggle()
        assert created.state is SessionState.RUNNING
        assert requests["start"] == [("turbo", "es")]
        created._on_toggle()
        assert created.state is SessionState.FINISHING
        assert requests["stop"] == 1
        created._on_toggle()
        assert requests["stop"] == 1
        signals.session_finished.emit("Session complete")
        assert created.state is SessionState.IDLE
        assert requests["ended"] == 1

    def test_closing_during_a_session_finishes_it_first(self, window, monkeypatch):
        created, signals, requests = window
        created._on_toggle()
        created.show()
        created.close()
        assert created.isVisible()
        assert created.state is SessionState.FINISHING
        assert requests["stop"] == 1
        signals.session_finished.emit("Session complete")
        assert not created.isVisible()

    def test_second_close_asks_before_abandoning_the_tail(self, window, monkeypatch):
        created, _, _ = window
        monkeypatch.setattr(QMessageBox, "question", lambda *arguments, **keyword: QMessageBox.StandardButton.No)
        created._on_toggle()
        created.show()
        created.close()
        created.close()
        assert created.isVisible()
        monkeypatch.setattr(QMessageBox, "question", lambda *arguments, **keyword: QMessageBox.StandardButton.Yes)
        created.close()
        assert not created.isVisible()

    def test_errors_stop_the_session_through_the_normal_path(self, window):
        created, signals, requests = window
        created._on_toggle()
        signals.error_occurred.emit("model failed")
        assert created.state is SessionState.FINISHING
        assert requests["stop"] == 1
