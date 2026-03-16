from PySide6.QtCore import QObject, Signal


class WorkerSignals(QObject):
    transcript_segment = Signal(str, float, float)
    status_update = Signal(str, str, str)
    simple_status = Signal(str)
    audio_levels = Signal(list)
    error_occurred = Signal(str)
