from PySide6.QtCore import Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QPlainTextEdit


class TranscriptView(QPlainTextEdit):
    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(10_000)
        self.setUndoRedoEnabled(False)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setFont(QFont("Segoe UI", 11))

    @Slot(str)
    def append_segment(self, text: str) -> None:
        self.appendPlainText(text)
