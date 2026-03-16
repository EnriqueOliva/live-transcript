from PySide6.QtCore import Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QPlainTextEdit


class StatusView(QPlainTextEdit):
    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(100)
        self.setUndoRedoEnabled(False)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setFont(QFont("Segoe UI", 10))
        self.setPlaceholderText("Status updates will appear here...")

    @Slot(str)
    def append_status(self, text: str) -> None:
        self.appendPlainText(text)
