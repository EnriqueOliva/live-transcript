from PySide6.QtCore import Slot
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

from whisper_transcriber.session.events import LineStyle

NORMAL_COLOR = QColor(220, 220, 220)
UNCERTAIN_COLOR = QColor(130, 130, 130)
PARTIAL_COLOR = QColor(120, 160, 200)
FAILURE_COLOR = QColor(239, 154, 154)
UNLIMITED_BLOCKS = 0
BOTTOM_TOLERANCE = 4


def _character_format(color: QColor, italic: bool) -> QTextCharFormat:
    character_format = QTextCharFormat()
    character_format.setForeground(color)
    character_format.setFontItalic(italic)
    return character_format


class TranscriptView(QPlainTextEdit):
    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(UNLIMITED_BLOCKS)
        self.setUndoRedoEnabled(False)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setFont(QFont("Segoe UI", 11))
        self._partial_active = False
        self._formats = {
            LineStyle.NORMAL.value: _character_format(NORMAL_COLOR, italic=False),
            LineStyle.UNCERTAIN.value: _character_format(UNCERTAIN_COLOR, italic=True),
            LineStyle.FAILURE.value: _character_format(FAILURE_COLOR, italic=True),
            LineStyle.NOTICE.value: _character_format(UNCERTAIN_COLOR, italic=True),
        }
        self._partial_format = _character_format(PARTIAL_COLOR, italic=True)

    @Slot(str)
    def append_segment(self, text: str) -> None:
        self.append_line(text, LineStyle.NORMAL.value)

    @Slot(str, str)
    def append_line(self, text: str, style: str) -> None:
        follow = self._is_at_bottom()
        self._remove_partial()
        self._insert_block(text, self._formats.get(style, self._formats[LineStyle.NORMAL.value]))
        if follow:
            self._scroll_to_bottom()

    @Slot(str)
    def set_partial(self, text: str) -> None:
        follow = self._is_at_bottom()
        self._remove_partial()
        if text:
            self._insert_block(text, self._partial_format)
            self._partial_active = True
        if follow:
            self._scroll_to_bottom()

    def has_partial(self) -> bool:
        return self._partial_active

    def committed_text(self) -> str:
        text = self.toPlainText()
        if self._partial_active:
            lines = text.split("\n")
            return "\n".join(lines[:-1])
        else:
            return text

    def clear_view(self) -> None:
        self.clear()
        self._partial_active = False

    def _insert_block(self, text: str, character_format: QTextCharFormat) -> None:
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.document().isEmpty():
            cursor.insertBlock()
        cursor.insertText(text, character_format)

    def _remove_partial(self) -> None:
        if not self._partial_active:
            return
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        if self.document().blockCount() > 1:
            cursor.deletePreviousChar()
        self._partial_active = False

    def _is_at_bottom(self) -> bool:
        scroll_bar = self.verticalScrollBar()
        return scroll_bar.value() >= scroll_bar.maximum() - BOTTOM_TOLERANCE

    def _scroll_to_bottom(self) -> None:
        scroll_bar = self.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
