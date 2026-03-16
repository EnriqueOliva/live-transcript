from __future__ import annotations

from PySide6.QtCore import QSize, Slot
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QWidget

NUM_BANDS = 24
BAR_GAP = 2
MIN_HEIGHT = 50
DECAY_FACTOR = 0.7


class AudioMeter(QWidget):
    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._levels: list[float] = [0.0] * NUM_BANDS
        self._display: list[float] = [0.0] * NUM_BANDS
        self.setMinimumHeight(MIN_HEIGHT)
        self.setMaximumHeight(70)

    def minimumSizeHint(self) -> QSize:
        return QSize(200, MIN_HEIGHT)

    @Slot(list)
    def update_levels(self, levels: list[float]) -> None:
        self._levels = levels[:NUM_BANDS] if len(levels) >= NUM_BANDS else levels + [0.0] * (NUM_BANDS - len(levels))
        for i in range(NUM_BANDS):
            target = self._levels[i]
            if target > self._display[i]:
                self._display[i] = target
            else:
                self._display[i] = self._display[i] * DECAY_FACTOR
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        bar_width = max(1, (w - BAR_GAP * (NUM_BANDS - 1)) // NUM_BANDS)
        total_width = bar_width * NUM_BANDS + BAR_GAP * (NUM_BANDS - 1)
        x_offset = (w - total_width) // 2

        for i in range(NUM_BANDS):
            level = min(1.0, self._display[i])
            bar_height = max(2, int(level * h * 0.9))
            x = x_offset + i * (bar_width + BAR_GAP)
            y = h - bar_height

            gradient = QLinearGradient(x, h, x, y)
            gradient.setColorAt(0.0, QColor(76, 175, 80))
            gradient.setColorAt(0.5, QColor(255, 235, 59))
            gradient.setColorAt(1.0, QColor(244, 67, 54))

            painter.setBrush(gradient)
            painter.setPen(QColor(0, 0, 0, 0))
            painter.drawRoundedRect(x, y, bar_width, bar_height, 2, 2)

        painter.end()

    def reset(self) -> None:
        self._levels = [0.0] * NUM_BANDS
        self._display = [0.0] * NUM_BANDS
        self.update()
