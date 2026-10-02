from PySide6.QtCore import Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

RECORDING_COLOR = "#4caf50"
BUSY_COLOR = "#ff9800"
IDLE_COLOR = "#9e9e9e"
LAG_WARNING_SECONDS = 30.0


class StatusBar(QWidget):
    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)

        self._state_label = QLabel("Idle")
        self._device_label = QLabel("")
        self._lag_label = QLabel("")

        layout.addWidget(self._state_label)
        layout.addStretch()
        layout.addWidget(self._device_label)
        layout.addWidget(self._lag_label)
        self.reset()

    @Slot(str, str)
    def update_status(self, state: str, detail: str) -> None:
        self._state_label.setText(state)
        if "Recording" in state:
            color = RECORDING_COLOR
        elif state in ("Idle", ""):
            color = IDLE_COLOR
        else:
            color = BUSY_COLOR
        self._state_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        if detail:
            self._device_label.setText(detail)

    @Slot(float, float)
    def update_progress(self, captured_seconds: float, transcribed_seconds: float) -> None:
        lag = max(0.0, captured_seconds - transcribed_seconds)
        minutes, seconds = divmod(int(captured_seconds), 60)
        self._lag_label.setText(f"{minutes:02d}:{seconds:02d} captured  |  {lag:.1f}s behind")
        color = BUSY_COLOR if lag > LAG_WARNING_SECONDS else IDLE_COLOR
        self._lag_label.setStyleSheet(f"color: {color};")

    def reset(self) -> None:
        self._state_label.setText("Idle")
        self._state_label.setStyleSheet(f"color: {IDLE_COLOR}; font-weight: bold;")
        self._device_label.setText("")
        self._lag_label.setText("")
