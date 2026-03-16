from PySide6.QtCore import Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget


class StatusBar(QWidget):
    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)

        self._state_label = QLabel("Idle")
        self._gpu_label = QLabel("")
        self._buffer_label = QLabel("")

        layout.addWidget(self._state_label)
        layout.addStretch()
        layout.addWidget(self._gpu_label)
        layout.addWidget(self._buffer_label)

    @Slot(str, str, str)
    def update_status(self, state: str, gpu_info: str, buffer_info: str) -> None:
        self._state_label.setText(state)
        color = "#4caf50" if "Recording" in state else "#ff9800" if "Loading" in state else "#9e9e9e"
        self._state_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        if gpu_info:
            self._gpu_label.setText(gpu_info)
        if buffer_info:
            self._buffer_label.setText(buffer_info)

    def reset(self) -> None:
        self._state_label.setText("Idle")
        self._state_label.setStyleSheet("color: #9e9e9e; font-weight: bold;")
        self._gpu_label.setText("")
        self._buffer_label.setText("")
