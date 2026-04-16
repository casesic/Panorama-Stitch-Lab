"""Bottom control bar for navigation and basic actions."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)


class ControlBar(QWidget):
    """Playback controls, seek slider, and step-size input."""

    open_requested = Signal()
    output_dir_requested = Signal()
    seek_requested = Signal(int)
    step_size_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the control bar widgets and signals.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._is_user_dragging = False
        self._updating = False
        self._total_frames = 0

        layout = QHBoxLayout(self)

        self.open_button = QPushButton("打开", self)
        self.output_dir_button = QPushButton("输出目录", self)
        self.slider = QSlider(Qt.Horizontal, self)
        self.position_label = QLabel("0 / 0", self)
        self.step_label = QLabel("步长:", self)
        self.step_spinbox = QSpinBox(self)

        self.slider.setTickPosition(QSlider.NoTicks)
        self.step_spinbox.setRange(1, 1000)
        self.step_spinbox.setValue(10)

        layout.addWidget(self.open_button)
        layout.addWidget(self.output_dir_button)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.position_label)
        layout.addWidget(self.step_label)
        layout.addWidget(self.step_spinbox)

        self.open_button.clicked.connect(self.open_requested.emit)
        self.output_dir_button.clicked.connect(self.output_dir_requested.emit)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.step_spinbox.valueChanged.connect(self.step_size_changed.emit)

    def set_video_info(self, info: object) -> None:
        """Reset slider range and labels from loaded video metadata.

        Args:
            info: VideoInfo-like object exposing ``total_frames``.
        """
        self._total_frames = int(info.total_frames)
        self.slider.setRange(0, max(0, self._total_frames - 1))
        self.slider.setValue(0)
        self.position_label.setText(f"0 / {self._total_frames}")

    def update_position(self, idx: int) -> None:
        """Update slider and text to reflect the current frame position.

        Args:
            idx: Current frame index.
        """
        if self._is_user_dragging:
            return

        self._updating = True
        self.slider.setValue(idx)
        self._updating = False
        self.position_label.setText(f"{idx} / {self._total_frames}")

    def _on_slider_pressed(self) -> None:
        """Mark the beginning of a user drag operation."""
        self._is_user_dragging = True

    def _on_slider_released(self) -> None:
        """Seek once when the user releases the slider."""
        self._is_user_dragging = False
        self.seek_requested.emit(self.slider.value())


if __name__ == "__main__":
    import sys
    from dataclasses import dataclass

    from PySide6.QtWidgets import QApplication

    @dataclass
    class DummyInfo:
        total_frames: int

    app = QApplication(sys.argv)
    bar = ControlBar()
    bar.set_video_info(DummyInfo(total_frames=1234))
    bar.seek_requested.connect(lambda idx: print("seek_requested:", idx))
    bar.step_size_changed.connect(lambda value: print("step_size_changed:", value))
    bar.show()
    sys.exit(app.exec())
