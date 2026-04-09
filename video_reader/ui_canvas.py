"""Video display canvas widget."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel

from config import CANVAS_MIN_HEIGHT, CANVAS_MIN_WIDTH
from utils import ndarray_to_qimage


class VideoCanvas(QLabel):
    """Display the current video frame while preserving the original ndarray."""

    def __init__(self, parent: QLabel | None = None) -> None:
        """Initialize the canvas widget.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setMinimumSize(CANVAS_MIN_WIDTH, CANVAS_MIN_HEIGHT)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self._current_frame: np.ndarray | None = None

    def show_frame(self, frame_idx: int, frame: np.ndarray) -> None:
        """Display one RGB frame on the canvas.

        Args:
            frame_idx: Current frame index. Unused by the canvas itself.
            frame: RGB frame image data.
        """
        _ = frame_idx
        self._current_frame = frame
        qimg = ndarray_to_qimage(frame)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def get_current_frame(self) -> np.ndarray | None:
        """Return the currently displayed original frame array."""
        return self._current_frame

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Rescale the current frame when the canvas size changes.

        Args:
            event: Qt resize event.
        """
        super().resizeEvent(event)
        if self._current_frame is not None:
            self.show_frame(0, self._current_frame)


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    canvas = VideoCanvas()

    width = 640
    height = 360
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    red = np.tile(x, (height, 1))
    green = np.tile(y[:, None], (1, width))
    blue = np.full((height, width), 128, dtype=np.uint8)
    frame = np.stack([red, green, blue], axis=2)

    canvas.show_frame(0, frame)
    canvas.show()
    sys.exit(app.exec())
