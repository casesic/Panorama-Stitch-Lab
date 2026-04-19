"""Dual-video display canvas widget."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from tools.video_reader.ui_canvas import VideoCanvas


class DualVideoCanvas(QWidget):
    """Display two synchronized videos side by side with status labels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the dual video canvas widget.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._path_a: Path | None = None
        self._path_b: Path | None = None
        self._frame_idx_a = 0
        self._frame_idx_b = 0

        self.label_a = QLabel("cam2: <not loaded>", self)
        self.label_b = QLabel("cam4: <not loaded>", self)
        self.canvas_a = VideoCanvas(self)
        self.canvas_b = VideoCanvas(self)

        self._setup_layout()

    def set_video_paths(self, path_a: str, path_b: str) -> None:
        """Update the displayed file names for both panels.

        Args:
            path_a: Camera 2 video path.
            path_b: Camera 4 video path.
        """
        self._path_a = Path(path_a)
        self._path_b = Path(path_b)
        self._frame_idx_a = 0
        self._frame_idx_b = 0
        self._refresh_labels()

    def show_frame_a(self, frame_idx: int, frame: np.ndarray) -> None:
        """Display one frame for camera 2.

        Args:
            frame_idx: Current synchronized frame index.
            frame: RGB frame image data.
        """
        self._frame_idx_a = frame_idx
        self.canvas_a.show_frame(frame_idx, frame)
        self._refresh_labels()

    def show_frame_b(self, frame_idx: int, frame: np.ndarray) -> None:
        """Display one frame for camera 4.

        Args:
            frame_idx: Current synchronized frame index.
            frame: RGB frame image data.
        """
        self._frame_idx_b = frame_idx
        self.canvas_b.show_frame(frame_idx, frame)
        self._refresh_labels()

    def _setup_layout(self) -> None:
        """Create the side-by-side layout."""
        layout = QHBoxLayout(self)
        layout.addLayout(self._create_panel(self.label_a, self.canvas_a), 1)
        layout.addLayout(self._create_panel(self.label_b, self.canvas_b), 1)

    def _create_panel(self, label: QLabel, canvas: VideoCanvas) -> QVBoxLayout:
        """Create one labeled video panel."""
        label.setWordWrap(True)
        panel = QVBoxLayout()
        panel.addWidget(label)
        panel.addWidget(canvas, 1)
        return panel

    def _refresh_labels(self) -> None:
        """Refresh the per-video file name and frame labels."""
        name_a = self._path_a.name if self._path_a is not None else "<not loaded>"
        name_b = self._path_b.name if self._path_b is not None else "<not loaded>"
        self.label_a.setText(f"cam2: {name_a} | frame {self._frame_idx_a}")
        self.label_b.setText(f"cam4: {name_b} | frame {self._frame_idx_b}")
