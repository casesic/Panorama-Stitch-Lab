"""Main application window for the video frame browser."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import QFileDialog, QMainWindow, QVBoxLayout, QWidget

from config import WINDOW_TITLE
from controller import PlaybackController
from ui_canvas import VideoCanvas
from ui_controlbar import ControlBar


class MainWindow(QMainWindow):
    """Top-level window combining canvas, controls, and playback controller."""

    def __init__(self) -> None:
        """Initialize the main window and wire up the UI."""
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.controller = PlaybackController(self)
        self.canvas = VideoCanvas(self)
        self.controlbar = ControlBar(self)
        self.statusbar = self.statusBar()

        self._setup_layout()
        self._setup_shortcuts()
        self._connect_signals()

    def _setup_layout(self) -> None:
        """Create and assign the main vertical layout."""
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.controlbar)
        self.setCentralWidget(central)

    def _setup_shortcuts(self) -> None:
        """Register keyboard shortcuts for playback and file actions."""
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self.controller.step(+1))
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self.controller.step(-1))
        QShortcut(
            QKeySequence("Shift+Right"),
            self,
            activated=lambda: self.controller.step_by_size(+1),
        )
        QShortcut(
            QKeySequence("Shift+Left"),
            self,
            activated=lambda: self.controller.step_by_size(-1),
        )
        QShortcut(QKeySequence("Ctrl+Right"), self, activated=lambda: self.controller.step(+30))
        QShortcut(QKeySequence("Ctrl+Left"), self, activated=lambda: self.controller.step(-30))
        QShortcut(QKeySequence("Home"), self, activated=self.controller.goto_first)
        QShortcut(QKeySequence("End"), self, activated=self.controller.goto_last)
        QShortcut(QKeySequence("S"), self, activated=self.controller.save_current)
        QShortcut(QKeySequence("O"), self, activated=self._open_video_dialog)
        QShortcut(QKeySequence("D"), self, activated=self._choose_output_dir_dialog)

    def _connect_signals(self) -> None:
        """Connect controller signals to widgets and dialogs."""
        self.controller.frame_changed.connect(self.canvas.show_frame)
        self.controller.frame_changed.connect(
            lambda idx, _: self.controlbar.update_position(idx),
        )
        self.controller.video_loaded.connect(self.controlbar.set_video_info)
        self.controller.save_status.connect(
            lambda msg: self.statusbar.showMessage(msg, 3000),
        )
        self.controlbar.open_requested.connect(self._open_video_dialog)
        self.controlbar.output_dir_requested.connect(self._choose_output_dir_dialog)
        self.controlbar.seek_requested.connect(self.controller.goto)
        self.controlbar.step_size_changed.connect(self.controller.set_step_size)

    def _open_video_dialog(self) -> None:
        """Open a file dialog and load the selected video."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开视频",
            "",
            "Videos (*.mp4 *.mkv *.avi *.mov);;All (*)",
        )
        if path:
            self.controller.load_video(path)
            self.statusbar.showMessage(f"已加载 {Path(path).name}", 3000)

    def _choose_output_dir_dialog(self) -> None:
        """Open a dialog for choosing the output image directory."""
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.controller.set_output_dir(Path(path))
            self.statusbar.showMessage(f"输出目录:{path}", 3000)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Shut down background resources before closing the window.

        Args:
            event: Qt close event.
        """
        self.controller.shutdown()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())
