"""Main application window for synchronized dual-video viewing."""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import QFileDialog, QMainWindow, QVBoxLayout, QWidget

from tools.video_reader.config import WINDOW_TITLE
from tools.video_reader.dual_canvas import DualVideoCanvas
from tools.video_reader.dual_controller import DualPlaybackController, DualVideoInfo
from tools.video_reader.ui_controlbar import ControlBar


class DualMainWindow(QMainWindow):
    """Top-level window combining a dual canvas, controls, and sync playback."""

    def __init__(self) -> None:
        """Initialize the dual main window and wire up the UI."""
        super().__init__()
        self.setWindowTitle(f"{WINDOW_TITLE} - Dual")
        self.controller = DualPlaybackController(self)
        self.canvas = DualVideoCanvas(self)
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
        self.controller.frame_changed_a.connect(self.canvas.show_frame_a)
        self.controller.frame_changed_b.connect(self.canvas.show_frame_b)
        self.controller.frame_changed_a.connect(
            lambda idx, _: self.controlbar.update_position(idx),
        )
        self.controller.video_loaded.connect(self._handle_video_loaded)
        self.controller.save_status.connect(
            lambda msg: self.statusbar.showMessage(msg, 4000),
        )
        self.controlbar.open_requested.connect(self._open_video_dialog)
        self.controlbar.output_dir_requested.connect(self._choose_output_dir_dialog)
        self.controlbar.seek_requested.connect(self.controller.goto)
        self.controlbar.step_size_changed.connect(self.controller.set_step_size)

    def _handle_video_loaded(self, info: DualVideoInfo) -> None:
        """Update UI after a paired video set is loaded.

        Args:
            info: Combined metadata for both videos.
        """
        self.controlbar.set_video_info(info)
        self.statusbar.showMessage(
            f"Loaded pair: {Path(info.info_a.path).name} | {Path(info.info_b.path).name}",
            4000,
        )

    def _open_video_dialog(self) -> None:
        """Open a file dialog, resolve its pair, and load both videos."""
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Paired Video",
            "",
            "Videos (*.mp4 *.mkv *.avi *.mov);;All (*)",
        )
        if not selected_path:
            return

        path_a, path_b = self._resolve_pair_from_selection(Path(selected_path))
        if path_a is None or path_b is None:
            self.statusbar.showMessage("No paired video was selected.", 3000)
            return

        self.canvas.set_video_paths(str(path_a), str(path_b))
        self.controller.load_videos(str(path_a), str(path_b))

    def _choose_output_dir_dialog(self) -> None:
        """Open a dialog for choosing the shared output directory."""
        path = QFileDialog.getExistingDirectory(self, "Choose Output Directory")
        if path:
            self.controller.set_output_dir(Path(path))
            self.statusbar.showMessage(f"Output root: {path}", 3000)

    def _resolve_pair_from_selection(self, selected_path: Path) -> tuple[Path | None, Path | None]:
        """Resolve or manually choose the companion file for a selection.

        Args:
            selected_path: The path chosen in the first dialog.

        Returns:
            Ordered camera 2 / camera 4 file paths, or ``(None, None)``.
        """
        paired_path = self._find_paired_path(selected_path)
        if paired_path is None:
            paired_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select the Paired Video",
                str(selected_path.parent),
                "Videos (*.mp4 *.mkv *.avi *.mov);;All (*)",
            )
            if not paired_path:
                return None, None
            return self._order_pair(selected_path, Path(paired_path))

        return self._order_pair(selected_path, paired_path)

    def _find_paired_path(self, path: Path) -> Path | None:
        """Find the matching paired video in the same directory.

        Args:
            path: One of the paired video paths.

        Returns:
            Matching paired path if found, otherwise ``None``.
        """
        candidates = self._candidate_pair_paths(path)
        for candidate in candidates:
            if candidate.exists() and candidate != path:
                return candidate
        return None

    def _candidate_pair_paths(self, path: Path) -> list[Path]:
        """Generate candidate paired file paths using the supported rules."""
        candidates: list[Path] = []

        dot_match = re.match(r"^(?P<stem>.+)\.(?P<cam>[01])(?P<ext>\.[^.]+)$", path.name)
        if dot_match is not None:
            swapped_cam = "1" if dot_match.group("cam") == "0" else "0"
            candidates.append(
                path.with_name(
                    f"{dot_match.group('stem')}.{swapped_cam}{dot_match.group('ext')}",
                ),
            )

        if re.search(r"cam2", path.name, flags=re.IGNORECASE):
            candidates.append(
                path.with_name(re.sub(r"cam2", "cam4", path.name, flags=re.IGNORECASE)),
            )
        if re.search(r"cam4", path.name, flags=re.IGNORECASE):
            candidates.append(
                path.with_name(re.sub(r"cam4", "cam2", path.name, flags=re.IGNORECASE)),
            )

        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate).lower()
            if key not in seen:
                seen.add(key)
                deduped.append(candidate)
        return deduped

    def _order_pair(self, first: Path, second: Path) -> tuple[Path, Path]:
        """Order two paths as camera 2 first and camera 4 second."""
        role_first = self._classify_camera_role(first)
        role_second = self._classify_camera_role(second)

        if role_first == "cam2" or role_second == "cam4":
            return first, second
        if role_first == "cam4" or role_second == "cam2":
            return second, first
        return first, second

    def _classify_camera_role(self, path: Path) -> str | None:
        """Infer whether a file belongs to camera 2 or camera 4."""
        if re.search(r"\.0\.[^.]+$", path.name):
            return "cam2"
        if re.search(r"\.1\.[^.]+$", path.name):
            return "cam4"
        if re.search(r"cam2", path.name, flags=re.IGNORECASE):
            return "cam2"
        if re.search(r"cam4", path.name, flags=re.IGNORECASE):
            return "cam4"
        return None

    def closeEvent(self, event: QCloseEvent) -> None:
        """Shut down background resources before closing the window.

        Args:
            event: Qt close event.
        """
        self.controller.shutdown()
        super().closeEvent(event)
