"""Dual-video playback controller for synchronized browsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from tools.video_reader.config import DEFAULT_STEP_SIZE
from tools.video_reader.controller import PlaybackController
from tools.video_reader.decoder import VideoInfo


@dataclass(slots=True)
class DualVideoInfo:
    """Combined metadata for a synchronized pair of videos."""

    info_a: VideoInfo
    info_b: VideoInfo
    total_frames: int


class DualPlaybackController(QObject):
    """Coordinate two playback controllers while keeping navigation in sync."""

    frame_changed_a = Signal(int, object)
    frame_changed_b = Signal(int, object)
    video_loaded = Signal(object)
    save_status = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the dual playback controller.

        Args:
            parent: Optional Qt parent object.
        """
        super().__init__(parent)
        self._controller_a = PlaybackController(self)
        self._controller_b = PlaybackController(self)
        self._video_info_a: VideoInfo | None = None
        self._video_info_b: VideoInfo | None = None
        self._video_path_a: Path | None = None
        self._video_path_b: Path | None = None
        self._output_root: Path | None = None
        self._current_idx = 0
        self._sync_target: int | None = None
        self._pending_sync_updates = 0
        self._pending_save_messages: list[str] = []
        self._pending_save_count = 0
        self._step_size = DEFAULT_STEP_SIZE

        self._controller_a.frame_changed.connect(self._handle_frame_changed_a)
        self._controller_b.frame_changed.connect(self._handle_frame_changed_b)
        self._controller_a.video_loaded.connect(self._handle_video_loaded_a)
        self._controller_b.video_loaded.connect(self._handle_video_loaded_b)
        self._controller_a.save_status.connect(
            lambda msg: self._handle_save_status("cam2", msg),
        )
        self._controller_b.save_status.connect(
            lambda msg: self._handle_save_status("cam4", msg),
        )

    def load_videos(self, path_a: str, path_b: str) -> None:
        """Load a synchronized video pair.

        Args:
            path_a: Camera 2 video path.
            path_b: Camera 4 video path.
        """
        self._video_path_a = Path(path_a)
        self._video_path_b = Path(path_b)
        self._video_info_a = None
        self._video_info_b = None
        self._current_idx = 0
        self._sync_target = 0
        self._pending_sync_updates = 2
        self._apply_output_dirs()
        self._controller_a.load_video(path_a)
        self._controller_b.load_video(path_b)

    def goto(self, target_idx: int) -> None:
        """Display the same target frame on both videos.

        Args:
            target_idx: Requested synchronized frame index.
        """
        max_idx = self._max_common_index()
        if max_idx is None:
            return

        target = max(0, min(target_idx, max_idx))
        if target == self._current_idx:
            return

        self._sync_target = target
        self._pending_sync_updates = 2
        self._controller_a.goto(target)
        self._controller_b.goto(target)
        self._current_idx = target

    def step(self, delta: int) -> None:
        """Move relative to the current synchronized frame index.

        Args:
            delta: Signed frame offset.
        """
        self.goto(self._current_idx + delta)

    def step_by_size(self, direction: int) -> None:
        """Move by the configured step size on both videos.

        Args:
            direction: Usually ``1`` for forward or ``-1`` for backward.
        """
        self.goto(self._current_idx + direction * self._step_size)

    def goto_first(self) -> None:
        """Jump both videos to the first frame."""
        self.goto(0)

    def goto_last(self) -> None:
        """Jump both videos to the last common frame."""
        max_idx = self._max_common_index()
        if max_idx is not None:
            self.goto(max_idx)

    def set_step_size(self, n: int) -> None:
        """Update the jump step size on both child controllers.

        Args:
            n: Desired jump size in frames.
        """
        self._step_size = max(1, n)
        self._controller_a.set_step_size(n)
        self._controller_b.set_step_size(n)

    def set_output_dir(self, path: Path) -> None:
        """Set the shared output root used for saving both videos.

        Args:
            path: Root output directory.
        """
        self._output_root = Path(path)
        self._apply_output_dirs()

    def save_current(self) -> None:
        """Save the current synchronized frame for both videos."""
        if self._output_root is None:
            self.save_status.emit("Please choose an output directory first.")
            return
        if self._video_path_a is None or self._video_path_b is None:
            self.save_status.emit("Please open a paired video set first.")
            return

        self._pending_save_messages = []
        self._pending_save_count = 2
        self._apply_output_dirs()
        self._controller_a.save_current()
        self._controller_b.save_current()

    def shutdown(self) -> None:
        """Stop both child controllers and their background workers."""
        self._controller_a.shutdown()
        self._controller_b.shutdown()

    def _handle_frame_changed_a(self, idx: int, frame: object) -> None:
        """Forward camera 2 frame updates using the synchronized index."""
        synced_idx = self._resolve_synced_index(idx)
        self.frame_changed_a.emit(synced_idx, frame)

    def _handle_frame_changed_b(self, idx: int, frame: object) -> None:
        """Forward camera 4 frame updates using the synchronized index."""
        synced_idx = self._resolve_synced_index(idx)
        self.frame_changed_b.emit(synced_idx, frame)

    def _handle_video_loaded_a(self, info: VideoInfo) -> None:
        """Store camera 2 metadata and emit combined metadata when ready."""
        self._video_info_a = info
        self._emit_video_loaded_if_ready()

    def _handle_video_loaded_b(self, info: VideoInfo) -> None:
        """Store camera 4 metadata and emit combined metadata when ready."""
        self._video_info_b = info
        self._emit_video_loaded_if_ready()

    def _emit_video_loaded_if_ready(self) -> None:
        """Emit combined metadata once both videos are loaded."""
        if self._video_info_a is None or self._video_info_b is None:
            return

        total_frames = min(
            self._video_info_a.total_frames,
            self._video_info_b.total_frames,
        )
        self._current_idx = 0
        self.video_loaded.emit(
            DualVideoInfo(
                info_a=self._video_info_a,
                info_b=self._video_info_b,
                total_frames=total_frames,
            ),
        )

    def _handle_save_status(self, label: str, message: str) -> None:
        """Aggregate one save status message from a child controller."""
        if self._pending_save_count <= 0:
            self.save_status.emit(f"{label}: {message}")
            return

        self._pending_save_messages.append(f"{label}: {message}")
        self._pending_save_count -= 1
        if self._pending_save_count == 0:
            self.save_status.emit(" | ".join(self._pending_save_messages))
            self._pending_save_messages = []

    def _resolve_synced_index(self, fallback_idx: int) -> int:
        """Return the synchronized index used for outward-facing signals."""
        if self._sync_target is not None:
            synced_idx = self._sync_target
            self._pending_sync_updates = max(0, self._pending_sync_updates - 1)
            if self._pending_sync_updates == 0:
                self._sync_target = None
        else:
            synced_idx = fallback_idx
        self._current_idx = synced_idx
        return synced_idx

    def _apply_output_dirs(self) -> None:
        """Push the current output root to both child controllers."""
        if self._output_root is None:
            return
        if self._video_path_a is not None:
            self._controller_a.set_output_dir(self._output_root / self._video_path_a.stem)
        if self._video_path_b is not None:
            self._controller_b.set_output_dir(self._output_root / self._video_path_b.stem)

    def _max_common_index(self) -> int | None:
        """Return the largest valid synchronized frame index."""
        if self._video_info_a is None or self._video_info_b is None:
            return None
        return min(
            self._video_info_a.total_frames,
            self._video_info_b.total_frames,
        ) - 1
