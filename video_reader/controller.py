"""Playback controller that connects decoder, cache, provider, and saver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from config import DEFAULT_STEP_SIZE
from decoder import VideoDecoder, VideoInfo
from frame_cache import FrameCache
from frame_provider import FrameProvider
from frame_saver import FrameSaver


class PlaybackController(QObject):
    """Coordinate playback navigation, caching, prefetch, and frame saving."""

    frame_changed = Signal(int, object)
    video_loaded = Signal(object)
    save_status = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the playback controller.

        Args:
            parent: Optional Qt parent object.
        """
        super().__init__(parent)
        self._decoder: VideoDecoder | None = None
        self._cache: FrameCache | None = None
        self._provider: FrameProvider | None = None
        self._saver = FrameSaver()
        self._video_info: VideoInfo | None = None
        self._current_idx: int = 0
        self._output_dir: Path | None = None
        self._step_size: int = DEFAULT_STEP_SIZE
        self._last_frame: np.ndarray | None = None

    def load_video(self, path: str) -> None:
        """Open a new video and initialize cache and prefetch provider.

        Args:
            path: Video file path.
        """
        if self._provider is not None:
            self._provider.stop()
        if self._decoder is not None:
            self._decoder.close()

        self._decoder = VideoDecoder()
        self._video_info = self._decoder.open(path)
        self._cache = FrameCache()
        self._provider = FrameProvider(path, self._cache)
        self._provider.start()
        self._current_idx = 0
        self._last_frame = None
        self.video_loaded.emit(self._video_info)
        self.goto(0)

    def goto(self, target_idx: int) -> None:
        """Display a target frame index using cache, sequential decode, or seek.

        Args:
            target_idx: Requested frame index.
        """
        if self._video_info is None:
            return
        if self._decoder is None or self._cache is None or self._provider is None:
            return

        target = max(0, min(target_idx, self._video_info.total_frames - 1))
        if target == self._current_idx and self._last_frame is not None:
            return

        frame = self._cache.get(target)
        if frame is None:
            if target == self._current_idx + 1:
                decoded_idx, frame = self._decoder.read_next()
                self._cache.put(decoded_idx, frame)
                target = decoded_idx
            else:
                decoded_idx, frame = self._decoder.seek_and_read(target)
                self._cache.put(decoded_idx, frame)
                target = decoded_idx

        self._current_idx = target
        self._last_frame = frame
        self._cache.set_current(target)
        self._provider.request_prefetch_around(target)
        self.frame_changed.emit(target, frame)

    def step(self, delta: int) -> None:
        """Move relative to the current frame index.

        Args:
            delta: Signed frame offset.
        """
        self.goto(self._current_idx + delta)

    def step_by_size(self, direction: int) -> None:
        """Move by the configured step size.

        Args:
            direction: Usually ``1`` for forward or ``-1`` for backward.
        """
        self.goto(self._current_idx + direction * self._step_size)

    def goto_first(self) -> None:
        """Jump to the first frame."""
        self.goto(0)

    def goto_last(self) -> None:
        """Jump to the last frame if a video is loaded."""
        if self._video_info is not None:
            self.goto(self._video_info.total_frames - 1)

    def set_step_size(self, n: int) -> None:
        """Update the jump step size.

        Args:
            n: Desired jump size in frames.
        """
        self._step_size = max(1, n)

    def set_output_dir(self, path: Path) -> None:
        """Set the output directory for saved PNG frames.

        Args:
            path: Output directory path.
        """
        self._output_dir = Path(path)

    def save_current(self) -> None:
        """Save the current frame as a PNG file."""
        if self._last_frame is None:
            self.save_status.emit("没有可保存的帧")
            return
        if self._output_dir is None:
            self.save_status.emit("请先选择输出目录(按 D)")
            return

        def on_done(path: str, success: bool) -> None:
            msg = f"已保存 {Path(path).name}" if success else "保存失败"
            self.save_status.emit(msg)

        self._saver.save_async(
            self._last_frame,
            self._current_idx,
            self._output_dir,
            on_done=on_done,
        )

    def shutdown(self) -> None:
        """Stop background services and close active resources."""
        if self._provider is not None:
            self._provider.stop()
        if self._decoder is not None:
            self._decoder.close()
        self._saver.shutdown(wait=False)


if __name__ == "__main__":
    import sys
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication(sys.argv)
    video_path = input("Enter a video path: ").strip()

    controller = PlaybackController()

    def handle_video_loaded(info: VideoInfo) -> None:
        print("video_loaded:", info)

    def handle_frame_changed(idx: int, frame: object) -> None:
        frame_array = frame
        print("frame_changed:", idx, frame_array.shape)

    controller.video_loaded.connect(handle_video_loaded)
    controller.frame_changed.connect(handle_frame_changed)

    controller.load_video(video_path)
    controller.goto(0)
    controller.step(1)
    controller.step(1)
    controller.step(1)
    controller.step_by_size(1)
    controller.shutdown()

    app.quit()
