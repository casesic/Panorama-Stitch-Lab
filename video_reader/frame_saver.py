"""Asynchronous frame saving helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from config import SAVE_FILENAME_FORMAT, SAVER_THREAD_COUNT


class FrameSaver:
    """Save RGB frames to disk using a background thread pool."""

    def __init__(self, thread_count: int = SAVER_THREAD_COUNT) -> None:
        """Initialize the background saver.

        Args:
            thread_count: Number of worker threads used for image writing.
        """
        self._executor = ThreadPoolExecutor(max_workers=thread_count)

    def save_async(
        self,
        frame_rgb: np.ndarray,
        frame_idx: int,
        out_dir: Path,
        on_done: Callable[[str, bool], None] | None = None,
    ) -> None:
        """Schedule saving one RGB frame as a PNG file.

        Args:
            frame_rgb: RGB image data to save.
            frame_idx: Frame index used to format the output filename.
            out_dir: Output directory for saved PNG files.
            on_done: Optional callback receiving `(path, success)`.
        """
        filename = SAVE_FILENAME_FORMAT.format(idx=frame_idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / filename
        frame_copy = frame_rgb.copy()
        self._executor.submit(self._write, frame_copy, output_path, on_done)

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the background writer pool.

        Args:
            wait: Whether to block until queued tasks finish.
        """
        self._executor.shutdown(wait=wait)

    def _write(
        self,
        frame_rgb: np.ndarray,
        path: Path,
        on_done: Callable[[str, bool], None] | None,
    ) -> None:
        """Write one RGB frame to disk and notify the optional callback."""
        try:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(str(path), frame_bgr)
            if on_done is not None:
                on_done(str(path), success)
        except Exception:
            if on_done is not None:
                on_done(str(path), False)


if __name__ == "__main__":
    output_dir = Path("video_reader_temp_output")
    width = 320
    height = 180

    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    red = np.tile(x, (height, 1))
    green = np.tile(y[:, None], (1, width))
    blue = np.full((height, width), 128, dtype=np.uint8)
    test_frame = np.stack([red, green, blue], axis=2)

    saver = FrameSaver()

    def handle_done(path: str, success: bool) -> None:
        print(f"on_done: path={path}, success={success}")

    saver.save_async(test_frame, 42, output_dir, handle_done)
    saver.shutdown()

    saved_path = output_dir / "00042.png"
    print(f"exists: {saved_path.exists()} -> {saved_path}")
