"""Stateless utility helpers for time conversion and display."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PySide6.QtGui import QImage


def frame_to_pts(frame_idx: int, fps: Fraction, time_base: Fraction) -> int:
    """Convert frame index to PyAV PTS value.

    Args:
        frame_idx: Zero-based frame index.
        fps: Video frames-per-second as Fraction.
        time_base: Stream time base as Fraction.

    Returns:
        Integer PTS corresponding to the frame index.
    """
    return int(frame_idx / fps / time_base)


def pts_to_frame(pts: int, fps: Fraction, time_base: Fraction, start_pts: int = 0) -> int:
    """Convert a PTS value back to frame index.

    Args:
        pts: Current frame PTS.
        fps: Video frames-per-second as Fraction.
        time_base: Stream time base as Fraction.
        start_pts: Start PTS offset of the stream.

    Returns:
        Nearest frame index for the input PTS.
    """
    return int(round((pts - start_pts) * time_base * fps))


def ndarray_to_qimage(frame_rgb: np.ndarray) -> QImage:
    """Convert an RGB numpy array into a detached QImage.

    Args:
        frame_rgb: RGB image array with shape (H, W, 3), dtype uint8.

    Returns:
        QImage in Format_RGB888 with copied pixel data.
    """
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("frame_rgb must have shape (H, W, 3)")
    if frame_rgb.dtype != np.uint8:
        raise ValueError("frame_rgb dtype must be uint8")

    from PySide6.QtGui import QImage

    h, w, _ = frame_rgb.shape
    arr = np.ascontiguousarray(frame_rgb)
    bytes_per_line = arr.strides[0]
    image = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return image.copy()


def format_timestamp(frame_idx: int, fps: float) -> str:
    """Format frame index as HH:MM:SS.mmm timestamp.

    Args:
        frame_idx: Zero-based frame index.
        fps: Frames per second.

    Returns:
        Timestamp string formatted as HH:MM:SS.mmm.
    """
    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    total_ms = int(round((frame_idx / fps) * 1000.0))
    total_seconds, ms = divmod(total_ms, 1000)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


if __name__ == "__main__":
    fps_frac = Fraction(30, 1)
    time_base = Fraction(1, 90000)
    idx = 1234

    pts = frame_to_pts(idx, fps_frac, time_base)
    ts = format_timestamp(idx, float(fps_frac))

    print(f"frame_to_pts({idx}, {fps_frac}, {time_base}) = {pts}")
    print(f"format_timestamp({idx}, {float(fps_frac)}) = {ts}")
