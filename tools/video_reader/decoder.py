"""Video decoding utilities built on top of PyAV."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator

import av
import numpy as np

from tools.video_reader.utils import frame_to_pts, pts_to_frame


@dataclass(slots=True)
class VideoInfo:
    """Basic metadata for an opened video."""

    path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_seconds: float


class VideoDecoder:
    """PyAV-based video decoder with sequential and seek-based read paths."""

    def __init__(self) -> None:
        """Initialize an empty decoder without opening a file."""
        self._container: av.container.InputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._iterator: Iterator[av.video.frame.VideoFrame] | None = None
        self._current_frame_idx: int = -1
        self._info: VideoInfo | None = None

    def open(self, path: str) -> VideoInfo:
        """Open a video file and return its metadata.

        Args:
            path: Video file path.

        Returns:
            Parsed video metadata for the first video stream.
        """
        self.close()

        container = av.open(path)
        stream = container.streams.video[0]

        fps_value = stream.average_rate or stream.guessed_rate
        if fps_value is None:
            raise ValueError("Unable to determine video FPS")
        fps_fraction = Fraction(fps_value)
        fps = float(fps_fraction)

        if stream.duration is not None:
            duration_seconds = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_seconds = float(container.duration / av.time_base)
        else:
            raise ValueError("Unable to determine video duration")

        total_frames = int(stream.frames)
        if total_frames == 0:
            total_frames = int(round(duration_seconds * fps))

        info = VideoInfo(
            path=path,
            fps=fps,
            total_frames=total_frames,
            width=stream.width,
            height=stream.height,
            duration_seconds=duration_seconds,
        )

        self._container = container
        self._stream = stream
        self._iterator = self._container.decode(self._stream)
        self._current_frame_idx = -1
        self._info = info
        return info

    def read_next(self) -> tuple[int, np.ndarray]:
        """Decode and return the next frame from the current iterator.

        Returns:
            A tuple of `(frame_idx, frame_rgb)` where `frame_rgb` is an RGB ndarray.

        Raises:
            RuntimeError: If no video is open.
            IndexError: If the stream is exhausted.
        """
        if self._iterator is None:
            raise RuntimeError("Video is not open")

        try:
            frame = next(self._iterator)
        except StopIteration as exc:
            raise IndexError("No more frames available") from exc

        frame_rgb = frame.to_ndarray(format="rgb24")
        self._current_frame_idx += 1
        return self._current_frame_idx, frame_rgb

    def seek_and_read(self, target_idx: int) -> tuple[int, np.ndarray]:
        """Seek close to a target frame index and decode until it is reached.

        Args:
            target_idx: Desired frame index.

        Returns:
            A tuple of `(frame_idx, frame_rgb)`. For VFR videos, `frame_idx` may be
            greater than `target_idx`.

        Raises:
            RuntimeError: If no video is open.
            IndexError: If the target frame cannot be decoded.
        """
        if self._container is None or self._stream is None or self._info is None:
            raise RuntimeError("Video is not open")
        if target_idx < 0:
            raise ValueError("target_idx must be non-negative")

        fps_fraction = Fraction(self._stream.average_rate or self._stream.guessed_rate)
        time_base = Fraction(self._stream.time_base)
        start_pts = self._stream.start_time or 0
        target_pts = frame_to_pts(target_idx, fps_fraction, time_base) + start_pts

        self._container.seek(
            target_pts,
            stream=self._stream,
            any_frame=False,
            backward=True,
        )
        self._iterator = self._container.decode(self._stream)

        while True:
            try:
                frame = next(self._iterator)
            except StopIteration as exc:
                raise IndexError(f"Unable to decode frame at index {target_idx}") from exc

            if frame.pts is None:
                continue

            frame_idx = pts_to_frame(
                frame.pts,
                fps=fps_fraction,
                time_base=time_base,
                start_pts=start_pts,
            )
            if frame_idx >= target_idx:
                self._current_frame_idx = frame_idx
                frame_rgb = frame.to_ndarray(format="rgb24")
                return frame_idx, frame_rgb

    def get_info(self) -> VideoInfo:
        """Return metadata for the currently opened video.

        Returns:
            Metadata for the active video.
        """
        if self._info is None:
            raise RuntimeError("Video is not open")
        return self._info

    def close(self) -> None:
        """Close the current video and clear decoder state."""
        if self._container is not None:
            self._container.close()

        self._container = None
        self._stream = None
        self._iterator = None
        self._current_frame_idx = -1
        self._info = None


if __name__ == "__main__":
    video_path = input("Enter a video path: ").strip()

    decoder = VideoDecoder()
    info = decoder.open(video_path)
    print("VideoInfo:", info)

    for _ in range(5):
        frame_idx, frame_rgb = decoder.read_next()
        print("read_next:", frame_idx, frame_rgb.shape)

    frame_idx, frame_rgb = decoder.seek_and_read(100)
    print("seek_and_read(100):", frame_idx, frame_rgb.shape)

    frame_idx, frame_rgb = decoder.seek_and_read(50)
    print("seek_and_read(50):", frame_idx, frame_rgb.shape)

    frame_idx, frame_rgb = decoder.seek_and_read(0)
    print("seek_and_read(0):", frame_idx, frame_rgb.shape)

    decoder.close()
    print("Decoder closed.")
