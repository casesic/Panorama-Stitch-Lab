"""Background frame prefetch provider."""

from __future__ import annotations

import queue
import threading
import time

from config import PREFETCH_BACKWARD, PREFETCH_DROP_DISTANCE, PREFETCH_FORWARD
from decoder import VideoDecoder
from frame_cache import FrameCache


class FrameProvider:
    """Prefetch frames around the current index using a dedicated decoder thread."""

    def __init__(self, video_path: str, cache: FrameCache) -> None:
        """Initialize a provider with its own decoder instance.

        Args:
            video_path: Path to the video file to decode.
            cache: Shared frame cache filled by this provider.
        """
        self._video_path = video_path
        self._cache = cache
        self._decoder = VideoDecoder()
        self._decoder.open(video_path)
        self._queue: queue.Queue[int | None] = queue.Queue()
        self._current_idx: int = 0
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background prefetch worker thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker thread and close the dedicated decoder."""
        self._running = False
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._decoder.close()

    def request_prefetch_around(self, idx: int) -> None:
        """Queue prefetch work around a center frame index.

        Args:
            idx: Center frame index for the prefetch window.
        """
        with self._lock:
            self._current_idx = idx
        self._enqueue_window(idx)

    def _enqueue_window(self, center: int) -> None:
        """Enqueue a frame window around the given center index."""
        total_frames = self._decoder.get_info().total_frames
        start = max(0, center - PREFETCH_BACKWARD)
        end = min(total_frames - 1, center + PREFETCH_FORWARD)

        ordered_indices: list[int] = [center]
        ordered_indices.extend(range(center + 1, end + 1))
        ordered_indices.extend(range(center - 1, start - 1, -1))

        for idx in ordered_indices:
            if self._cache.contains(idx):
                continue
            self._queue.put(idx)

    def _worker_loop(self) -> None:
        """Run the background decode loop until a sentinel is received."""
        last_decoded_idx: int | None = None

        while self._running:
            target = self._queue.get()
            if target is None:
                break

            with self._lock:
                current = self._current_idx

            if abs(target - current) > PREFETCH_DROP_DISTANCE:
                continue
            if self._cache.contains(target):
                continue

            try:
                if last_decoded_idx is not None and target == last_decoded_idx + 1:
                    decoded_idx, frame = self._decoder.read_next()
                else:
                    decoded_idx, frame = self._decoder.seek_and_read(target)

                self._cache.put(decoded_idx, frame)
                last_decoded_idx = decoded_idx
            except Exception as exc:
                last_decoded_idx = None
                print(f"[Provider] decode {target} failed: {exc}")


if __name__ == "__main__":
    video_path = input("Enter a video path: ").strip()

    cache = FrameCache()
    provider = FrameProvider(video_path, cache)
    provider.start()

    cache.set_current(100)
    provider.request_prefetch_around(100)
    time.sleep(2.0)

    info = provider._decoder.get_info()
    first_start = max(0, 100 - PREFETCH_BACKWARD)
    first_end = min(info.total_frames - 1, 100 + PREFETCH_FORWARD)
    first_hits = [idx for idx in range(first_start, first_end + 1) if cache.contains(idx)]
    print("after 100 stats:", cache.stats())
    print("cached near 100:", first_hits)

    cache.set_current(500)
    provider.request_prefetch_around(500)
    time.sleep(2.0)

    second_start = max(0, 500 - PREFETCH_BACKWARD)
    second_end = min(info.total_frames - 1, 500 + PREFETCH_FORWARD)
    second_hits = [idx for idx in range(second_start, second_end + 1) if cache.contains(idx)]
    print("after 500 stats:", cache.stats())
    print("cached near 500:", second_hits)

    provider.stop()
    print("Provider stopped.")
