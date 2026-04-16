"""Thread-safe frame cache for decoded video frames."""

from __future__ import annotations

from collections import OrderedDict
import threading

import numpy as np

from tools.video_reader.config import CACHE_MAX_SIZE


class FrameCache:
    """Cache decoded frames with distance-based eviction around the current frame."""

    def __init__(self, max_size: int = CACHE_MAX_SIZE) -> None:
        """Initialize an empty frame cache.

        Args:
            max_size: Maximum number of frames kept in memory.
        """
        self.max_size = max_size
        self._data: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._current_idx: int = 0
        self._hits: int = 0
        self._misses: int = 0

    def get(self, idx: int) -> np.ndarray | None:
        """Return a cached frame if it exists.

        Args:
            idx: Frame index to query.

        Returns:
            Cached frame array if present, otherwise ``None``.
        """
        with self._lock:
            frame = self._data.get(idx)
            if frame is None:
                self._misses += 1
                return None

            self._hits += 1
            self._data.move_to_end(idx)
            return frame

    def put(self, idx: int, frame: np.ndarray) -> None:
        """Insert or update a cached frame.

        Args:
            idx: Frame index to store.
            frame: RGB frame data.
        """
        with self._lock:
            self._data[idx] = frame
            self._data.move_to_end(idx)
            if len(self._data) > self.max_size:
                self._evict_farthest()

    def contains(self, idx: int) -> bool:
        """Check whether a frame index exists in cache.

        Args:
            idx: Frame index to query.

        Returns:
            ``True`` if the frame is cached, else ``False``.
        """
        with self._lock:
            return idx in self._data

    def set_current(self, idx: int) -> None:
        """Update the current frame index used by the eviction policy.

        Args:
            idx: Current viewer frame index.
        """
        with self._lock:
            self._current_idx = idx

    def clear(self) -> None:
        """Remove all cached frames and reset cache contents."""
        with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, float | int]:
        """Return cache hit and size statistics.

        Returns:
            Dictionary with cache size, hit count, miss count, and hit rate.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = 0.0 if total == 0 else self._hits / total
            return {
                "size": len(self._data),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def _evict_farthest(self) -> None:
        """Evict the cached frame farthest from the current frame index."""
        farthest_idx = max(
            self._data.keys(),
            key=lambda key: abs(key - self._current_idx),
        )
        self._data.pop(farthest_idx)


if __name__ == "__main__":
    cache = FrameCache(max_size=5)
    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    cache.set_current(100)
    for frame_idx in range(95, 106):
        cache.put(frame_idx, fake_frame)

    remaining = sorted(cache._data.keys())
    print("remaining:", remaining)

    hit = cache.get(100)
    miss = cache.get(200)
    print("get(100) is not None:", hit is not None)
    print("get(200) is None:", miss is None)
    print("stats:", cache.stats())
