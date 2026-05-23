from __future__ import annotations

import bisect
from typing import Any, Sequence, Tuple


class TimestampIterator:
    """Nearest timestamp lookup with linear interpolation for position."""

    def __init__(self, data: Sequence[Tuple[Any, ...]]):
        self.data = list(data)
        self.timestamps = [abs(float(item[0])) for item in self.data]

    def find_nearest(self, target_timestamp: float) -> Tuple[Any, ...]:
        pos = bisect.bisect_left(self.timestamps, target_timestamp)

        if pos == 0:
            return self.data[0]
        if pos == len(self.timestamps):
            return self.data[-1]

        before = self.data[pos - 1]
        after = self.data[pos]

        if float(after[0]) == target_timestamp:
            return after

        return self._create_interpolated(target_timestamp, before, after)

    def _create_interpolated(
        self,
        target_ts: float,
        point_a: Tuple[Any, ...],
        point_b: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        ts_a, ts_b = float(point_a[0]), float(point_b[0])
        if ts_b == ts_a:
            return point_a

        alpha = (target_ts - ts_a) / (ts_b - ts_a)

        interpolated = [target_ts]
        for idx, (val_a, val_b) in enumerate(zip(point_a[1:], point_b[1:])):
            if idx == 0:
                interpolated_val = val_a + alpha * (val_b - val_a)
            else:
                interpolated_val = val_b if alpha > 0.5 else val_a
            interpolated.append(interpolated_val)
        return tuple(interpolated)

