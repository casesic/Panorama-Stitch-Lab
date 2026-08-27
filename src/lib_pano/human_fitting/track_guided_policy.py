"""Frame-mode scheduling and same-frame recovery for track-guided delivery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence


class DeliveryFrameMode(str, Enum):
    FULL_DETECTION = "FULL_DETECTION"
    TRACK_GUIDED = "TRACK_GUIDED"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True)
class DeliveryDecision:
    requested_mode: DeliveryFrameMode
    actual_mode: DeliveryFrameMode
    observations: Sequence[Any]
    guided_failures: Sequence[Any]
    detector_calls: int
    densekp_calls: int


class TrackGuidedScheduler:
    """Deterministic keyframe scheduler; recovery never shifts keyframes."""

    def __init__(self, detect_interval: int, sequence_start_frame: int):
        if detect_interval < 1:
            raise ValueError("detect_interval must be >= 1")
        self.detect_interval = int(detect_interval)
        self.sequence_start_frame = int(sequence_start_frame)

    def mode(self, frame: int, active_track_count: int) -> DeliveryFrameMode:
        if active_track_count == 0:
            return DeliveryFrameMode.FULL_DETECTION
        if (int(frame) - self.sequence_start_frame) % self.detect_interval == 0:
            return DeliveryFrameMode.FULL_DETECTION
        return DeliveryFrameMode.TRACK_GUIDED


def run_delivery_with_same_frame_recovery(
    requested_mode: DeliveryFrameMode,
    active_tracks: Sequence[Any],
    full_delivery: Callable[[], tuple[Sequence[Any], dict[str, int]]],
    guided_delivery: Callable[[Sequence[Any]], tuple[Sequence[Any], Sequence[Any], dict[str, int]]],
) -> DeliveryDecision:
    """Run guided delivery and immediately recover with the full path on failure."""
    if requested_mode == DeliveryFrameMode.FULL_DETECTION:
        observations, counts = full_delivery()
        return DeliveryDecision(
            requested_mode, requested_mode, observations, (),
            int(counts.get("detector_calls", 0)),
            int(counts.get("densekp_calls", 0)),
        )
    guided, failures, counts = guided_delivery(active_tracks)
    if not failures and len(guided) == len(active_tracks):
        return DeliveryDecision(
            requested_mode, DeliveryFrameMode.TRACK_GUIDED, guided, (),
            0, int(counts.get("densekp_calls", 0)),
        )
    recovered, recovery_counts = full_delivery()
    return DeliveryDecision(
        requested_mode, DeliveryFrameMode.RECOVERY, recovered, failures,
        int(recovery_counts.get("detector_calls", 0)),
        int(counts.get("densekp_calls", 0))
        + int(recovery_counts.get("densekp_calls", 0)),
    )

