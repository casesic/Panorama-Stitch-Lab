from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QCloseEvent, QKeyEvent, QKeySequence
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from tools.video_reader.controller import PlaybackController
from tools.video_reader.decoder import VideoDecoder
from tools.video_reader.frame_cache import FrameCache
from tools.video_reader.frame_provider import FrameProvider
from tools.video_reader.ui_canvas import VideoCanvas


COCO_SKELETON = np.array(
    [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [5, 6],
        [5, 7],
        [7, 9],
        [6, 8],
        [8, 10],
        [5, 11],
        [6, 12],
        [11, 12],
        [11, 13],
        [13, 15],
        [12, 14],
        [14, 16],
    ],
    dtype=np.int32,
)

def draw_annotations(frame_rgb: np.ndarray, candidates: list[dict[str, Any]]) -> np.ndarray:
    """Draw bbox, keypoints, skeleton, and summary text for all candidates on a frame."""
    frame = np.asarray(frame_rgb)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected frame_rgb shape (H, W, 3), got {frame.shape}")

    annotated = frame.copy()
    for candidate in candidates:
        bbox = np.rint(np.asarray(candidate["fisheye_bbox"], dtype=np.float32)).astype(np.int32)
        keypoints = np.rint(np.asarray(candidate["keypoints_fisheye"], dtype=np.float32)).astype(
            np.int32
        )
        scores = np.asarray(candidate["keypoint_scores"], dtype=np.float32)

        x1, y1, x2, y2 = bbox.tolist()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

        for joint_a, joint_b in COCO_SKELETON:
            if scores[joint_a] > 0.3 and scores[joint_b] > 0.3:
                pt_a = tuple(keypoints[joint_a].tolist())
                pt_b = tuple(keypoints[joint_b].tolist())
                cv2.line(annotated, pt_a, pt_b, (0, 255, 255), 2)

        for idx, point in enumerate(keypoints):
            center = tuple(point.tolist())
            if scores[idx] > 0.3:
                cv2.circle(annotated, center, 6, (255, 255, 0), -1)
            else:
                cv2.circle(annotated, center, 4, (128, 128, 128), -1)

        text_lines = [
            (
                f"theta_mean={float(candidate['keypoints_mean_theta']):.1f} deg  "
                f"theta_max={float(candidate['keypoints_max_theta']):.1f} deg"
            ),
            (
                f"vis_kps={int(candidate['visible_keypoints_count'])}  "
                f"completeness={float(candidate['bbox_completeness']):.2f}"
            ),
            f"view={candidate['source_view']}",
        ]
        text_x = max(10, x1)
        base_y = max(24, y1 - 12)
        line_height = 22
        if base_y - (len(text_lines) - 1) * line_height < 18:
            base_y = min(annotated.shape[0] - 10, y1 + 24)

        for line_idx, text in enumerate(text_lines):
            y = base_y + line_idx * line_height
            cv2.putText(
                annotated,
                text,
                (text_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return annotated


class ReviewCanvas(VideoCanvas):
    """Video canvas that can optionally overlay candidate annotations."""

    @staticmethod
    def _downsample_for_display(frame: np.ndarray) -> np.ndarray:
        """Downsample large frames before handing them to the base canvas."""
        height, width = frame.shape[:2]
        longest_edge = max(height, width)
        if longest_edge <= 960:
            return frame

        scale = 960.0 / float(longest_edge)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def show_frame(
        self,
        frame_idx: int,
        frame: np.ndarray,
        candidates: list[dict[str, Any]] | None = None,
        partner_frame: np.ndarray | None = None,
    ) -> None:
        """Display a frame with candidate overlays when provided."""
        if candidates is not None:
            frame = draw_annotations(frame, candidates)
        main_frame = self._downsample_for_display(frame)

        if partner_frame is not None:
            partner = self._downsample_for_display(partner_frame)
            h_main, h_partner = main_frame.shape[0], partner.shape[0]
            if h_main != h_partner:
                partner = cv2.resize(
                    partner,
                    (int(partner.shape[1] * h_main / h_partner), h_main),
                    interpolation=cv2.INTER_AREA,
                )
            combined = np.concatenate([main_frame, partner], axis=1)
        else:
            combined = main_frame

        super().show_frame(frame_idx, combined)


class ReviewController(QObject):
    """Manage candidate navigation and review results."""

    candidate_changed = Signal(int, int)
    status_message = Signal(str)

    def __init__(self, canvas: ReviewCanvas, parent: QObject | None = None) -> None:
        """Initialize the reviewer controller."""
        super().__init__(parent)
        self._candidates: list[dict[str, Any]] = []
        self._current_candidate_idx: int = 0
        self._results: dict[tuple[str, int], str] = {}
        self._playback = PlaybackController(parent)
        self._partner_playback = PlaybackController(parent)
        self._canvas = canvas
        self._current_video_path: str | None = None
        self._current_partner_path: str | None = None

        # Keep explicit references to the shared video_reader building blocks.
        self._video_decoder_cls = VideoDecoder
        self._frame_cache_cls = FrameCache
        self._frame_provider_cls = FrameProvider

    @staticmethod
    def _get_partner_path(video_path: str) -> str | None:
        """Derive the partner camera path (.0.mp4 <-> .1.mp4)."""
        if video_path.endswith(".0.mp4"):
            partner = video_path[:-6] + ".1.mp4"
        elif video_path.endswith(".1.mp4"):
            partner = video_path[:-6] + ".0.mp4"
        else:
            return None
        return partner if Path(partner).exists() else None

    def load_candidates(self, candidates_json_path: str) -> None:
        """Load candidates.json, sort entries, and jump to the first candidate."""
        path = Path(candidates_json_path)
        with path.open("r", encoding="utf-8") as file:
            raw_candidates = json.load(file)

        if not isinstance(raw_candidates, list):
            raise ValueError("candidates.json must contain a list")

        grouped_candidates: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for candidate in raw_candidates:
            key = (str(candidate["video_path"]), int(candidate["frame_idx"]))
            grouped_candidates.setdefault(key, []).append(candidate)

        self._candidates = [
            {
                "video_path": video_path,
                "frame_idx": frame_idx,
                "candidates": grouped_candidates[(video_path, frame_idx)],
            }
            for video_path, frame_idx in sorted(grouped_candidates)
        ]
        self._results.clear()
        self._current_candidate_idx = 0

        if not self._candidates:
            self.status_message.emit("No candidates found")
            self.candidate_changed.emit(0, 0)
            return

        self._goto_candidate(0)

    def _goto_candidate(self, idx: int) -> None:
        """Jump to one candidate, loading a new video if required."""
        if not self._candidates:
            return
        target_idx = max(0, min(idx, len(self._candidates) - 1))
        frame_unit = self._candidates[target_idx]
        video_path = str(frame_unit["video_path"])
        frame_idx = int(frame_unit["frame_idx"])

        if video_path != self._current_video_path:
            self._playback.load_video(video_path)
            self._current_video_path = video_path

        self._playback.goto(frame_idx)
        frame = self._playback._last_frame
        if frame is None:
            self.status_message.emit(f"Unable to display frame: {video_path} @ {frame_idx}")
            return

        partner_path = self._get_partner_path(video_path)
        partner_frame = None
        if partner_path:
            if partner_path != self._current_partner_path:
                self._partner_playback.load_video(partner_path)
                self._current_partner_path = partner_path
            self._partner_playback.goto(frame_idx)
            partner_frame = self._partner_playback._last_frame
        else:
            self._current_partner_path = None

        self._current_candidate_idx = target_idx
        self._canvas.show_frame(frame_idx, frame, list(frame_unit["candidates"]), partner_frame)
        self.candidate_changed.emit(target_idx, len(self._candidates))

    def next_candidate(self) -> None:
        """Move to the next candidate in order."""
        if not self._candidates:
            return
        if self._current_candidate_idx >= len(self._candidates) - 1:
            self.status_message.emit("Already at the last candidate")
            return
        self._goto_candidate(self._current_candidate_idx + 1)

    def prev_candidate(self) -> None:
        """Move to the previous candidate in order."""
        if not self._candidates:
            return
        if self._current_candidate_idx <= 0:
            self.status_message.emit("Already at the first candidate")
            return
        self._goto_candidate(self._current_candidate_idx - 1)

    def record_and_next(self, result: str) -> None:
        """Record a review decision and advance to the next undecided candidate."""
        if result not in {"keep", "discard", "skip"}:
            raise ValueError(f"Invalid review result: {result}")
        if not self._candidates:
            return

        frame_unit = self._candidates[self._current_candidate_idx]
        current_key = (str(frame_unit["video_path"]), int(frame_unit["frame_idx"]))
        self._results[current_key] = result
        if result in {"keep", "skip"} and frame_unit["candidates"]:
            self._insert_following_frames(self._current_candidate_idx)
        self.candidate_changed.emit(self._current_candidate_idx, len(self._candidates))

        for next_idx in range(self._current_candidate_idx + 1, len(self._candidates)):
            item = self._candidates[next_idx]
            next_key = (str(item["video_path"]), int(item["frame_idx"]))
            if next_key not in self._results:
                self._goto_candidate(next_idx)
                return

        self.status_message.emit("All candidates have been reviewed")

    def _insert_following_frames(self, candidate_idx: int) -> None:
        """Insert subsequent frames spanning roughly half a second after a kept candidate."""
        frame_unit = self._candidates[candidate_idx]
        video_path = str(frame_unit["video_path"])
        frame_idx = int(frame_unit["frame_idx"])
        total_frames = self._get_total_frames()
        fps = self._get_video_fps()
        half_second_frames = max(1, round(fps / 2.0)) if fps else 12
        expand_count = half_second_frames - 1

        existing_frames = {
            (str(item["video_path"]), int(item["frame_idx"])) for item in self._candidates
        }
        new_candidates: list[dict[str, Any]] = []
        for offset in range(1, expand_count + 1):
            next_frame_idx = frame_idx + offset
            if total_frames is not None and next_frame_idx >= total_frames:
                break

            key = (video_path, next_frame_idx)
            if key in existing_frames:
                continue

            existing_frames.add(key)
            new_candidates.append(
                {
                    "video_path": video_path,
                    "frame_idx": next_frame_idx,
                    "candidates": [],
                }
            )

        if not new_candidates:
            return

        insert_at = candidate_idx + 1
        self._candidates[insert_at:insert_at] = new_candidates

    def _get_total_frames(self) -> int | None:
        """Return the loaded video's total frame count when available."""
        video_info = getattr(self._playback, "_video_info", None)
        total_frames = getattr(video_info, "total_frames", None)
        if total_frames is None:
            return None
        try:
            return int(total_frames)
        except (TypeError, ValueError):
            return None

    def _get_video_fps(self) -> float | None:
        """Return the loaded video's FPS when available."""
        video_info = getattr(self._playback, "_video_info", None)
        fps = getattr(video_info, "fps", None)
        if fps is None:
            return None
        try:
            return float(fps)
        except (TypeError, ValueError):
            return None

    def save_results(self, output_path: str) -> None:
        """Write kept frames to reviewed.json."""
        reviewed: list[dict[str, Any]] = []
        for frame_unit in self._candidates:
            key = (str(frame_unit["video_path"]), int(frame_unit["frame_idx"]))
            if self._results.get(key) != "keep":
                continue
            reviewed.append(
                {
                    "video_path": frame_unit["video_path"],
                    "frame_idx": frame_unit["frame_idx"],
                }
            )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(reviewed, file, ensure_ascii=False, indent=2)
        self.status_message.emit(f"Saved to {path}")

    def shutdown(self) -> None:
        """Release playback resources."""
        self._playback.shutdown()
        self._partner_playback.shutdown()

    def get_total_candidates(self) -> int:
        """Return the total number of loaded candidates."""
        return len(self._candidates)

    def get_done_count(self) -> int:
        """Return the number of candidates with a recorded decision."""
        return len(self._results)

    def get_current_candidate_index(self) -> int:
        """Return the current candidate index."""
        return self._current_candidate_idx

    def get_current_frame_idx(self) -> int | None:
        """Return the current frame index for the selected frame unit."""
        if not self._candidates:
            return None
        return int(self._candidates[self._current_candidate_idx]["frame_idx"])


class ReviewWindow(QMainWindow):
    """Main window for reviewing fisheye extraction candidates."""

    def __init__(self, output_path: str, parent: QWidget | None = None) -> None:
        """Initialize the review window and connect controller signals."""
        super().__init__(parent)
        self.setWindowTitle("Fisheye Candidate Reviewer")
        self.resize(1400, 1100)

        self._output_path = output_path
        self._canvas = ReviewCanvas(self)
        self._status_label = QLabel(self)
        self._status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._status_label.setStyleSheet("padding: 8px;")

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._status_label, stretch=0)
        self.setCentralWidget(container)

        self._controller = ReviewController(self._canvas, self)
        self._controller.candidate_changed.connect(self._update_status_bar)
        self._controller.status_message.connect(self._set_status_message)
        self._last_message = ""
        self._update_status_bar(0, 0)

    def load_candidates(self, candidates_json_path: str) -> None:
        """Load candidate entries into the controller."""
        self._controller.load_candidates(candidates_json_path)

    def _update_status_bar(self, current_idx: int, total: int) -> None:
        """Refresh the persistent progress text in the status area."""
        done = self._controller.get_done_count()
        frame_idx = self._controller.get_current_frame_idx()
        frame_str = f"frame={frame_idx}" if frame_idx is not None else "frame=?"
        if total <= 0:
            base = (
                f"Candidate 0/0  |  {frame_str}  |  Reviewed 0  |  "
                "Left: prev  Right: next  Y: keep  N: discard  Space: skip  Ctrl+S: save"
            )
        else:
            base = (
                f"Candidate {current_idx + 1}/{total}  |  {frame_str}  |  Reviewed {done}  |  "
                "Left: prev  Right: next  Y: keep  N: discard  Space: skip  Ctrl+S: save"
            )
        if self._last_message:
            self._status_label.setText(f"{base}  |  {self._last_message}")
        else:
            self._status_label.setText(base)

    def _set_status_message(self, message: str) -> None:
        """Show a transient status message alongside progress information."""
        self._last_message = message
        self._update_status_bar(
            self._controller.get_current_candidate_index(),
            self._controller.get_total_candidates(),
        )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle reviewer keyboard shortcuts."""
        if event.matches(QKeySequence.Save):
            self._controller.save_results(self._output_path)
            event.accept()
            return

        key = event.key()
        if key == Qt.Key_Y:
            self._controller.record_and_next("keep")
            event.accept()
            return
        if key == Qt.Key_N:
            self._controller.record_and_next("discard")
            event.accept()
            return
        if key == Qt.Key_Space:
            self._controller.record_and_next("skip")
            event.accept()
            return
        if key == Qt.Key_Left:
            self._controller.prev_candidate()
            event.accept()
            return
        if key == Qt.Key_Right:
            self._controller.next_candidate()
            event.accept()
            return

        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Persist review progress and release playback resources on close."""
        self._controller.save_results(self._output_path)
        self._controller.shutdown()
        super().closeEvent(event)


def run_reviewer(candidates_json_path: str, output_path: str) -> None:
    """Launch the interactive reviewer UI."""
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)

    window = ReviewWindow(output_path)
    window.load_candidates(candidates_json_path)
    window.show()

    if owns_app:
        app.exec()
