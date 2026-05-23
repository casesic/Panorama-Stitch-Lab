from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QKeyEvent, QMouseEvent, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox, QSizePolicy

from batch_utils import load_reviewed_frame_indices, split_frame_segments


MAX_DISPLAY_SIZE = 400
DOT_COLOR = (0, 0, 255)
DOT_RADIUS = 8
DOT_THICKNESS = -1
SEPARATOR_COLOR = (255, 255, 255)
SEPARATOR_WIDTH = 2
CAMERA_NAMES = ("cam2", "cam4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate person click points on the first frame of each reviewed segment."
    )
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json.")
    parser.add_argument(
        "--fisheye_dir",
        type=str,
        required=True,
        help="Fisheye image root directory containing cam2/ and cam4/ subdirectories.",
    )
    parser.add_argument("--output", type=str, required=True, help="Path to click_points.json.")
    return parser.parse_args()


def load_existing_output(output_path: Path) -> dict[str, dict[str, list[int]]]:
    if not output_path.exists():
        return {}

    with output_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Existing output must be a dict")

    normalized: dict[str, dict[str, list[int]]] = {}
    for camera_name, frame_map in data.items():
        if not isinstance(frame_map, dict):
            continue
        normalized[camera_name] = {}
        for frame_idx, point in frame_map.items():
            if (
                isinstance(point, list)
                and len(point) == 2
                and all(isinstance(value, (int, float)) for value in point)
            ):
                normalized[camera_name][str(frame_idx)] = [int(round(point[0])), int(round(point[1]))]
    for camera_name in CAMERA_NAMES:
        normalized.setdefault(camera_name, {})
    return normalized


def save_output(output_path: Path, all_points: dict[str, dict[str, list[int]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {camera_name: all_points.get(camera_name, {}) for camera_name in CAMERA_NAMES}
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, ensure_ascii=False, indent=2)


class ClickableImageLabel(QLabel):
    clicked = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            position = event.position()
            self.clicked.emit(position.x(), position.y())
            event.accept()
            return
        super().mousePressEvent(event)


class AnnotateClickPointsWindow(QMainWindow):
    def __init__(
        self,
        frames: list[int],
        fisheye_dir: Path,
        output_path: Path,
        existing_output: dict[str, dict[str, list[int]]],
    ) -> None:
        super().__init__()
        self.frames = frames
        self.fisheye_dir = fisheye_dir
        self.output_path = output_path
        self.all_points = existing_output
        self.current_index = 0
        self.current_frame: int | None = None
        self.current_images: dict[str, np.ndarray] = {}
        self.current_display_image: np.ndarray | None = None
        self.display_width = 0
        self.display_height = 0
        self.current_layouts: dict[str, dict[str, int]] = {}
        self.is_closing = False

        self.image_label = ClickableImageLabel()
        self.image_label.clicked.connect(self.handle_image_clicked)
        self.setCentralWidget(self.image_label)
        self.statusBar()
        self.setWindowTitle("Annotate Click Points - cam2 | cam4")

        if not self.frames:
            raise ValueError("No valid frames found to annotate")

        for camera_name in CAMERA_NAMES:
            self.all_points.setdefault(camera_name, {})

        self.load_current_frame()

    def get_image_path(self, camera_name: str, frame_idx: int) -> Path:
        return self.fisheye_dir / camera_name / f"{frame_idx:05d}.jpg"

    def get_current_point(self, camera_name: str) -> list[int] | None:
        if self.current_frame is None:
            return None
        return self.all_points[camera_name].get(str(self.current_frame))

    def handle_image_clicked(self, x_display: float, y_display: float) -> None:
        if self.current_frame is None or self.display_height <= 0:
            return

        x_int = int(np.floor(x_display))
        y_int = int(np.floor(y_display))
        if not (0 <= x_int < self.display_width and 0 <= y_int < self.display_height):
            return

        for camera_name, layout in self.current_layouts.items():
            x_offset = layout["x_offset"]
            panel_width = layout["display_width"]
            panel_height = layout["display_height"]
            if not (x_offset <= x_int < x_offset + panel_width):
                continue

            local_x = x_int - x_offset
            local_y = min(max(y_int, 0), panel_height - 1)
            image = self.current_images[camera_name]
            orig_height, orig_width = image.shape[:2]
            x_orig = int(round(local_x * orig_width / panel_width))
            y_orig = int(round(local_y * orig_height / panel_height))
            x_orig = min(max(x_orig, 0), orig_width - 1)
            y_orig = min(max(y_orig, 0), orig_height - 1)
            self.all_points[camera_name][str(self.current_frame)] = [x_orig, y_orig]
            self.refresh_view()
            self.update_status_bar()
            return

    def build_display_image(self) -> np.ndarray:
        if not self.current_images:
            raise RuntimeError("No images available for display")

        max_height = max(image.shape[0] for image in self.current_images.values())
        display_height = min(MAX_DISPLAY_SIZE, max_height)

        panels: list[tuple[str, np.ndarray]] = []
        self.current_layouts = {}
        x_offset = 0

        for idx, camera_name in enumerate(CAMERA_NAMES):
            image = self.current_images.get(camera_name)
            if image is None:
                continue

            orig_height, orig_width = image.shape[:2]
            display_width = max(1, int(round(orig_width * display_height / orig_height)))
            resized = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
            point = self.get_current_point(camera_name)
            if point is not None:
                x_point = int(round(point[0] * display_width / orig_width))
                y_point = int(round(point[1] * display_height / orig_height))
                x_point = min(max(x_point, 0), display_width - 1)
                y_point = min(max(y_point, 0), display_height - 1)
                cv2.circle(
                    resized,
                    center=(x_point, y_point),
                    radius=DOT_RADIUS,
                    color=DOT_COLOR,
                    thickness=DOT_THICKNESS,
                    lineType=cv2.LINE_AA,
                )

            self.current_layouts[camera_name] = {
                "x_offset": x_offset,
                "display_width": display_width,
                "display_height": display_height,
            }
            panels.append((camera_name, resized))
            x_offset += display_width

            remaining_cameras = CAMERA_NAMES[idx + 1 :]
            if any(self.current_images.get(other_camera) is not None for other_camera in remaining_cameras):
                separator = np.full((display_height, SEPARATOR_WIDTH, 3), SEPARATOR_COLOR, dtype=np.uint8)
                panels.append((f"{camera_name}_separator", separator))
                x_offset += SEPARATOR_WIDTH

        combined = np.concatenate([panel for _, panel in panels], axis=1)
        self.display_height, self.display_width = combined.shape[:2]
        return combined

    def render_pixmap(self) -> QPixmap:
        if self.current_display_image is None:
            raise RuntimeError("No display image available")

        image_rgb = cv2.cvtColor(self.current_display_image, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)
        qimage = QImage(
            image_rgb.data,
            image_rgb.shape[1],
            image_rgb.shape[0],
            int(image_rgb.strides[0]),
            QImage.Format.Format_RGB888,
        )
        return QPixmap.fromImage(qimage.copy())

    def refresh_view(self) -> None:
        self.current_display_image = self.build_display_image()
        pixmap = self.render_pixmap()
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(self.display_width, self.display_height)
        self.setFixedSize(self.image_label.width(), self.image_label.height() + self.statusBar().height())

    def update_status_bar(self) -> None:
        frame_text = self.current_frame if self.current_frame is not None else "-"
        status = (
            f"序列 {self.current_index + 1}/{len(self.frames)} | "
            f"frame_idx={frame_text} | "
            f"cam2 已标注 {self.count_annotated('cam2')} | "
            f"cam4 已标注 {self.count_annotated('cam4')}"
        )
        self.statusBar().showMessage(status)

    def count_annotated(self, camera_name: str) -> int:
        return sum(1 for frame_idx in self.frames if str(frame_idx) in self.all_points[camera_name])

    def load_current_frame(self) -> None:
        frame_idx = self.frames[self.current_index]
        current_images: dict[str, np.ndarray] = {}

        for camera_name in CAMERA_NAMES:
            image_path = self.get_image_path(camera_name, frame_idx)
            if not image_path.exists():
                print(f"Warning: image not found for {camera_name}, frame {frame_idx}: {image_path}")
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: failed to load image for {camera_name}, frame {frame_idx}: {image_path}")
                continue
            current_images[camera_name] = image

        if not current_images:
            raise FileNotFoundError(f"No available images for frame {frame_idx}")

        self.current_frame = frame_idx
        self.current_images = current_images
        self.refresh_view()
        self.update_status_bar()

    def go_to_index(self, new_index: int) -> None:
        new_index = min(max(new_index, 0), len(self.frames) - 1)
        if new_index == self.current_index:
            self.update_status_bar()
            return
        self.current_index = new_index
        self.load_current_frame()

    def go_next(self) -> None:
        if self.current_index >= len(self.frames) - 1:
            self.update_status_bar()
            return
        self.go_to_index(self.current_index + 1)

    def go_previous(self) -> None:
        if self.current_index <= 0:
            self.update_status_bar()
            return
        self.go_to_index(self.current_index - 1)

    def persist(self) -> None:
        save_output(self.output_path, self.all_points)
        print(f"Saved annotations to: {self.output_path}")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key.Key_S and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.persist()
            self.update_status_bar()
            event.accept()
            return

        if key in (Qt.Key.Key_Right, Qt.Key.Key_Space):
            self.go_next()
            event.accept()
            return

        if key == Qt.Key.Key_Left:
            self.go_previous()
            event.accept()
            return

        if key == Qt.Key.Key_Q:
            self.close()
            event.accept()
            return

        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.is_closing:
            event.accept()
            return

        self.is_closing = True
        try:
            self.persist()
            event.accept()
        except Exception as exc:
            self.is_closing = False
            QMessageBox.critical(self, "Save Failed", f"Failed to save annotations:\n{exc}")
            event.ignore()


def collect_target_frames(reviewed_path: Path, fisheye_dir: Path) -> list[int]:
    frame_indices = load_reviewed_frame_indices(reviewed_path)
    segments = split_frame_segments(frame_indices)
    target_frames = [segment[0] for segment in segments if segment]

    valid_frames_by_camera: dict[str, set[int]] = {camera_name: set() for camera_name in CAMERA_NAMES}
    for frame_idx in target_frames:
        for camera_name in CAMERA_NAMES:
            image_path = fisheye_dir / camera_name / f"{frame_idx:05d}.jpg"
            if image_path.exists():
                valid_frames_by_camera[camera_name].add(frame_idx)

    valid_union = valid_frames_by_camera["cam2"] | valid_frames_by_camera["cam4"]

    final_frames: list[int] = []
    for frame_idx in target_frames:
        if frame_idx in valid_union:
            final_frames.append(frame_idx)
        else:
            print(f"Warning: images not found in cam2/cam4, skipping frame {frame_idx}")
    return final_frames


def main() -> int:
    args = parse_args()
    reviewed_path = Path(args.reviewed)
    fisheye_dir = Path(args.fisheye_dir)
    output_path = Path(args.output)

    if not reviewed_path.exists():
        raise FileNotFoundError(f"reviewed.json not found: {reviewed_path}")
    if not fisheye_dir.exists():
        raise FileNotFoundError(f"fisheye_dir not found: {fisheye_dir}")

    frames = collect_target_frames(reviewed_path, fisheye_dir)
    if not frames:
        print("No valid frames found to annotate.")
        return 0

    existing_output = load_existing_output(output_path)

    app = QApplication(sys.argv)
    window = AnnotateClickPointsWindow(
        frames=frames,
        fisheye_dir=fisheye_dir,
        output_path=output_path,
        existing_output=existing_output,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


r'''
python src\script\test_function\debug035_annotate_click_points.py `
  --reviewed data_process\Input\20260311_riding\review.json `
  --fisheye_dir data_process\Output\20260311_riding\image\fisheye `
  --output data_process/Output/20260311_riding/click_points.json
'''
