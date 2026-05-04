from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batch_utils import parse_frame_idx
from lib_pano.pipeline.invalid_io import load_invalid_frames, load_problem_frames

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_images(image_dir: Path) -> dict[int, Path]:
    images: dict[int, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        frame_idx = parse_frame_idx(path)
        if frame_idx is not None:
            images[frame_idx] = path
    return images


def resolve_review_dirs(selected_dir: Path) -> tuple[Path, Path, Path] | None:
    if (selected_dir / "cam2" / "vis").is_dir() and (selected_dir / "cam4" / "vis").is_dir():
        return selected_dir / "cam2" / "vis", selected_dir / "cam4" / "vis", selected_dir

    if (selected_dir / "cam2").is_dir() and (selected_dir / "cam4").is_dir():
        vis_dir = selected_dir
        return vis_dir / "cam2", vis_dir / "cam4", vis_dir

    if (selected_dir / "vis" / "cam2").is_dir() and (selected_dir / "vis" / "cam4").is_dir():
        vis_dir = selected_dir / "vis"
        return vis_dir / "cam2", vis_dir / "cam4", vis_dir

    if selected_dir.name == "cam2":
        cam2_dir = selected_dir
        cam4_dir = selected_dir.parent / "cam4"
        if cam4_dir.is_dir():
            return cam2_dir, cam4_dir, selected_dir.parent

    if selected_dir.name == "cam4":
        cam4_dir = selected_dir
        cam2_dir = selected_dir.parent / "cam2"
        if cam2_dir.is_dir():
            return cam2_dir, cam4_dir, selected_dir.parent

    return None


def load_existing_marks(json_path: Path) -> tuple[set[int], set[int]]:
    if not json_path.exists():
        return set(), set()

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and ("invalid_frames" in data or "problem_frames" in data):
        invalid = load_invalid_frames(json_path, missing_ok=True, recurse_all=False)
        problem = load_problem_frames(json_path, recurse_all=False)
        problem -= invalid
        return invalid, problem

    return load_invalid_frames(json_path, missing_ok=True, recurse_all=False), set()


def make_missing_image(text: str) -> np.ndarray:
    image = np.zeros((720, 480, 3), dtype=np.uint8)
    cv2.putText(image, text, (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (220, 220, 220), 3, cv2.LINE_AA)
    return image


def load_rgb_image(path: Path | None, missing_text: str) -> np.ndarray:
    if path is None:
        return make_missing_image(missing_text)

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return make_missing_image(missing_text)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_overlay(image_rgb: np.ndarray, text: str | None, color: tuple[int, int, int] | None) -> np.ndarray:
    image = image_rgb.copy()
    if text is None or color is None:
        return image
    cv2.putText(image, text, (32, 72), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5, cv2.LINE_AA)
    return image


class ImageLabel(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("QLabel { background: #111; color: #ccc; }")
        self._title = title
        self._pixmap: QPixmap | None = None

    def set_image(self, image_rgb: np.ndarray) -> None:
        image_rgb = np.ascontiguousarray(image_rgb)
        h, w, ch = image_rgb.shape
        qimage = QImage(image_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self._refresh()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap is None:
            self.setText(self._title)
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


class InvalidImageReviewer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Review Invalid Images")

        selected = QFileDialog.getExistingDirectory(self, "Select vis/result/camera directory")
        if not selected:
            raise SystemExit(0)

        selected_dir = Path(selected)
        resolved = resolve_review_dirs(selected_dir)
        if resolved is None:
            single_images = scan_images(selected_dir)
            if not single_images:
                QMessageBox.critical(self, "Invalid Folder", "Failed to find cam2/cam4 image directories or reviewable images.")
                raise SystemExit(1)

            self.review_mode = "single"
            self.image_dir = selected_dir
            self.json_path = self.image_dir / "invalid.json"
            self.single_images = single_images
            self.frame_indices = sorted(self.single_images)
        else:
            self.review_mode = "dual"
            self.cam2_dir, self.cam4_dir, self.vis_dir = resolved
            self.json_path = self.vis_dir.parent / "invalid.json"
            self.cam2_images = scan_images(self.cam2_dir)
            self.cam4_images = scan_images(self.cam4_dir)
            self.frame_indices = sorted(set(self.cam2_images) | set(self.cam4_images))
        if not self.frame_indices:
            QMessageBox.critical(self, "No Images", "No supported images were found.")
            raise SystemExit(1)

        self.invalid_frames, self.problem_frames = load_existing_marks(self.json_path)
        self.current_index = 0

        self.status_bar = QStatusBar()

        image_layout = QHBoxLayout()
        if self.review_mode == "single":
            self.single_label = ImageLabel("combined")
            image_layout.addWidget(self.single_label, 1)
        else:
            self.cam2_label = ImageLabel("cam2")
            self.cam4_label = ImageLabel("cam4")
            image_layout.addWidget(self.cam2_label, 1)
            image_layout.addWidget(self.cam4_label, 1)

        layout = QVBoxLayout()
        layout.addLayout(image_layout, 1)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)

        self._setup_shortcuts()
        self.refresh()

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Right"), self, activated=lambda: self.step(1))
        QShortcut(QKeySequence("Left"), self, activated=lambda: self.step(-1))
        QShortcut(QKeySequence("Shift+Right"), self, activated=lambda: self.step(10))
        QShortcut(QKeySequence("Shift+Left"), self, activated=lambda: self.step(-10))
        QShortcut(QKeySequence("Home"), self, activated=lambda: self.go_to(0))
        QShortcut(QKeySequence("End"), self, activated=lambda: self.go_to(len(self.frame_indices) - 1))
        QShortcut(QKeySequence("I"), self, activated=self.toggle_invalid)
        QShortcut(QKeySequence("P"), self, activated=self.toggle_problem)
        QShortcut(QKeySequence("S"), self, activated=self.save_json)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_json)

    def current_frame(self) -> int:
        return self.frame_indices[self.current_index]

    def current_state(self) -> str:
        frame_idx = self.current_frame()
        if frame_idx in self.invalid_frames:
            return "invalid"
        if frame_idx in self.problem_frames:
            return "problem"
        return "good"

    def overlay(self) -> tuple[str | None, tuple[int, int, int] | None]:
        state = self.current_state()
        if state == "invalid":
            return "INVALID", (255, 0, 0)
        if state == "problem":
            return "PROBLEM", (255, 255, 0)
        return None, None

    def refresh(self) -> None:
        frame_idx = self.current_frame()
        overlay_text, overlay_color = self.overlay()

        if self.review_mode == "single":
            image_path = self.single_images.get(frame_idx)
            image_name = image_path.name if image_path is not None else "missing"
            image = load_rgb_image(image_path, "missing image")
            self.single_label.set_image(draw_overlay(image, overlay_text, overlay_color))

            status = (
                f"frame {frame_idx:05d} | "
                f"image {self.current_index + 1}/{len(self.frame_indices)} | "
                f"invalid={len(self.invalid_frames)} problem={len(self.problem_frames)} | "
                f"current={self.current_state()} | "
                f"image={image_name}"
            )
            self.status_bar.showMessage(status)
            return

        cam2_path = self.cam2_images.get(frame_idx)
        cam4_path = self.cam4_images.get(frame_idx)
        cam2_name = cam2_path.name if cam2_path is not None else "missing"
        cam4_name = cam4_path.name if cam4_path is not None else "missing"

        cam2_image = load_rgb_image(cam2_path, "missing cam2")
        cam4_image = load_rgb_image(cam4_path, "missing cam4")
        self.cam2_label.set_image(draw_overlay(cam2_image, overlay_text, overlay_color))
        self.cam4_label.set_image(draw_overlay(cam4_image, overlay_text, overlay_color))

        status = (
            f"frame {frame_idx:05d} | "
            f"image {self.current_index + 1}/{len(self.frame_indices)} | "
            f"invalid={len(self.invalid_frames)} problem={len(self.problem_frames)} | "
            f"current={self.current_state()} | "
            f"cam2={cam2_name} | cam4={cam4_name}"
        )
        self.status_bar.showMessage(status)

    def step(self, delta: int) -> None:
        self.go_to(self.current_index + delta)

    def go_to(self, index: int) -> None:
        self.current_index = max(0, min(len(self.frame_indices) - 1, index))
        self.refresh()

    def toggle_invalid(self) -> None:
        frame_idx = self.current_frame()
        if frame_idx in self.invalid_frames:
            self.invalid_frames.remove(frame_idx)
        else:
            self.invalid_frames.add(frame_idx)
            self.problem_frames.discard(frame_idx)
        self.refresh()

    def toggle_problem(self) -> None:
        frame_idx = self.current_frame()
        if frame_idx in self.problem_frames:
            self.problem_frames.remove(frame_idx)
        else:
            self.problem_frames.add(frame_idx)
            self.invalid_frames.discard(frame_idx)
        self.refresh()

    def save_json(self) -> None:
        self.problem_frames -= self.invalid_frames
        if self.review_mode == "single":
            data = {
                "invalid_frames": sorted(self.invalid_frames),
                "problem_frames": sorted(self.problem_frames),
                "meta": {
                    "image_dir": str(self.image_dir),
                    "review_mode": "single",
                    "total_invalid": len(self.invalid_frames),
                    "total_problem": len(self.problem_frames),
                },
            }
        else:
            data = {
                "invalid_frames": sorted(self.invalid_frames),
                "problem_frames": sorted(self.problem_frames),
                "meta": {
                    "cam2_dir": str(self.cam2_dir),
                    "cam4_dir": str(self.cam4_dir),
                    "total_invalid": len(self.invalid_frames),
                    "total_problem": len(self.problem_frames),
                },
            }
        with self.json_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        self.status_bar.showMessage(f"saved to {self.json_path}")

    def closeEvent(self, event) -> None:
        self.save_json()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = InvalidImageReviewer()
    window.resize(1600, 800)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
