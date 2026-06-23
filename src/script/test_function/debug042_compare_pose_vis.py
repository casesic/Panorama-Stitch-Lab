from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMessageBox, QStatusBar, QVBoxLayout, QWidget


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DECISIONS = {"a_better", "b_better", "tie", "both_bad"}


def parse_frame_idx(path: Path) -> int | None:
    long_matches = re.findall(r"\d{5,}", path.stem)
    if long_matches:
        return int(long_matches[0])

    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return None

    if matches[-1] in ("2", "4") and len(matches) > 1:
        return int(matches[-2])

    return int(matches[-1])


def scan_images(image_dir: Path) -> dict[int, Path]:
    images: dict[int, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        frame_idx = parse_frame_idx(path)
        if frame_idx is not None:
            images[frame_idx] = path
    return images


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

    cv2.putText(image, text, (32, 72), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 9, cv2.LINE_AA)
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
        self._zoom_factor = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._dragging = False
        self._last_drag_pos = QPointF(0.0, 0.0)
        self.setMouseTracking(True)

    def set_image(self, image_rgb: np.ndarray, reset_view: bool = True) -> None:
        image_rgb = np.ascontiguousarray(image_rgb)
        h, w, ch = image_rgb.shape
        qimage = QImage(image_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        if reset_view:
            self.reset_view()
        else:
            self.update()

    def reset_view(self) -> None:
        self._zoom_factor = 1.0
        self._pan_offset = self._fit_offset()
        self.update()

    def _fit_scale(self) -> float:
        if self._pixmap is None or self._pixmap.isNull():
            return 1.0

        width_scale = self.width() / max(1, self._pixmap.width())
        height_scale = self.height() / max(1, self._pixmap.height())
        return min(width_scale, height_scale)

    def _fit_offset(self) -> QPointF:
        if self._pixmap is None or self._pixmap.isNull():
            return QPointF(0.0, 0.0)

        scale = self._fit_scale() * self._zoom_factor
        scaled_width = self._pixmap.width() * scale
        scaled_height = self._pixmap.height() * scale
        return QPointF((self.width() - scaled_width) / 2.0, (self.height() - scaled_height) / 2.0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._zoom_factor == 1.0:
            self.reset_view()
        else:
            self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if self._pixmap is None:
            painter.setPen(Qt.lightGray)
            painter.drawText(self.rect(), Qt.AlignCenter, self._title)
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        scale = self._fit_scale() * self._zoom_factor
        target = QRectF(
            self._pan_offset.x(),
            self._pan_offset.y(),
            self._pixmap.width() * scale,
            self._pixmap.height() * scale,
        )
        source = QRectF(0.0, 0.0, self._pixmap.width(), self._pixmap.height())
        painter.drawPixmap(target, self._pixmap, source)

    def wheelEvent(self, event) -> None:
        if self._pixmap is None:
            return

        old_zoom = self._zoom_factor
        if event.angleDelta().y() > 0:
            new_zoom = old_zoom * 1.25
        else:
            new_zoom = old_zoom / 1.25
        new_zoom = max(1.0, min(8.0, new_zoom))
        if new_zoom == old_zoom:
            event.accept()
            return

        mouse_pos = event.position()
        fit_scale = self._fit_scale()
        old_scale = fit_scale * old_zoom
        new_scale = fit_scale * new_zoom
        image_pos = (mouse_pos - self._pan_offset) / old_scale

        self._zoom_factor = new_zoom
        if self._zoom_factor == 1.0:
            self._pan_offset = self._fit_offset()
        else:
            self._pan_offset = mouse_pos - image_pos * new_scale
        self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._pixmap is not None and self._zoom_factor > 1.0:
            self._dragging = True
            self._last_drag_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            current_pos = event.position()
            self._pan_offset += current_pos - self._last_drag_pos
            self._last_drag_pos = current_pos
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pose visualization outputs from two methods.")
    parser.add_argument("--method_a_dir", type=Path, required=True, help="Image directory for method A")
    parser.add_argument("--method_b_dir", type=Path, required=True, help="Image directory for method B")
    parser.add_argument("--method_a_name", type=str, default="method_a", help="Display name for method A")
    parser.add_argument("--method_b_name", type=str, default="method_b", help="Display name for method B")
    parser.add_argument("--output", type=Path, default=None, help="Path to save compare JSON")
    return parser.parse_args()


def default_output_path(method_a_name: str, method_b_name: str) -> Path:
    safe_a = re.sub(r"[^A-Za-z0-9_.-]+", "_", method_a_name).strip("_") or "method_a"
    safe_b = re.sub(r"[^A-Za-z0-9_.-]+", "_", method_b_name).strip("_") or "method_b"
    return Path(f"compare_{safe_a}_vs_{safe_b}.json")


def load_existing_decisions(json_path: Path) -> dict[int, str]:
    if not json_path.exists():
        return {}

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    frames = data.get("frames", {}) if isinstance(data, dict) else {}
    decisions: dict[int, str] = {}
    if not isinstance(frames, dict):
        return decisions

    for key, value in frames.items():
        try:
            frame_idx = int(key)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("decision") in DECISIONS:
            decisions[frame_idx] = str(value["decision"])
    return decisions


class ComparePoseVisReviewer(QWidget):
    def __init__(
        self,
        method_a_dir: Path,
        method_b_dir: Path,
        method_a_name: str,
        method_b_name: str,
        output_path: Path,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"Select Best TPS Direction: {method_a_name} vs {method_b_name}")

        self.method_a_dir = method_a_dir
        self.method_b_dir = method_b_dir
        self.method_a_name = method_a_name
        self.method_b_name = method_b_name
        self.output_path = output_path

        self.method_a_images = scan_images(self.method_a_dir)
        self.method_b_images = scan_images(self.method_b_dir)
        self.frame_indices = sorted(set(self.method_a_images) | set(self.method_b_images))
        if not self.frame_indices:
            QMessageBox.critical(self, "No Images", "No supported images were found.")
            raise SystemExit(1)

        self.decisions = load_existing_decisions(self.output_path)
        self.current_index = 0
        self._displayed_frame_idx: int | None = None

        self.method_a_label = ImageLabel(self.method_a_name)
        self.method_b_label = ImageLabel(self.method_b_name)
        self.status_bar = QStatusBar()

        left_title = QLabel(self.method_a_name)
        right_title = QLabel(self.method_b_name)
        for title in (left_title, right_title):
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("QLabel { font-weight: 600; font-size: 16px; }")

        left_layout = QVBoxLayout()
        left_layout.addWidget(left_title)
        left_layout.addWidget(self.method_a_label, 1)

        right_layout = QVBoxLayout()
        right_layout.addWidget(right_title)
        right_layout.addWidget(self.method_b_label, 1)

        image_layout = QHBoxLayout()
        image_layout.addLayout(left_layout, 1)
        image_layout.addLayout(right_layout, 1)

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
        QShortcut(QKeySequence("1"), self, activated=lambda: self.set_decision("a_better"))
        QShortcut(QKeySequence("2"), self, activated=lambda: self.set_decision("b_better"))
        QShortcut(QKeySequence("3"), self, activated=lambda: self.set_decision("tie"))
        QShortcut(QKeySequence("4"), self, activated=lambda: self.set_decision("both_bad"))
        QShortcut(QKeySequence("C"), self, activated=self.clear_decision)
        QShortcut(QKeySequence("R"), self, activated=self.reset_views)
        QShortcut(QKeySequence("S"), self, activated=self.save_json)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_json)

    def current_frame(self) -> int:
        return self.frame_indices[self.current_index]

    def current_decision(self) -> str | None:
        return self.decisions.get(self.current_frame())

    def overlays(self) -> tuple[tuple[str | None, tuple[int, int, int] | None], tuple[str | None, tuple[int, int, int] | None]]:
        decision = self.current_decision()
        if decision == "a_better":
            return ("A BETTER", (0, 255, 0)), (None, None)
        if decision == "b_better":
            return (None, None), ("B BETTER", (0, 255, 0))
        if decision == "tie":
            return ("TIE", (255, 255, 0)), ("TIE", (255, 255, 0))
        if decision == "both_bad":
            return ("BOTH BAD", (255, 0, 0)), ("BOTH BAD", (255, 0, 0))
        return (None, None), (None, None)

    def refresh(self) -> None:
        frame_idx = self.current_frame()
        reset_view = frame_idx != self._displayed_frame_idx
        a_overlay, b_overlay = self.overlays()

        method_a_path = self.method_a_images.get(frame_idx)
        method_b_path = self.method_b_images.get(frame_idx)
        method_a_name = method_a_path.name if method_a_path is not None else "missing"
        method_b_name = method_b_path.name if method_b_path is not None else "missing"

        method_a_image = load_rgb_image(method_a_path, f"missing {self.method_a_name}")
        method_b_image = load_rgb_image(method_b_path, f"missing {self.method_b_name}")
        self.method_a_label.set_image(draw_overlay(method_a_image, a_overlay[0], a_overlay[1]), reset_view=reset_view)
        self.method_b_label.set_image(draw_overlay(method_b_image, b_overlay[0], b_overlay[1]), reset_view=reset_view)
        self._displayed_frame_idx = frame_idx

        current = self.current_decision() or "unreviewed"
        status = (
            f"frame {frame_idx:05d} | "
            f"image {self.current_index + 1}/{len(self.frame_indices)} | "
            f"current={current} | "
            f"A={method_a_name} | B={method_b_name} | "
            f"1=A better, 2=B better, 3=tie, 4=both bad, S=save, Left/Right=prev/next, R=reset view"
        )
        self.status_bar.showMessage(status)

    def step(self, delta: int) -> None:
        self.go_to(self.current_index + delta)

    def go_to(self, index: int) -> None:
        self.current_index = max(0, min(len(self.frame_indices) - 1, index))
        self.refresh()

    def set_decision(self, decision: str) -> None:
        current_frame = self.current_frame()
        if self.decisions.get(current_frame) == decision:
            self.decisions.pop(current_frame, None)
        else:
            self.decisions[current_frame] = decision
        self.refresh()

    def clear_decision(self) -> None:
        self.decisions.pop(self.current_frame(), None)
        self.refresh()

    def reset_views(self) -> None:
        self.method_a_label.reset_view()
        self.method_b_label.reset_view()

    def summary(self) -> dict[str, int]:
        counts = {decision: 0 for decision in sorted(DECISIONS)}
        for frame_idx in self.frame_indices:
            decision = self.decisions.get(frame_idx)
            if decision in counts:
                counts[decision] += 1
        return {
            "a_better": counts["a_better"],
            "b_better": counts["b_better"],
            "tie": counts["tie"],
            "both_bad": counts["both_bad"],
            "unreviewed": len(self.frame_indices) - sum(counts.values()),
        }

    def build_output(self) -> dict:
        valid_frames = set(self.frame_indices)
        frames = {}
        for frame_idx, decision in sorted(self.decisions.items()):
            if frame_idx not in valid_frames or decision not in DECISIONS:
                continue

            if decision == "a_better":
                selected_method = self.method_a_name
                selected_image = str(self.method_a_images.get(frame_idx, ""))
            elif decision == "b_better":
                selected_method = self.method_b_name
                selected_image = str(self.method_b_images.get(frame_idx, ""))
            else:
                selected_method = decision
                selected_image = None

            if selected_image is not None and not Path(selected_image).exists():
                selected_image = None

            frames[f"{frame_idx:05d}"] = {
                "decision": decision,
                "selected_method": selected_method,
                "selected_image": selected_image,
            }

        return {
            "meta": {
                "task": "select_best_tps_direction",
                "method_a_name": self.method_a_name,
                "method_b_name": self.method_b_name,
                "method_a_dir": str(self.method_a_dir),
                "method_b_dir": str(self.method_b_dir),
                "total_frames": len(self.frame_indices),
                "output_usage": "Use selected_image to generate final video.",
            },
            "frames": frames,
            "summary": self.summary(),
        }

    def save_json(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.build_output()
        with self.output_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        self.status_bar.showMessage(f"saved to {self.output_path}")

    def closeEvent(self, event) -> None:
        self.save_json()
        super().closeEvent(event)


def main() -> int:
    args = parse_args()
    method_a_dir = args.method_a_dir.resolve()
    method_b_dir = args.method_b_dir.resolve()
    output_path = args.output if args.output is not None else default_output_path(args.method_a_name, args.method_b_name)

    app = QApplication(sys.argv)
    if not method_a_dir.is_dir():
        QMessageBox.critical(None, "Invalid Folder", f"method_a_dir does not exist:\n{method_a_dir}")
        return 1
    if not method_b_dir.is_dir():
        QMessageBox.critical(None, "Invalid Folder", f"method_b_dir does not exist:\n{method_b_dir}")
        return 1

    window = ComparePoseVisReviewer(
        method_a_dir=method_a_dir,
        method_b_dir=method_b_dir,
        method_a_name=args.method_a_name,
        method_b_name=args.method_b_name,
        output_path=output_path,
    )
    window.resize(1600, 800)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

r"""
# Compare cam2_to_cam4 vs cam4_to_cam2 directional filled prior
python src/script/test_function/debug042_compare_pose_vis.py `
  --method_a_dir data_process/Output/20260311_riding/tps_result/kp_densekp_fitted_filter_all/cam2_to_cam4/directional_filled_prior `
  --method_b_dir data_process/Output/20260311_riding/tps_result/kp_densekp_fitted_filter_all/cam4_to_cam2/directional_filled_prior `
  --method_a_name cam2_to_cam4 `
  --method_b_name cam4_to_cam2 `
  --output data_process/Output/20260311_riding/tps_result/kp_densekp_fitted_filter_all/best_direction_selection.json
"""
