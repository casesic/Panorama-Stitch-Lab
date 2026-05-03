from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QStatusBar, QVBoxLayout, QWidget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw bad projected regions and export matched SMPL vertex ids.")
    parser.add_argument("--corr_dir", type=str, required=True)
    parser.add_argument("--domain", choices=("erp", "fisheye"), required=True)
    parser.add_argument("--vis_dir", type=str, required=True)
    parser.add_argument("--exclude_dir", type=str, required=True)
    parser.add_argument("--filtered_corr_dir", type=str, default=None)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    return parser.parse_args()


def discover_frames(corr_dir: Path, domain: str, start_frame: int | None, end_frame: int | None) -> list[int]:
    subdir = corr_dir / f"{domain}_corr"
    pattern = re.compile(rf"^(\d+)_{domain}_corr\.npz$")
    frames: list[int] = []
    for path in subdir.glob(f"*_{domain}_corr.npz"):
        match = pattern.match(path.name)
        if match is None:
            continue
        frame_idx = int(match.group(1))
        if start_frame is not None and frame_idx < start_frame:
            continue
        if end_frame is not None and frame_idx > end_frame:
            continue
        frames.append(frame_idx)
    return sorted(frames)


def corr_path(corr_dir: Path, domain: str, frame_idx: int) -> Path:
    return corr_dir / f"{domain}_corr" / f"{frame_idx:05d}_{domain}_corr.npz"


def load_corr(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["matched_vertex_ids"].astype(np.int64), data["pairs"].astype(np.float32)


def load_rgb_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_ids_txt(path: Path) -> set[int]:
    text = path.read_text(encoding="utf-8").replace(",", " ")
    return {int(item) for item in text.split()}


def load_existing_frame_ids(exclude_dir: Path) -> dict[int, set[int]]:
    frame_ids: dict[int, set[int]] = {}
    for path in exclude_dir.glob("*.txt"):
        frame_ids[int(path.stem)] = load_ids_txt(path)
    return frame_ids


def save_frame_ids(exclude_dir: Path, frame_idx: int, ids: set[int]) -> None:
    exclude_dir.mkdir(parents=True, exist_ok=True)
    path = exclude_dir / f"{frame_idx:05d}.txt"
    with path.open("w", encoding="utf-8") as file:
        for vertex_id in sorted(ids):
            file.write(f"{vertex_id}\n")


def filter_one_corr_npz(input_path: Path, output_path: Path, exclude_ids: set[int]) -> None:
    with np.load(input_path, allow_pickle=False) as data:
        matched_vertex_ids = data["matched_vertex_ids"].astype(np.int64)
        pairs = data["pairs"].astype(np.float32)
        if exclude_ids:
            remove_ids = np.asarray(sorted(exclude_ids), dtype=np.int64)
            keep = ~np.isin(matched_vertex_ids, remove_ids)
        else:
            keep = np.ones(len(matched_vertex_ids), dtype=bool)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            frame_idx=data["frame_idx"],
            segment_name=data["segment_name"],
            image_width=data["image_width"],
            image_height=data["image_height"],
            matched_vertex_ids=matched_vertex_ids[keep],
            pairs=pairs[keep],
        )


def filter_corr_frame(
    corr_dir: Path,
    filtered_corr_dir: Path,
    frame_idx: int,
    exclude_ids: set[int],
) -> None:
    for domain in ("erp", "fisheye"):
        input_path = corr_dir / f"{domain}_corr" / f"{frame_idx:05d}_{domain}_corr.npz"
        if input_path.exists():
            output_path = filtered_corr_dir / f"{domain}_corr" / f"{frame_idx:05d}_{domain}_corr.npz"
            filter_one_corr_npz(input_path, output_path, exclude_ids)


def polygon_to_vertex_ids(
    polygon: np.ndarray,
    pts: np.ndarray,
    matched_vertex_ids: np.ndarray,
) -> np.ndarray:
    if len(polygon) < 3:
        return np.empty((0,), dtype=np.int64)

    poly = polygon.astype(np.float32)
    inside = []
    for point in pts:
        if not np.isfinite(point).all():
            inside.append(False)
            continue
        inside.append(cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0)
    return matched_vertex_ids[np.asarray(inside, dtype=bool)]


def rgb_to_pixmap(image_rgb: np.ndarray) -> QPixmap:
    image_rgb = np.ascontiguousarray(image_rgb)
    h, w, ch = image_rgb.shape
    qimage = QImage(image_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


class ImageLabel(QLabel):
    def __init__(self, cam_id: str, viewer: "BadRegionViewer") -> None:
        super().__init__()
        self.cam_id = cam_id
        self.viewer = viewer
        self.image_rgb: np.ndarray | None = None
        self.stroke_points: list[tuple[float, float]] = []
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._panning = False
        self._last_pan_pos: QPointF | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("QLabel { background: #111; color: #ccc; }")

    def set_image(self, image_rgb: np.ndarray) -> None:
        self.image_rgb = image_rgb
        self.stroke_points = []
        self.reset_view()

    def set_stroke(self, points: list[tuple[float, float]]) -> None:
        self.stroke_points = points
        self.update()

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def _transform(self) -> tuple[float, float, float]:
        if self.image_rgb is None:
            return 0.0, 0.0, 1.0
        img_h, img_w = self.image_rgb.shape[:2]
        base_scale = min(self.width() / img_w, self.height() / img_h)
        display_scale = base_scale * self.zoom
        disp_w = img_w * display_scale
        disp_h = img_h * display_scale
        x0 = (self.width() - disp_w) / 2.0 + self.pan_x
        y0 = (self.height() - disp_h) / 2.0 + self.pan_y
        return x0, y0, display_scale

    def image_to_view(self, x_img: float, y_img: float) -> tuple[float, float]:
        x0, y0, display_scale = self._transform()
        return x0 + x_img * display_scale, y0 + y_img * display_scale

    def view_to_image(self, x_view: float, y_view: float) -> tuple[float, float] | None:
        if self.image_rgb is None:
            return None
        img_h, img_w = self.image_rgb.shape[:2]
        x0, y0, display_scale = self._transform()
        x_img = (x_view - x0) / display_scale
        y_img = (y_view - y0) / display_scale
        if x_img < 0 or x_img >= img_w or y_img < 0 or y_img >= img_h:
            return None
        return x_img, y_img

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.image_rgb is None:
            painter.setPen(Qt.lightGray)
            painter.drawText(self.rect(), Qt.AlignCenter, self.cam_id)
            return

        img_h, img_w = self.image_rgb.shape[:2]
        x0, y0, display_scale = self._transform()
        pixmap = rgb_to_pixmap(self.image_rgb)
        painter.drawPixmap(QRectF(x0, y0, img_w * display_scale, img_h * display_scale), pixmap, QRectF(0, 0, img_w, img_h))

        if len(self.stroke_points) >= 2:
            pen = QPen(Qt.red, 3)
            painter.setPen(pen)
            view_points = [QPointF(*self.image_to_view(x, y)) for x, y in self.stroke_points]
            for p0, p1 in zip(view_points[:-1], view_points[1:]):
                painter.drawLine(p0, p1)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._last_pan_pos = event.position()
            return
        elif event.button() == Qt.LeftButton:
            point = self.view_to_image(event.position().x(), event.position().y())
            if point is not None:
                self.viewer.start_stroke(self.cam_id, point)
        elif event.button() == Qt.RightButton:
            self.viewer.clear_stroke()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning and self._last_pan_pos is not None:
            delta = event.position() - self._last_pan_pos
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self._last_pan_pos = event.position()
            self.update()
            return
        if event.buttons() & Qt.LeftButton:
            point = self.view_to_image(event.position().x(), event.position().y())
            if point is not None:
                self.viewer.add_stroke_point(self.cam_id, point)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self._last_pan_pos = None
            return
        if event.button() == Qt.LeftButton:
            point = self.view_to_image(event.position().x(), event.position().y())
            if point is not None:
                self.viewer.add_stroke_point(self.cam_id, point)
            self.viewer.finish_stroke(self.cam_id)

    def wheelEvent(self, event) -> None:
        mouse_pos = event.position()
        old_img_pos = self.view_to_image(mouse_pos.x(), mouse_pos.y())
        if event.angleDelta().y() > 0:
            self.zoom = min(8.0, self.zoom * 1.25)
        else:
            self.zoom = max(1.0, self.zoom / 1.25)

        if self.zoom == 1.0:
            self.pan_x = 0.0
            self.pan_y = 0.0
        elif old_img_pos is not None:
            new_view = self.image_to_view(old_img_pos[0], old_img_pos[1])
            self.pan_x += mouse_pos.x() - new_view[0]
            self.pan_y += mouse_pos.y() - new_view[1]
        self.update()


class BadRegionViewer(QWidget):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.corr_dir = Path(args.corr_dir)
        self.vis_dir = Path(args.vis_dir)
        self.exclude_dir = Path(args.exclude_dir)
        self.frame_indices = discover_frames(self.corr_dir, args.domain, args.start_frame, args.end_frame)
        self.current_index = 0
        self.frame_excluded_ids = load_existing_frame_ids(self.exclude_dir)
        self.matched_vertex_ids = np.empty((0,), dtype=np.int64)
        self.pairs = np.empty((0, 4), dtype=np.float32)
        self.active_cam: str | None = None
        self.stroke_points: list[tuple[float, float]] = []

        self.setWindowTitle("Draw Bad Sparse Correspondence Regions")
        self.cam2_label = ImageLabel("cam2", self)
        self.cam4_label = ImageLabel("cam4", self)
        self.status_bar = QStatusBar()

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.cam2_label, 1)
        image_layout.addWidget(self.cam4_label, 1)

        layout = QVBoxLayout()
        layout.addLayout(image_layout, 1)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)
        self.setFocusPolicy(Qt.StrongFocus)
        self.resize(1500, 820)
        self.load_current_frame()

    @property
    def frame_idx(self) -> int:
        return self.frame_indices[self.current_index]

    def load_current_frame(self) -> None:
        frame_idx = self.frame_idx
        self.matched_vertex_ids, self.pairs = load_corr(corr_path(self.corr_dir, self.args.domain, frame_idx))
        self.cam2_label.set_image(load_rgb_image(self.vis_dir / "cam2" / f"{frame_idx:05d}.png"))
        self.cam4_label.set_image(load_rgb_image(self.vis_dir / "cam4" / f"{frame_idx:05d}.png"))
        self.clear_stroke()
        self.update_status()

    def update_status(self) -> None:
        frame_count = len(self.frame_excluded_ids.get(self.frame_idx, set()))
        total_frames = sum(1 for ids in self.frame_excluded_ids.values() if len(ids) > 0)
        self.status_bar.showMessage(
            f"frame {self.frame_idx:05d} | frame excluded ids: {frame_count} | total frames with ids: {total_frames}"
        )

    def start_stroke(self, cam_id: str, point: tuple[float, float]) -> None:
        self.active_cam = cam_id
        self.stroke_points = [point]
        self.update_stroke_display()

    def add_stroke_point(self, cam_id: str, point: tuple[float, float]) -> None:
        if self.active_cam != cam_id or not self.stroke_points:
            return
        last = self.stroke_points[-1]
        if np.hypot(point[0] - last[0], point[1] - last[1]) >= 2.0:
            self.stroke_points.append(point)
            self.update_stroke_display()

    def finish_stroke(self, cam_id: str) -> None:
        if self.active_cam == cam_id and len(self.stroke_points) >= 3:
            polygon = np.asarray(self.stroke_points, dtype=np.float32)
            pts = self.pairs[:, 0:2] if cam_id == "cam2" else self.pairs[:, 2:4]
            vertex_ids = polygon_to_vertex_ids(polygon, pts, self.matched_vertex_ids)
            frame_ids = self.frame_excluded_ids.setdefault(self.frame_idx, set())
            frame_ids.update(int(vertex_id) for vertex_id in vertex_ids)
        self.clear_stroke()
        self.update_status()

    def clear_stroke(self) -> None:
        self.active_cam = None
        self.stroke_points = []
        self.update_stroke_display()

    def update_stroke_display(self) -> None:
        self.cam2_label.set_stroke(self.stroke_points if self.active_cam == "cam2" else [])
        self.cam4_label.set_stroke(self.stroke_points if self.active_cam == "cam4" else [])

    def go_to_index(self, index: int) -> None:
        self.current_index = max(0, min(index, len(self.frame_indices) - 1))
        self.load_current_frame()

    def save_current_frame(self) -> None:
        save_frame_ids(self.exclude_dir, self.frame_idx, self.frame_excluded_ids.get(self.frame_idx, set()))
        self.update_status()

    def save_all_frames(self) -> None:
        for frame_idx, ids in self.frame_excluded_ids.items():
            save_frame_ids(self.exclude_dir, frame_idx, ids)
        self.filter_all_corr_frames()
        self.update_status()

    def filter_all_corr_frames(self) -> None:
        if self.args.filtered_corr_dir is None:
            return
        filtered_corr_dir = Path(self.args.filtered_corr_dir)
        for frame_idx in self.frame_indices:
            exclude_ids = self.frame_excluded_ids.get(frame_idx, set())
            filter_corr_frame(self.corr_dir, filtered_corr_dir, frame_idx, exclude_ids)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key_Right:
            self.go_to_index(self.current_index + 1)
        elif key == Qt.Key_Left:
            self.go_to_index(self.current_index - 1)
        elif key == Qt.Key_Home:
            self.go_to_index(0)
        elif key == Qt.Key_End:
            self.go_to_index(len(self.frame_indices) - 1)
        elif key == Qt.Key_C:
            self.clear_stroke()
        elif key == Qt.Key_X:
            self.frame_excluded_ids[self.frame_idx] = set()
            self.update_status()
        elif key == Qt.Key_R:
            self.cam2_label.reset_view()
            self.cam4_label.reset_view()
        elif key == Qt.Key_S or (key == Qt.Key_S and event.modifiers() & Qt.ControlModifier):
            self.save_all_frames()
        elif key in (Qt.Key_Q, Qt.Key_Escape):
            self.save_all_frames()
            self.close()
        else:
            super().keyPressEvent(event)


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    viewer = BadRegionViewer(args)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

r'''
python src/script/test_function/debug050_draw_bad_regions_to_vertex_ids.py `
  --corr_dir data_process/Output/20260311_riding/sparse_smpl_corr_debug045 `
  --domain fisheye `
  --vis_dir data_process/Output/20260311_riding/sparse_smpl_corr_vis/fisheye/fisheye_points_bg `
  --exclude_dir data_process/Output/20260311_riding/frame_exclude_vertex_ids `
  --filtered_corr_dir data_process/Output/20260311_riding/sparse_smpl_corr_debug050_filtered `
  --start_frame 513 `
  --end_frame 518

python src/script/test_function/debug050_draw_bad_regions_to_vertex_ids.py `
  --corr_dir data_process/Output/20260311_riding/sparse_smpl_corr_debug045 `
  --domain erp `
  --vis_dir data_process/Output/20260311_riding/sparse_smpl_corr_vis/erp/erp_points_bg `
  --exclude_dir data_process/Output/20260311_riding/frame_exclude_vertex_ids `
  --filtered_corr_dir data_process/Output/20260311_riding/sparse_smpl_corr_debug050_filtered `
  --start_frame 513 `
  --end_frame 518
'''
