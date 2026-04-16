from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rtmlib import Body

from config import MIN_BBOX_AREA_RATIO
from perspective_views import persp_pixel_to_fisheye, remap_to_perspective


def build_detector(device: str = "cuda") -> Body:
    """Build an RTMLib Body detector with RTMDet-M and RTMPose-M."""
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"device must be 'cuda' or 'cpu', got {device}")
    
    
    cache_dir = Path.home() / ".cache" / "rtmlib" / "hub" / "checkpoints"
    det_model = str(cache_dir / "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx")
    pose_model = str(cache_dir / "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx")

    return Body(
        det=det_model,
        det_input_size=(640, 640),
        pose=pose_model,
        pose_input_size=(192, 256),
        mode="balanced",
        backend="onnxruntime",
        device=device,
    )


def _compute_bbox_from_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    padding: float = 10.0,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Estimate a perspective-space bbox from confident keypoints with fixed padding."""
    points = np.asarray(keypoints, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    if points.shape != (17, 2):
        raise ValueError(f"Expected keypoints shape (17, 2), got {points.shape}")
    if scores_arr.shape != (17,):
        raise ValueError(f"Expected scores shape (17,), got {scores_arr.shape}")

    valid = scores_arr > score_threshold
    if int(valid.sum()) < 2:
        valid = np.ones(17, dtype=bool)
    points = points[valid]

    x1 = float(points[:, 0].min()) - padding
    y1 = float(points[:, 1].min()) - padding
    x2 = float(points[:, 0].max()) + padding
    y2 = float(points[:, 1].max()) + padding

    bbox = np.array(
        [
            np.clip(x1, 0.0, 639.0),
            np.clip(y1, 0.0, 639.0),
            np.clip(x2, 0.0, 639.0),
            np.clip(y2, 0.0, 639.0),
        ],
        dtype=np.float32,
    )
    return bbox


def detect_in_perspective(
    persp_img: np.ndarray,
    body: Body,
) -> list[dict[str, np.ndarray]]:
    """Run RTMLib on a perspective image and return keypoints, scores, and derived bbox."""
    image = np.asarray(persp_img)
    if image.shape != (640, 640, 3):
        raise ValueError(f"Expected perspective image shape (640, 640, 3), got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected perspective image dtype uint8, got {image.dtype}")

    keypoints, scores = body(image)
    keypoints_arr = np.asarray(keypoints, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)

    if keypoints_arr.size == 0 or scores_arr.size == 0:
        return []
    if keypoints_arr.ndim != 3 or keypoints_arr.shape[1:] != (17, 2):
        raise ValueError(f"Expected keypoints shape (N, 17, 2), got {keypoints_arr.shape}")
    if scores_arr.ndim != 2 or scores_arr.shape[1] != 17 or scores_arr.shape[0] != keypoints_arr.shape[0]:
        raise ValueError(f"Expected scores shape (N, 17), got {scores_arr.shape}")

    detections: list[dict[str, np.ndarray]] = []
    for person_kps, person_scores in zip(keypoints_arr, scores_arr):
        if np.all(person_scores <= 0.0):
            continue
        bbox = _compute_bbox_from_keypoints(person_kps, person_scores)
        detections.append(
            {
                "keypoints_persp": person_kps.astype(np.float32),
                "keypoint_scores": person_scores.astype(np.float32),
                "bbox_persp": bbox,
            }
        )
    return detections


def project_detections_to_fisheye(
    fisheye_img: np.ndarray,
    detections: list[dict[str, np.ndarray]],
    R: np.ndarray,
    fov_deg: float,
    out_size: tuple[int, int],
    ds_params: dict[str, Any],
) -> list[dict[str, np.ndarray]]:
    """Project perspective detections back to fisheye coordinates."""
    fisheye_detections: list[dict[str, np.ndarray]] = []
    fisheye_h, fisheye_w = fisheye_img.shape[:2]

    for detection in detections:
        keypoints_persp = np.asarray(detection["keypoints_persp"], dtype=np.float32)
        keypoint_scores = np.asarray(detection["keypoint_scores"], dtype=np.float32)
        bbox_persp = np.asarray(detection["bbox_persp"], dtype=np.float32)

        if keypoints_persp.shape != (17, 2):
            raise ValueError(f"Expected keypoints_persp shape (17, 2), got {keypoints_persp.shape}")
        if keypoint_scores.shape != (17,):
            raise ValueError(f"Expected keypoint_scores shape (17,), got {keypoint_scores.shape}")
        if bbox_persp.shape != (4,):
            raise ValueError(f"Expected bbox_persp shape (4,), got {bbox_persp.shape}")

        keypoints_fisheye, keypoints_valid = persp_pixel_to_fisheye(
            keypoints_persp.astype(np.float64),
            R,
            fov_deg,
            out_size,
            ds_params,
        )

        x1, y1, x2, y2 = bbox_persp.astype(np.float64)
        edge_samples = np.linspace(0.0, 1.0, num=10, dtype=np.float64)
        top_edge = np.stack(
            [
                x1 + (x2 - x1) * edge_samples,
                np.full_like(edge_samples, y1),
            ],
            axis=1,
        )
        bottom_edge = np.stack(
            [
                x1 + (x2 - x1) * edge_samples,
                np.full_like(edge_samples, y2),
            ],
            axis=1,
        )
        left_edge = np.stack(
            [
                np.full_like(edge_samples, x1),
                y1 + (y2 - y1) * edge_samples,
            ],
            axis=1,
        )
        right_edge = np.stack(
            [
                np.full_like(edge_samples, x2),
                y1 + (y2 - y1) * edge_samples,
            ],
            axis=1,
        )
        bbox_edge_points = np.concatenate(
            [top_edge, bottom_edge, left_edge, right_edge],
            axis=0,
        )
        bbox_edge_points_fisheye, bbox_valid = persp_pixel_to_fisheye(
            bbox_edge_points,
            R,
            fov_deg,
            out_size,
            ds_params,
        )

        bbox_edge_points_fisheye = np.asarray(bbox_edge_points_fisheye, dtype=np.float32)
        bbox_edge_points_fisheye[:, 0] = np.clip(bbox_edge_points_fisheye[:, 0], 0.0, float(fisheye_w - 1))
        bbox_edge_points_fisheye[:, 1] = np.clip(bbox_edge_points_fisheye[:, 1], 0.0, float(fisheye_h - 1))
        valid_bbox_points = bbox_edge_points_fisheye[np.isfinite(bbox_edge_points_fisheye).all(axis=1)]
        valid_keypoints_mask = (np.asarray(keypoints_valid, dtype=bool)) & (keypoint_scores > 0.3)
        valid_keypoints_points = np.asarray(keypoints_fisheye, dtype=np.float32)[valid_keypoints_mask]
        valid_points: list[np.ndarray] = []
        if len(valid_bbox_points) > 0:
            valid_points.append(np.asarray(valid_bbox_points, dtype=np.float32))
        if len(valid_keypoints_points) > 0:
            valid_points.append(valid_keypoints_points)
        if not valid_points:
            continue

        combined_valid_points = np.concatenate(valid_points, axis=0)
        bbox_fisheye = np.array(
            [
                combined_valid_points[:, 0].min(),
                combined_valid_points[:, 1].min(),
                combined_valid_points[:, 0].max(),
                combined_valid_points[:, 1].max(),
            ],
            dtype=np.float32,
        )

        fisheye_detections.append(
            {
                "keypoints_fisheye": np.asarray(keypoints_fisheye, dtype=np.float32),
                "keypoint_scores": keypoint_scores,
                "bbox_fisheye": bbox_fisheye,
                "keypoints_valid": np.asarray(keypoints_valid, dtype=bool),
            }
        )

    return fisheye_detections


def run_detector_on_view(
    fisheye_img: np.ndarray,
    lut: dict[str, np.ndarray],
    view_name: str,
    R: np.ndarray,
    fov_deg: float,
    out_size: tuple[int, int],
    ds_params: dict[str, Any],
    body: Body,
    detections: list[dict[str, np.ndarray]] | None = None,
    persp_img: np.ndarray | None = None,
) -> list[dict[str, np.ndarray | str]]:
    """Run the detector on one perspective view and project results back to fisheye coordinates."""
    required_keys = {"map_x", "map_y", "valid_mask"}
    if not required_keys.issubset(lut):
        raise KeyError(f"lut must contain keys {required_keys}, got {set(lut.keys())}")

    if persp_img is None:
        persp_img = remap_to_perspective(
            fisheye_img,
            np.asarray(lut["map_x"], dtype=np.float32),
            np.asarray(lut["map_y"], dtype=np.float32),
            np.asarray(lut["valid_mask"], dtype=bool),
        )
    if detections is None:
        detections = detect_in_perspective(persp_img, body)
    fisheye_detections = project_detections_to_fisheye(
        fisheye_img,
        detections,
        R,
        fov_deg,
        out_size,
        ds_params,
    )

    for detection in fisheye_detections:
        detection["source_view"] = view_name
    return fisheye_detections


def detect_person_bbox_in_fisheye(
    fisheye_img: np.ndarray,
    luts: dict[str, dict[str, np.ndarray]],
    fov_deg: float,
    out_size: tuple[int, int],
    ds_params: dict[str, Any],
    body: Body,
    click_point: list[float] | None = None,
    vis_dir: Path | None = None,
    frame_tag: str = "",
) -> np.ndarray | None:
    all_fisheye_detections: list[dict[str, np.ndarray | str]] = []
    vis_tiles: list[np.ndarray] = []

    for view_name, lut in luts.items():
        persp_img = remap_to_perspective(
            fisheye_img,
            np.asarray(lut["map_x"], dtype=np.float32),
            np.asarray(lut["map_y"], dtype=np.float32),
            np.asarray(lut["valid_mask"], dtype=bool),
        )
        detections = detect_in_perspective(persp_img, body)

        if vis_dir is not None:
            vis_image = persp_img.copy()
            for detection in detections:
                x1, y1, x2, y2 = np.asarray(detection["bbox_persp"], dtype=np.float32).reshape(4)
                cv2.rectangle(
                    vis_image,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                vis_image,
                view_name,
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            vis_tiles.append(vis_image)

        fisheye_detections = run_detector_on_view(
            fisheye_img,
            lut,
            view_name,
            lut["R"],
            fov_deg,
            out_size,
            ds_params,
            body,
            detections=detections,
            persp_img=persp_img,
        )
        all_fisheye_detections.extend(fisheye_detections)

    if vis_dir is not None and vis_tiles:
        vis_dir.mkdir(parents=True, exist_ok=True)
        tile_h, tile_w = vis_tiles[0].shape[:2]
        num_tiles = len(vis_tiles)
        grid_cols = int(np.ceil(np.sqrt(num_tiles)))
        grid_rows = int(np.ceil(num_tiles / grid_cols))
        blank_tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        padded_tiles = vis_tiles + [blank_tile.copy() for _ in range(grid_rows * grid_cols - num_tiles)]
        row_images = [
            np.hstack(padded_tiles[row_idx * grid_cols:(row_idx + 1) * grid_cols])
            for row_idx in range(grid_rows)
        ]
        grid_image = np.vstack(row_images)
        output_name = f"{frame_tag}_perspective.jpg" if frame_tag else "perspective.jpg"
        cv2.imwrite(str(vis_dir / output_name), grid_image)

    if not all_fisheye_detections:
        return None

    def _bbox_area(detection: dict) -> float:
        bbox = np.asarray(detection["bbox_fisheye"], dtype=np.float32)
        return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    fisheye_h, fisheye_w = fisheye_img.shape[:2]
    min_fisheye_area = fisheye_h * fisheye_w * MIN_BBOX_AREA_RATIO

    large_detections = all_fisheye_detections

    if not large_detections:
        return None

    if click_point is not None:
        click_x = float(click_point[0])
        click_y = float(click_point[1])
        # print(f"click_point={click_point}, large_detections count={len(large_detections)}")
        # for d in large_detections:
        #     bbox = np.asarray(d["bbox_fisheye"], dtype=np.float32)
        #     cx = (bbox[0] + bbox[2]) * 0.5
        #     cy = (bbox[1] + bbox[3]) * 0.5
        #     print(f"  bbox center=({cx:.1f}, {cy:.1f})")
        
        def _bbox_center_distance(detection: dict) -> float:
            bbox = np.asarray(detection["bbox_fisheye"], dtype=np.float32).reshape(4)
            center_x = float((bbox[0] + bbox[2]) * 0.5)
            center_y = float((bbox[1] + bbox[3]) * 0.5)
            return float(np.hypot(center_x - click_x, center_y - click_y))

        best_detection = min(
            large_detections,
            key=_bbox_center_distance,
        )
    else:
        best_detection = max(
            large_detections,
            key=lambda d: float(np.asarray(d["keypoint_scores"], dtype=np.float32).mean()),
        )
    return np.asarray(best_detection["bbox_fisheye"], dtype=np.float32)
