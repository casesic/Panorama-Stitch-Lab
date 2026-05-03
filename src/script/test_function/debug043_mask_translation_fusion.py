from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is only for CLI ergonomics.
    tqdm = None


CAMERAS = ("cam2", "cam4")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse dual-camera SAM3 crop masks with a coarse feature-match translation baseline. "
            "The output mask/ folders are compatible with debug029 --seg_dir."
        )
    )
    parser.add_argument("--crop_image_dir", type=str, required=True, help="Root containing cam2/00513.jpg and cam4/00513.jpg.")
    parser.add_argument("--seg_dir", type=str, required=True, help="SAM3 segmentation root containing camX/mask/00513.png.")
    parser.add_argument("--bbox_dir", type=str, required=True, help="Directory containing cam2_bboxes.json and cam4_bboxes.json.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory for fused masks and debug data.")
    parser.add_argument("--start_frame", type=int, default=None, help="Optional start frame, inclusive.")
    parser.add_argument("--end_frame", type=int, default=None, help="Optional end frame, inclusive.")
    parser.add_argument("--method", choices=("loftr", "orb"), default="loftr", help="Feature matching method.")
    parser.add_argument("--max_width", type=int, default=640, help="Max width for LoFTR resize.")
    parser.add_argument("--max_height", type=int, default=480, help="Max height for LoFTR resize.")
    parser.add_argument("--min_matches", type=int, default=20, help="Minimum raw matches for translation estimation.")
    parser.add_argument("--min_inliers", type=int, default=8, help="Minimum robust inliers for a valid translation.")
    parser.add_argument("--max_mad", type=float, default=25.0, help="Residual threshold around median translation.")
    parser.add_argument("--image_width", type=int, default=5760, help="Full equirect_rotate canvas width.")
    parser.add_argument("--image_height", type=int, default=2880, help="Full equirect_rotate canvas height.")
    parser.add_argument("--min_component_area", type=int, default=0, help="Optional removal of tiny mask components; 0 disables.")
    parser.add_argument("--close_kernel", type=int, default=0, help="Optional morphology close kernel size; 0 disables.")
    parser.add_argument("--save_vis", action="store_true", help="Save match and fusion visualizations.")
    return parser.parse_args()


def load_bbox_dict(json_path: Path) -> dict[str, list[int]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"bbox json must contain a dict: {json_path}")
    return {str(key): [int(round(v)) for v in value[:4]] for key, value in data.items() if value is not None}


def get_bbox(bboxes: dict[str, list[int]], frame_idx: int) -> list[int] | None:
    return bboxes.get(str(frame_idx)) or bboxes.get(f"{frame_idx:05d}")


def find_image_path(image_dir: Path, frame_idx: int) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = image_dir / f"{frame_idx:05d}{suffix}"
        if path.exists():
            return path
    return None


def discover_frames(crop_image_dir: Path, seg_dir: Path, cam2_bboxes: dict[str, list[int]], cam4_bboxes: dict[str, list[int]]) -> list[int]:
    frame_indices: set[int] = set()
    for camera in CAMERAS:
        for root in (crop_image_dir / camera, seg_dir / camera / "mask"):
            if not root.exists():
                continue
            for path in root.iterdir():
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    try:
                        frame_indices.add(int(path.stem))
                    except ValueError:
                        pass
    for bboxes in (cam2_bboxes, cam4_bboxes):
        for key in bboxes:
            try:
                frame_indices.add(int(key))
            except ValueError:
                pass
    return sorted(frame_indices)


def read_image_bgr(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def read_mask(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return ((mask > 0).astype(np.uint8) * 255)


def resize_to_limit(image: np.ndarray, max_width: int, max_height: int) -> tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_width / max(1, width), max_height / max(1, height))
    if scale >= 1.0:
        return image, 1.0, 1.0
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, width / new_width, height / new_height


def match_loftr(
    image0_bgr: np.ndarray,
    image1_bgr: np.ndarray,
    max_width: int,
    max_height: int,
    matcher_cache: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from kornia.feature import LoFTR

    device = matcher_cache.get("device")
    matcher = matcher_cache.get("matcher")
    if matcher is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        last_error: Exception | None = None
        for pretrained in ("outdoor", "indoor_new"):
            try:
                matcher = LoFTR(pretrained=pretrained).eval().to(device)
                matcher_cache["matcher"] = matcher
                matcher_cache["device"] = device
                break
            except Exception as exc:
                last_error = exc
        if matcher is None:
            raise RuntimeError("Failed to load LoFTR with 'outdoor' or 'indoor_new'.") from last_error

    resized0, scale_x0, scale_y0 = resize_to_limit(image0_bgr, max_width, max_height)
    resized1, scale_x1, scale_y1 = resize_to_limit(image1_bgr, max_width, max_height)
    gray0 = cv2.cvtColor(resized0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2GRAY)
    # Avoid torch.from_numpy because some Windows environments have torch built
    # against NumPy 1.x while the active runtime uses NumPy 2.x.
    tensor0 = torch.frombuffer(gray0.tobytes(), dtype=torch.uint8).reshape(gray0.shape).float()
    tensor1 = torch.frombuffer(gray1.tobytes(), dtype=torch.uint8).reshape(gray1.shape).float()
    tensor0 = tensor0.div(255.0).unsqueeze(0).unsqueeze(0).to(device)
    tensor1 = tensor1.div(255.0).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        correspondences = matcher({"image0": tensor0, "image1": tensor1})

    pts0 = np.asarray(correspondences["keypoints0"].detach().cpu().tolist(), dtype=np.float32).reshape(-1, 2)
    pts1 = np.asarray(correspondences["keypoints1"].detach().cpu().tolist(), dtype=np.float32).reshape(-1, 2)
    confidence = np.asarray(correspondences["confidence"].detach().cpu().tolist(), dtype=np.float32).reshape(-1)
    if pts0.size:
        pts0[:, 0] *= scale_x0
        pts0[:, 1] *= scale_y0
        pts1[:, 0] *= scale_x1
        pts1[:, 1] *= scale_y1
    return pts0, pts1, confidence


def match_orb(image0_bgr: np.ndarray, image1_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray0 = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=4000)
    kps0, desc0 = orb.detectAndCompute(gray0, None)
    kps1, desc1 = orb.detectAndCompute(gray1, None)
    if desc0 is None or desc1 is None or not kps0 or not kps1:
        return empty_matches()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(desc0, desc1), key=lambda item: item.distance)
    pts0 = np.array([kps0[item.queryIdx].pt for item in matches], dtype=np.float32)
    pts1 = np.array([kps1[item.trainIdx].pt for item in matches], dtype=np.float32)
    if not matches:
        return empty_matches()
    distances = np.array([item.distance for item in matches], dtype=np.float32)
    confidence = 1.0 - distances / max(float(distances.max()), 1.0)
    return pts0, pts1, confidence


def empty_matches() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )


def estimate_translation(
    pts_cam2_crop: np.ndarray,
    pts_cam4_crop: np.ndarray,
    cam2_bbox: list[int],
    cam4_bbox: list[int],
    min_matches: int,
    min_inliers: int,
    max_mad: float,
) -> dict[str, Any]:
    match_count = int(len(pts_cam2_crop))
    result: dict[str, Any] = {
        "valid_translation": False,
        "dx_cam2_to_cam4": 0.0,
        "dy_cam2_to_cam4": 0.0,
        "match_count": match_count,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "mad": None,
        "reason": "ok",
        "inlier_mask": np.zeros(match_count, dtype=bool),
    }
    if match_count < min_matches:
        result["reason"] = "low_matches"
        return result

    pts2_full = pts_cam2_crop.copy()
    pts4_full = pts_cam4_crop.copy()
    pts2_full[:, 0] += cam2_bbox[0]
    pts2_full[:, 1] += cam2_bbox[1]
    pts4_full[:, 0] += cam4_bbox[0]
    pts4_full[:, 1] += cam4_bbox[1]

    dx_all = pts4_full[:, 0] - pts2_full[:, 0]
    dy_all = pts4_full[:, 1] - pts2_full[:, 1]
    dx0 = float(np.median(dx_all))
    dy0 = float(np.median(dy_all))
    residual = np.sqrt((dx_all - dx0) ** 2 + (dy_all - dy0) ** 2)
    inlier_mask = residual < max_mad
    inlier_count = int(inlier_mask.sum())
    mad = float(np.median(residual)) if residual.size else None

    result.update(
        {
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_count / max(match_count, 1)),
            "mad": mad,
            "inlier_mask": inlier_mask,
        }
    )
    if inlier_count < min_inliers:
        result["reason"] = "low_inliers"
        return result

    result.update(
        {
            "valid_translation": True,
            "dx_cam2_to_cam4": float(np.median(dx_all[inlier_mask])),
            "dy_cam2_to_cam4": float(np.median(dy_all[inlier_mask])),
            "reason": "ok",
        }
    )
    return result


def clip_bbox(bbox: list[int], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def place_crop_mask_on_full(mask_crop: np.ndarray, bbox: list[int], width: int, height: int) -> np.ndarray:
    clipped = clip_bbox(bbox, width, height)
    full = np.zeros((height, width), dtype=np.uint8)
    if clipped is None:
        return full
    x1, y1, x2, y2 = clipped
    target_w = x2 - x1
    target_h = y2 - y1
    if mask_crop.shape[:2] != (target_h, target_w):
        mask_crop = cv2.resize(mask_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    full[y1:y2, x1:x2] = (mask_crop > 0).astype(np.uint8) * 255
    return full


def translate_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = mask.shape[:2]
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def postprocess_mask(mask: np.ndarray, min_component_area: int = 0, close_kernel: int = 0) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if close_kernel and close_kernel > 1:
        kernel_size = int(close_kernel)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if min_component_area and min_component_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
        cleaned = np.zeros_like(binary)
        for label in range(1, num_labels):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_component_area:
                cleaned[labels == label] = 255
        binary = cleaned
    return binary


def crop_full_mask(full_mask: np.ndarray, bbox: list[int], width: int, height: int) -> np.ndarray:
    clipped = clip_bbox(bbox, width, height)
    if clipped is None:
        return np.zeros((1, 1), dtype=np.uint8)
    x1, y1, x2, y2 = clipped
    return full_mask[y1:y2, x1:x2].copy()


def ensure_output_dirs(output_dir: Path) -> None:
    for camera in CAMERAS:
        for name in ("mask", "full_original_mask", "full_completed_mask", "vis"):
            (output_dir / camera / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "cam2" / "full_warped_from_cam4").mkdir(parents=True, exist_ok=True)
    (output_dir / "cam4" / "full_warped_from_cam2").mkdir(parents=True, exist_ok=True)
    (output_dir / "matches").mkdir(parents=True, exist_ok=True)


def save_original_fallback(seg_dir: Path, output_dir: Path, frame_idx: int, reason: str) -> None:
    frame_name = f"{frame_idx:05d}.png"
    for camera in CAMERAS:
        source = seg_dir / camera / "mask" / frame_name
        target = output_dir / camera / "mask" / frame_name
        target.parent.mkdir(parents=True, exist_ok=True)
        mask = read_mask(source)
        if mask is not None:
            cv2.imwrite(str(target), mask)
    print(f"Warning: fallback original masks for frame {frame_idx:05d}: {reason}")


def draw_mask_contour(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], label: str) -> np.ndarray:
    vis = image_bgr.copy()
    if mask.shape[:2] != vis.shape[:2]:
        mask = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color, 2)
    cv2.putText(vis, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return vis


def make_fusion_vis(
    image_bgr: np.ndarray,
    original_crop: np.ndarray,
    warped_crop: np.ndarray,
    completed_crop: np.ndarray,
) -> np.ndarray:
    panels = [
        draw_mask_contour(image_bgr, original_crop, (0, 255, 255), "original SAM3"),
        draw_mask_contour(image_bgr, warped_crop, (255, 0, 255), "warped support"),
        draw_mask_contour(image_bgr, completed_crop, (0, 255, 0), "completed support"),
    ]
    return cv2.hconcat(panels)


def draw_matches_vis(
    image0_bgr: np.ndarray,
    image1_bgr: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    confidence: np.ndarray,
    inlier_mask: np.ndarray,
    max_draw: int = 200,
) -> np.ndarray:
    if len(pts0) == 0:
        height = max(image0_bgr.shape[0], image1_bgr.shape[0])
        width = image0_bgr.shape[1] + image1_bgr.shape[1]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[: image0_bgr.shape[0], : image0_bgr.shape[1]] = image0_bgr
        canvas[: image1_bgr.shape[0], image0_bgr.shape[1] :] = image1_bgr
        cv2.putText(canvas, "no matches", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return canvas

    order = np.argsort(-confidence) if confidence.size == len(pts0) else np.arange(len(pts0))
    selected = order[: min(max_draw, len(order))]
    kps0 = [cv2.KeyPoint(float(pts0[i, 0]), float(pts0[i, 1]), 1) for i in selected]
    kps1 = [cv2.KeyPoint(float(pts1[i, 0]), float(pts1[i, 1]), 1) for i in selected]
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(len(selected))]
    vis = cv2.drawMatches(
        image0_bgr,
        kps0,
        image1_bgr,
        kps1,
        matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(180, 180, 180),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    offset_x = image0_bgr.shape[1]
    for local_idx, source_idx in enumerate(selected):
        if source_idx >= len(inlier_mask) or bool(inlier_mask[source_idx]):
            continue
        p0 = tuple(np.round(kps0[local_idx].pt).astype(int))
        p1_raw = np.round(kps1[local_idx].pt).astype(int)
        p1 = (int(p1_raw[0] + offset_x), int(p1_raw[1]))
        cv2.line(vis, p0, p1, (0, 0, 255), 1, cv2.LINE_AA)
    return vis


def validate_inputs(
    crop_image_dir: Path,
    seg_dir: Path,
    cam2_bboxes: dict[str, list[int]],
    cam4_bboxes: dict[str, list[int]],
    frame_idx: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    missing: list[str] = []
    data: dict[str, Any] = {}
    for camera, bboxes in (("cam2", cam2_bboxes), ("cam4", cam4_bboxes)):
        image_path = find_image_path(crop_image_dir / camera, frame_idx)
        mask_path = seg_dir / camera / "mask" / f"{frame_idx:05d}.png"
        bbox = get_bbox(bboxes, frame_idx)
        if image_path is None:
            missing.append(f"{camera}_crop_image")
        if not mask_path.exists():
            missing.append(f"{camera}_mask")
        if bbox is None:
            missing.append(f"{camera}_bbox")
        data[camera] = {"image_path": image_path, "mask_path": mask_path, "bbox": bbox}

    if missing:
        return None, missing

    for camera in CAMERAS:
        image = read_image_bgr(data[camera]["image_path"])
        mask = read_mask(data[camera]["mask_path"])
        if image is None:
            missing.append(f"{camera}_crop_image_read_failed")
        if mask is None:
            missing.append(f"{camera}_mask_read_failed")
        data[camera]["image"] = image
        data[camera]["mask"] = mask

    if missing:
        return None, missing
    return data, []


def process_frame(
    frame_idx: int,
    args: argparse.Namespace,
    crop_image_dir: Path,
    seg_dir: Path,
    output_dir: Path,
    cam2_bboxes: dict[str, list[int]],
    cam4_bboxes: dict[str, list[int]],
    matcher_cache: dict[str, Any],
) -> dict[str, Any]:
    frame_key = f"{frame_idx:05d}"
    data, missing = validate_inputs(crop_image_dir, seg_dir, cam2_bboxes, cam4_bboxes, frame_idx)
    cam2_bbox = get_bbox(cam2_bboxes, frame_idx)
    cam4_bbox = get_bbox(cam4_bboxes, frame_idx)
    base_meta: dict[str, Any] = {
        "valid_translation": False,
        "dx_cam2_to_cam4": 0.0,
        "dy_cam2_to_cam4": 0.0,
        "match_count": 0,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "mad": None,
        "reason": "ok",
        "cam2_bbox": cam2_bbox,
        "cam4_bbox": cam4_bbox,
    }
    if data is None:
        reason = "missing_input:" + ",".join(missing)
        save_original_fallback(seg_dir, output_dir, frame_idx, reason)
        base_meta["reason"] = reason
        return base_meta

    image2 = data["cam2"]["image"]
    image4 = data["cam4"]["image"]
    mask2_crop = data["cam2"]["mask"]
    mask4_crop = data["cam4"]["mask"]
    cam2_bbox = data["cam2"]["bbox"]
    cam4_bbox = data["cam4"]["bbox"]
    assert cam2_bbox is not None and cam4_bbox is not None

    try:
        if args.method == "loftr":
            pts2_crop, pts4_crop, confidence = match_loftr(image2, image4, args.max_width, args.max_height, matcher_cache)
            matcher_used = "loftr"
        else:
            pts2_crop, pts4_crop, confidence = match_orb(image2, image4)
            matcher_used = "orb"
    except Exception as exc:
        if args.method == "loftr":
            print(f"Warning: LoFTR failed for frame {frame_key}, trying ORB fallback: {exc}")
            pts2_crop, pts4_crop, confidence = match_orb(image2, image4)
            matcher_used = "orb_fallback"
        else:
            print(f"Warning: ORB failed for frame {frame_key}: {exc}")
            pts2_crop, pts4_crop, confidence = empty_matches()
            matcher_used = "orb_failed"

    translation = estimate_translation(
        pts_cam2_crop=pts2_crop,
        pts_cam4_crop=pts4_crop,
        cam2_bbox=cam2_bbox,
        cam4_bbox=cam4_bbox,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        max_mad=args.max_mad,
    )

    full2_original = place_crop_mask_on_full(mask2_crop, cam2_bbox, args.image_width, args.image_height)
    full4_original = place_crop_mask_on_full(mask4_crop, cam4_bbox, args.image_width, args.image_height)

    if translation["valid_translation"]:
        dx = float(translation["dx_cam2_to_cam4"])
        dy = float(translation["dy_cam2_to_cam4"])
        full2_to_4 = translate_mask(full2_original, dx, dy)
        full4_to_2 = translate_mask(full4_original, -dx, -dy)
        full2_completed = cv2.bitwise_or(full2_original, full4_to_2)
        full4_completed = cv2.bitwise_or(full4_original, full2_to_4)
    else:
        full2_to_4 = np.zeros_like(full2_original)
        full4_to_2 = np.zeros_like(full4_original)
        full2_completed = full2_original.copy()
        full4_completed = full4_original.copy()

    full2_completed = postprocess_mask(full2_completed, args.min_component_area, args.close_kernel)
    full4_completed = postprocess_mask(full4_completed, args.min_component_area, args.close_kernel)

    mask2_completed_crop = crop_full_mask(full2_completed, cam2_bbox, args.image_width, args.image_height)
    mask4_completed_crop = crop_full_mask(full4_completed, cam4_bbox, args.image_width, args.image_height)
    warped4_to_2_crop = crop_full_mask(full4_to_2, cam2_bbox, args.image_width, args.image_height)
    warped2_to_4_crop = crop_full_mask(full2_to_4, cam4_bbox, args.image_width, args.image_height)

    cv2.imwrite(str(output_dir / "cam2" / "mask" / f"{frame_key}.png"), mask2_completed_crop)
    cv2.imwrite(str(output_dir / "cam4" / "mask" / f"{frame_key}.png"), mask4_completed_crop)
    cv2.imwrite(str(output_dir / "cam2" / "full_original_mask" / f"{frame_key}.png"), full2_original)
    cv2.imwrite(str(output_dir / "cam4" / "full_original_mask" / f"{frame_key}.png"), full4_original)
    cv2.imwrite(str(output_dir / "cam2" / "full_warped_from_cam4" / f"{frame_key}.png"), full4_to_2)
    cv2.imwrite(str(output_dir / "cam4" / "full_warped_from_cam2" / f"{frame_key}.png"), full2_to_4)
    cv2.imwrite(str(output_dir / "cam2" / "full_completed_mask" / f"{frame_key}.png"), full2_completed)
    cv2.imwrite(str(output_dir / "cam4" / "full_completed_mask" / f"{frame_key}.png"), full4_completed)

    if args.save_vis:
        vis2 = make_fusion_vis(image2, mask2_crop, warped4_to_2_crop, mask2_completed_crop)
        vis4 = make_fusion_vis(image4, mask4_crop, warped2_to_4_crop, mask4_completed_crop)
        cv2.imwrite(str(output_dir / "cam2" / "vis" / f"{frame_key}.jpg"), vis2)
        cv2.imwrite(str(output_dir / "cam4" / "vis" / f"{frame_key}.jpg"), vis4)
        match_vis = draw_matches_vis(image2, image4, pts2_crop, pts4_crop, confidence, translation["inlier_mask"])
        cv2.imwrite(str(output_dir / "matches" / f"{frame_key}.jpg"), match_vis)

    meta = dict(base_meta)
    meta.update({key: value for key, value in translation.items() if key != "inlier_mask"})
    meta["matcher_used"] = matcher_used
    meta["cam2_bbox"] = cam2_bbox
    meta["cam4_bbox"] = cam4_bbox
    return meta


def summarize(frames_meta: dict[str, dict[str, Any]]) -> dict[str, int]:
    summary = {
        "total_frames": len(frames_meta),
        "valid_translation": 0,
        "fallback_original_only": 0,
        "missing_input": 0,
        "low_matches": 0,
    }
    for frame_meta in frames_meta.values():
        reason = str(frame_meta.get("reason", ""))
        if frame_meta.get("valid_translation"):
            summary["valid_translation"] += 1
        else:
            summary["fallback_original_only"] += 1
        if reason.startswith("missing_input"):
            summary["missing_input"] += 1
        if reason == "low_matches":
            summary["low_matches"] += 1
    return summary


def main() -> None:
    args = parse_args()
    crop_image_dir = Path(args.crop_image_dir)
    seg_dir = Path(args.seg_dir)
    bbox_dir = Path(args.bbox_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dirs(output_dir)

    cam2_bboxes = load_bbox_dict(bbox_dir / "cam2_bboxes.json")
    cam4_bboxes = load_bbox_dict(bbox_dir / "cam4_bboxes.json")
    frame_indices = discover_frames(crop_image_dir, seg_dir, cam2_bboxes, cam4_bboxes)
    if args.start_frame is not None:
        frame_indices = [idx for idx in frame_indices if idx >= args.start_frame]
    if args.end_frame is not None:
        frame_indices = [idx for idx in frame_indices if idx <= args.end_frame]

    if not frame_indices:
        print("Warning: no frames found after filtering.")

    frames_meta: dict[str, dict[str, Any]] = {}
    matcher_cache: dict[str, Any] = {}
    iterator = tqdm(frame_indices, desc="Fusing masks", unit="frame") if tqdm is not None else frame_indices
    for frame_idx in iterator:
        frames_meta[f"{frame_idx:05d}"] = process_frame(
            frame_idx=frame_idx,
            args=args,
            crop_image_dir=crop_image_dir,
            seg_dir=seg_dir,
            output_dir=output_dir,
            cam2_bboxes=cam2_bboxes,
            cam4_bboxes=cam4_bboxes,
            matcher_cache=matcher_cache,
        )

    meta = {
        "meta": {
            "crop_image_dir": str(crop_image_dir),
            "seg_dir": str(seg_dir),
            "bbox_dir": str(bbox_dir),
            "method": args.method,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "note": "Coarse support/completed mask baseline; not semantic-flow ground truth.",
        },
        "frames": frames_meta,
        "summary": summarize(frames_meta),
    }
    with (output_dir / "translation_meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)

    print(f"Processed {len(frames_meta)} frames.")
    print(f"Valid translations: {meta['summary']['valid_translation']}")
    print(f"Saved fused seg_dir to: {output_dir}")


if __name__ == "__main__":
    main()


"""
Example:

python src/script/test_function/debug043_mask_translation_fusion.py `
  --crop_image_dir data_process/Output/20260311_riding/image/equirect_crop_rotate `
  --seg_dir data_process/Output/20260311_riding/seg/equirect_crop_rotate_sam3 `
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate `
  --output_dir data_process/Output/20260311_riding/seg/equirect_crop_rotate_sam3_fused_translation `
  --start_frame 513 `
  --end_frame 518 `
  --method loftr `
  --save_vis
"""

'''
组合然后分组的思路是对的
'''

