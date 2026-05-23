from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def resize_to_limit(image: np.ndarray, max_width: int, max_height: int) -> tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_width / max(1, width), max_height / max(1, height))
    if scale >= 1.0:
        return image, 1.0, 1.0
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, width / new_width, height / new_height


def empty_matches() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )


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
