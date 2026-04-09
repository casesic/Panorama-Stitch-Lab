import numpy as np
import torch
import logging
import cv2 as cv
import mediapipe as mp
from typing import Dict, Tuple, Optional
from lib_pano.human3d.skeletion_constant import (
    COCO_17_KEYPOINTS, COCO_17_SKELETON_TOPOLOGY,
    mp_to_coco_idx, mp_keypoints_format,
    COCO_133_KEYPOINTS
)
from lib_pano.vis.visual_human import render_silhouette

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BBOX_YMAX_RATIO = 0.9  # bbox 底部 y 超过图像高度此比例的检测框直接排除


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def _rtmlib_bbox_to_internal(
    bbox: np.ndarray,   # [xmin, ymin, xmax, ymax]
    h: int,
    w: int,
) -> Tuple[int, int, int, int]:
    """rtmlib bbox [xmin,ymin,xmax,ymax] → 内部格式 (ymin,ymax,xmin,xmax)，clip 到图像范围。"""
    xmin = max(0,     int(bbox[0]))
    ymin = max(0,     int(bbox[1]))
    xmax = min(w - 1, int(bbox[2]))
    ymax = min(h - 1, int(bbox[3]))
    return (ymin, ymax, xmin, xmax)


# ---------------------------------------------------------------------------
# 可视化工具
# ---------------------------------------------------------------------------

def draw_keypoints(
    image: np.ndarray,
    result,
    conf_thresh: float = 0.3,
    draw_face: bool = False,
    draw_hands: bool = False,
    is_fitted: bool = False,
):
    """
    在图像上绘制关键点和骨架。
    支持 Result_Single_Human_Single_Frame 对象或直接的 numpy 数组。
    """
    img_draw = image.copy()

    if hasattr(result, 'flag_valid'):
        if not result.flag_valid or result.keypoints is None:
            return image
        kpts = result.keypoints   # (N, 3)
        if result.bbox is not None:
            ymin, ymax, xmin, xmax = result.bbox
            cv.rectangle(img_draw, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    else:
        kpts = np.zeros((len(result), 3), dtype=np.float32)
        kpts[:, :2] = result[:, :2]
        kpts[:, 2]  = result[:, 2] if result.shape[1] >= 3 else 1.0
        conf_thresh = 0.5

    line_color = (255, 0, 255) if is_fitted else (0, 255, 255)
    dot_color  = (0, 255, 255) if is_fitted else (0, 0, 255)
    text_color = (0, 0, 255)   if is_fitted else (255, 0, 0)

    # 身体骨架（COCO-17 拓扑）
    for u, v in COCO_17_SKELETON_TOPOLOGY:
        if kpts[u][2] > conf_thresh and kpts[v][2] > conf_thresh:
            cv.line(img_draw,
                    (int(kpts[u][0]), int(kpts[u][1])),
                    (int(kpts[v][0]), int(kpts[v][1])),
                    line_color, 2)

    for i in range(min(17, kpts.shape[0])):
        x, y, conf = kpts[i]
        if conf > conf_thresh:
            cv.circle(img_draw, (int(x), int(y)), 4, dot_color, -1)
            cv.putText(img_draw, str(i), (int(x), int(y) - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv.LINE_AA)

    # COCO-133 手部（左手 91~111，右手 112~132）
    if draw_hands and kpts.shape[0] >= 133:
        for i in range(91, 133):
            x, y, conf = kpts[i]
            if conf > conf_thresh:
                color = (0, 255, 0) if i < 112 else (0, 200, 255)
                cv.circle(img_draw, (int(x), int(y)), 3, color, -1)
                cv.putText(img_draw,
                           str(i - 91 if i < 112 else i - 112),
                           (int(x), int(y)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv.LINE_AA)

    # COCO-133 面部（23~90）
    if draw_face and kpts.shape[0] >= 91:
        for i in range(23, 91):
            x, y, conf = kpts[i]
            if conf > conf_thresh:
                cv.circle(img_draw, (int(x), int(y)), 1, (255, 0, 0), -1)

    # MediaPipe 527 点的手部（兼容旧格式）
    if draw_hands and kpts.shape[0] >= 59:
        for i in range(17, 59):
            x, y, conf = kpts[i]
            if conf > conf_thresh:
                cv.circle(img_draw, (int(x), int(y)), 2, (0, 255, 0), -1)
                cv.putText(img_draw, str(i), (int(x), int(y)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv.LINE_AA)

    if draw_face and kpts.shape[0] >= 527:
        for i in range(59, 527):
            x, y, conf = kpts[i]
            if conf > conf_thresh:
                cv.circle(img_draw, (int(x), int(y)), 1, (255, 0, 0), -1)

    return img_draw


def save_mesh_overlay(img, verts_cam, faces, camera_params, overlay_path):
    """Render fisheye mesh silhouette and save overlay image."""
    K, D, xi = camera_params['K'], camera_params['D'], camera_params['xi']
    cam = {
        'fx': float(K[0, 0].item()), 'fy': float(K[1, 1].item()),
        'cx': float(K[0, 2].item()), 'cy': float(K[1, 2].item()),
        'k1': float(D[0].item()),    'k2': float(D[1].item()),
        'p1': float(D[2].item()),    'p2': float(D[3].item()),
        'xi': float(xi[0].item()),
    }
    H, W     = img.shape[:2]
    sil_mask = render_silhouette(verts_cam, faces, cam, H, W)
    overlay  = img.copy().astype(np.float32)
    fg       = sil_mask > 0
    overlay[fg] = overlay[fg] * 0.45 + np.array([40, 120, 255], dtype=np.float32) * 0.55
    overlay  = np.clip(overlay, 0, 255).astype(np.uint8)
    contours, _ = cv.findContours(sil_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(overlay, contours, -1, (255, 220, 20), 2)
    cv.imwrite(overlay_path, overlay)
    logger.info(f"Mesh overlay saved to {overlay_path}")


# ---------------------------------------------------------------------------
# 结果数据类
# ---------------------------------------------------------------------------

class Result_Single_Human_Single_Frame:
    def __init__(
        self,
        bbox: Optional[Tuple[int, int, int, int]],
        bbox_score: float,
        keypoints: Optional[np.ndarray],
        keypoints_format: Dict[str, int],
        flag_with_keypoints: bool = True,
    ):
        self.flag_valid          = bbox is not None
        self.bbox                = bbox
        self.bbox_score          = bbox_score
        self.keypoints           = keypoints
        self.keypoints_format    = keypoints_format
        self.flag_with_keypoints = flag_with_keypoints


# ---------------------------------------------------------------------------
# 检测器
# ---------------------------------------------------------------------------

class WholePoseDetector:
    """
    主力检测器：rtmlib Wholebody → COCO-133。
    一次推理输出身体(17) + 脚部(6) + 面部(68) + 左手(21) + 右手(21) = 133 个关键点。

    新项目直接用这个，无需 MediaPipe。
    """

    def __init__(self, device: str = 'cpu'):
        device_str = 'cuda' if 'cuda' in str(device) else 'cpu'
        logger.info(f"Loading RTMLib Wholebody model on {device_str}...")
        from rtmlib import Wholebody
        self.body_model       = Wholebody(device=device_str)
        self.keypoints_format = COCO_133_KEYPOINTS

    def detect_single_image(self, image: np.ndarray) -> Result_Single_Human_Single_Frame:
        bboxes = self.body_model.det_model(image)
        if len(bboxes) == 0:
            return Result_Single_Human_Single_Frame(
                None, 0.0, None, self.keypoints_format
            )

        h, w      = image.shape[:2]
        ymax_thresh = h * BBOX_YMAX_RATIO
        filtered_bboxes = bboxes[bboxes[:, 3] <= ymax_thresh]
        if len(filtered_bboxes) > 0:
            areas = (filtered_bboxes[:, 2] - filtered_bboxes[:, 0]) * (filtered_bboxes[:, 3] - filtered_bboxes[:, 1])
            best_bbox = filtered_bboxes[np.argmax(areas)]
        else:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            best_bbox = bboxes[np.argmax(areas)]

        kpts, scores = self.body_model.pose_model(
            image, bboxes=np.array([best_bbox])
        )
        # kpts:   (1, 133, 2)
        # scores: (1, 133) 或 (1, 133, 1)，展平成 (133,)
        scores_flat = np.array(scores[0]).reshape(133)

        keypoints        = np.zeros((133, 3), dtype=np.float32)
        keypoints[:, :2] = kpts[0]       # (133, 2)
        keypoints[:, 2]  = scores_flat   # (133,)

        return Result_Single_Human_Single_Frame(
            _rtmlib_bbox_to_internal(best_bbox, h, w),
            1.0, keypoints, self.keypoints_format
        )


class PoseDetector:
    """
    仅检测 COCO-17 身体关键点（保留兼容，新项目推荐用 WholePoseDetector）。
    """

    def __init__(self, device: str = 'cpu'):
        device_str = 'cuda' if 'cuda' in str(device) else 'cpu'
        logger.info(f"Loading RTMLib Body model on {device_str}...")
        from rtmlib import Body
        self.body_model       = Body(device=device_str)
        self.keypoints_format = COCO_17_KEYPOINTS

    def detect_single_image(self, image: np.ndarray) -> Result_Single_Human_Single_Frame:
        bboxes = self.body_model.det_model(image)
        if len(bboxes) == 0:
            return Result_Single_Human_Single_Frame(
                None, 0.0, None, self.keypoints_format
            )

        h, w      = image.shape[:2]
        ymax_thresh = h * BBOX_YMAX_RATIO
        filtered_bboxes = bboxes[bboxes[:, 3] <= ymax_thresh]
        if len(filtered_bboxes) > 0:
            areas = (filtered_bboxes[:, 2] - filtered_bboxes[:, 0]) * (filtered_bboxes[:, 3] - filtered_bboxes[:, 1])
            best_bbox = filtered_bboxes[np.argmax(areas)]
        else:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            best_bbox = bboxes[np.argmax(areas)]

        kpts, scores = self.body_model.pose_model(
            image, bboxes=np.array([best_bbox])
        )
        keypoints        = np.zeros((17, 3), dtype=np.float32)
        keypoints[:, :2] = kpts[0]
        keypoints[:, 2]  = scores[0]

        return Result_Single_Human_Single_Frame(
            _rtmlib_bbox_to_internal(best_bbox, h, w),
            1.0, keypoints, self.keypoints_format
        )


class HolisticDetector:
    """MediaPipe Holistic：527 个关键点（COCO-17 + 左手21 + 右手21 + 面部468）。"""

    def __init__(self, static_image_mode: bool = True, min_detection_confidence: float = 0.5):
        self.mp_holistic = mp.solutions.holistic
        self.holistic    = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            model_complexity=2,
        )
        logger.info("MediaPipe Holistic model loaded.")

    def detect_single_image(self, image: np.ndarray) -> Result_Single_Human_Single_Frame:
        img_rgb   = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results   = self.holistic.process(img_rgb)
        h, w, _   = image.shape
        keypoints = np.zeros((527, 3), dtype=np.float32)
        flag_valid = False

        if results.pose_landmarks:
            flag_valid = True
            for coco_i, mp_i in enumerate(mp_to_coco_idx):
                lm = results.pose_landmarks.landmark[mp_i]
                keypoints[coco_i] = [lm.x * w, lm.y * h, lm.visibility]
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                keypoints[17 + i] = [lm.x * w, lm.y * h, 0.8]
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                keypoints[38 + i] = [lm.x * w, lm.y * h, 0.8]
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                keypoints[59 + i] = [lm.x * w, lm.y * h, 0.8]

        if not flag_valid:
            return Result_Single_Human_Single_Frame(None, 0.0, None, mp_keypoints_format)

        valid_pts  = keypoints[keypoints[:, 2] > 0.1]
        bbox       = None
        if len(valid_pts) > 0:
            xmin, ymin = np.min(valid_pts[:, :2], axis=0)
            xmax, ymax = np.max(valid_pts[:, :2], axis=0)
            m = 20
            bbox = (int(ymin - m), int(ymax + m), int(xmin - m), int(xmax + m))

        return Result_Single_Human_Single_Frame(bbox, 1.0, keypoints, mp_keypoints_format)


class CombinedDetector:
    """RTMPose（身体精度高）+ MediaPipe Holistic（手部、面部）组合。"""

    def __init__(
        self,
        device: str = 'cpu',
        static_image_mode: bool = True,
        min_detection_confidence: float = 0.2,
    ):
        self.pose_detector     = PoseDetector(device=device)
        self.holistic_detector = HolisticDetector(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
        )

    def detect_single_image(self, image: np.ndarray) -> Result_Single_Human_Single_Frame:
        yolo_out     = self.pose_detector.detect_single_image(image)
        holistic_out = self.holistic_detector.detect_single_image(image)

        if not yolo_out.flag_valid and not holistic_out.flag_valid:
            return Result_Single_Human_Single_Frame(None, 0.0, None, mp_keypoints_format)

        final_keypoints = np.zeros((527, 3), dtype=np.float32)
        bbox = None

        if holistic_out.flag_valid and holistic_out.keypoints is not None:
            final_keypoints[...] = holistic_out.keypoints
            bbox = holistic_out.bbox

        # RTMPose 身体关键点精度更高，覆盖 MediaPipe 身体部分
        if yolo_out.flag_valid and yolo_out.keypoints is not None:
            final_keypoints[:17] = yolo_out.keypoints
            bbox = yolo_out.bbox

        return Result_Single_Human_Single_Frame(bbox, 1.0, final_keypoints, mp_keypoints_format)