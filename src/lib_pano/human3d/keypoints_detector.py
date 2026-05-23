import numpy as np
import logging
import cv2 as cv
from typing import Dict, Tuple, Optional
from lib_pano.human3d.skeletion_constant import (
    COCO_17_KEYPOINTS, COCO_17_SKELETON_TOPOLOGY,
    COCO_133_KEYPOINTS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BBOX_YMAX_RATIO = 0.9  


def _rtmlib_bbox_to_internal(
    bbox: np.ndarray,   # [xmin, ymin, xmax, ymax]
    h: int,
    w: int,
) -> Tuple[int, int, int, int]:
    """rtmlib bbox [xmin,ymin,xmax,ymax] éˆ«?éå‘´å„´éç…Žç´¡ (ymin,ymax,xmin,xmax)é”›å®‘lip é’æ¿æµ˜éå¿šå¯–é¥æ·¬â‚¬?"""
    xmin = max(0,     int(bbox[0]))
    ymin = max(0,     int(bbox[1]))
    xmax = min(w - 1, int(bbox[2]))
    ymax = min(h - 1, int(bbox[3]))
    return (ymin, ymax, xmin, xmax)


def draw_keypoints(
    image: np.ndarray,
    result,
    conf_thresh: float = 0.3,
    draw_face: bool = False,
    draw_hands: bool = False,
    is_fitted: bool = False,
):
    
    img_draw = image.copy()

    if not result.flag_valid or result.keypoints is None:
        return image
    kpts = result.keypoints   # (N, 3)
    if result.bbox is not None:
        ymin, ymax, xmin, xmax = result.bbox
        cv.rectangle(img_draw, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    line_color = (255, 0, 255) if is_fitted else (0, 255, 255)
    dot_color  = (0, 255, 255) if is_fitted else (0, 0, 255)
    text_color = (0, 0, 255)   if is_fitted else (255, 0, 0)

    
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
                
    if draw_face and kpts.shape[0] >= 91:
        for i in range(23, 91):
            x, y, conf = kpts[i]
            if conf > conf_thresh:
                cv.circle(img_draw, (int(x), int(y)), 1, (255, 0, 0), -1)

    return img_draw


def draw_bbox(
    image: np.ndarray,
    result: "Result_Single_Human_Single_Frame",
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """在图像上绘制 bbox，result 无效或 bbox 为 None 时返回原图。"""
    if not result.flag_valid or result.bbox is None:
        return image
    img_draw = image.copy()
    ymin, ymax, xmin, xmax = result.bbox
    cv.rectangle(img_draw, (xmin, ymin), (xmax, ymax), color, thickness)
    return img_draw



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


class WholePoseDetector:
    

    def __init__(self, device: str = 'cpu'):
        device_str = 'cuda' if 'cuda' in str(device) else 'cpu'
        logger.info(f"Loading RTMLib Wholebody model on {device_str}...")
        from rtmlib import Wholebody
        self.body_model       = Wholebody(device=device_str)
        self.keypoints_format = COCO_133_KEYPOINTS

    def _detect_impl(
        self,
        image: np.ndarray,
        mask=None,
    ) -> Result_Single_Human_Single_Frame:
        if mask is not None:
            masked_image = image.copy()
            masked_image[mask == False] = 0
        else:
            masked_image = image

        bboxes = self.body_model.det_model(masked_image)
        if len(bboxes) == 0:
            return Result_Single_Human_Single_Frame(
                None, 0.0, None, self.keypoints_format
            )

        h, w      = image.shape[:2]
        ymax_thresh = h * BBOX_YMAX_RATIO
        filtered_bboxes = bboxes[bboxes[:, 3] <= ymax_thresh]
        if len(filtered_bboxes) == 0:
            return Result_Single_Human_Single_Frame(
                None, 0.0, None, self.keypoints_format
            )

        areas = (filtered_bboxes[:, 2] - filtered_bboxes[:, 0]) * (filtered_bboxes[:, 3] - filtered_bboxes[:, 1])
        best_bbox = filtered_bboxes[np.argmax(areas)]

        kpts, scores = self.body_model.pose_model(
            image, bboxes=np.array([best_bbox])
        )
        
        scores_flat = np.array(scores[0]).reshape(133)

        keypoints        = np.zeros((133, 3), dtype=np.float32)
        keypoints[:, :2] = kpts[0]       # (133, 2)
        keypoints[:, 2]  = scores_flat   # (133,)

        return Result_Single_Human_Single_Frame(
            _rtmlib_bbox_to_internal(best_bbox, h, w),
            1.0, keypoints, self.keypoints_format
        )

    def detect_single_image(self, image: np.ndarray) -> Result_Single_Human_Single_Frame:
        return self._detect_impl(image, mask=None)

    def detect_single_image_with_bbox(
        self,
        image: np.ndarray,
        bbox_xyxy: np.ndarray | list[float] | tuple[float, float, float, float],
    ) -> Result_Single_Human_Single_Frame:
        h, w = image.shape[:2]
        xmin = max(0, min(w - 1, int(round(bbox_xyxy[0]))))
        ymin = max(0, min(h - 1, int(round(bbox_xyxy[1]))))
        xmax = max(0, min(w - 1, int(round(bbox_xyxy[2]))))
        ymax = max(0, min(h - 1, int(round(bbox_xyxy[3]))))

        if xmax <= xmin or ymax <= ymin:
            return Result_Single_Human_Single_Frame(
                None, 0.0, None, self.keypoints_format
            )

        best_bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        kpts, scores = self.body_model.pose_model(
            image,
            bboxes=np.array([best_bbox], dtype=np.float32),
        )

        if kpts is None or scores is None or len(kpts) == 0 or len(scores) == 0:
            return Result_Single_Human_Single_Frame(
                None, 0.0, None, self.keypoints_format
            )

        scores_flat = np.array(scores[0]).reshape(133)

        keypoints = np.zeros((133, 3), dtype=np.float32)
        keypoints[:, :2] = kpts[0]
        keypoints[:, 2] = scores_flat

        return Result_Single_Human_Single_Frame(
            _rtmlib_bbox_to_internal(best_bbox, h, w),
            1.0, keypoints, self.keypoints_format
        )

    def detect_single_image_with_fov_mask(
        self,
        image: np.ndarray,
        fov_mask,
    ) -> Result_Single_Human_Single_Frame:
        return self._detect_impl(image, mask=fov_mask)
