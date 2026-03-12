"""
PersonSegmenter — 基于 YOLO11 (检测) + SAM2 (精细分割) 的人像分割模块。
与 src/lib_pano/human/detect_mask.py 使用相同的模型组合，
但提供面向内存 numpy 数组的 API，避免文件 I/O 开销。
"""

from typing import Optional, Tuple
import numpy as np
import cv2
import os

# Ultralytics 统一接口
try:
    from ultralytics import YOLO, SAM
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False


def _get_model_path(model_name: str) -> str:
    """
    优先从工程根目录（detect_mask.py 同级两层上的根）查找模型权重，
    找不到则返回原名称（Ultralytics 会自动下载）。
    """
    # 尝试从典型的工程根路径查找
    candidate_dirs = [
        os.path.dirname(os.path.abspath(__file__)),   # 当前文件所在目录
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'),  # 工程根
    ]
    for d in candidate_dirs:
        full_path = os.path.normpath(os.path.join(d, model_name))
        if os.path.isfile(full_path):
            return full_path
    return model_name  # 回退：交由 Ultralytics 自动下载


class PersonSegmenter:
    """
    人像分割器：YOLO11 负责快速检测人体边界框，SAM2 据此生成精细掩码。
    与 detect_mask.run_sam2_human_segmentation 使用完全相同的模型组合。

    Args:
        yolo_model (str): YOLO 检测模型名称，默认 'yolo11l.pt'。
        sam_model  (str): SAM2 模型名称，默认 'sam2_b.pt'。
        device     (str): 'cuda' / 'cpu'，None 则自动选择。
    """

    def __init__(
        self,
        yolo_model: str = 'yolo11l.pt',
        sam_model: str = 'sam2_b.pt',
        device: Optional[str] = None,
    ):
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "[PersonSegmenter] 未找到 ultralytics 包。请运行: pip install ultralytics"
            )

        self.device = device  # None 时 Ultralytics 会自动选 GPU/CPU

        yolo_path = _get_model_path(yolo_model)
        sam_path  = _get_model_path(sam_model)

        print(f'[PersonSegmenter] 加载 YOLO 检测模型: {yolo_path}')
        self.det_model = YOLO(yolo_path)

        print(f'[PersonSegmenter] 加载 SAM2 分割模型: {sam_path}')
        self.sam_model = SAM(sam_path)

        if self.device:
            self.det_model.to(self.device)
            self.sam_model.to(self.device)

        print('[PersonSegmenter] 初始化完成。')

    def get_mask(
        self,
        img_bgr: np.ndarray,
        threshold: float = 0.5,
        dilate_pixels: int = 0,
        conf: float = 0.3,
    ) -> np.ndarray:
        """
        输入 BGR 图像，返回人像 Mask (numpy uint8, 0 or 255)。

        Args:
            img_bgr      : BGR 格式的 numpy 图像。
            threshold    : SAM2 掩码二值化阈值（0~1）。
            dilate_pixels: 膨胀像素数，用于覆盖手持物品（如手机）。
            conf         : YOLO 检测置信度阈值。
        Returns:
            mask: np.ndarray, shape=(H, W), dtype=uint8, 值为 0 或 255。
        """
        h, w = img_bgr.shape[:2]
        mask_combined = np.zeros((h, w), dtype=np.uint8)

        # ── Step 1: YOLO 检测人体，获取边界框 ──────────────────────────────
        # Ultralytics 接受 BGR numpy 数组（内部会自动转换）
        det_results = self.det_model(img_bgr, classes=[0], conf=conf, verbose=False)

        if (
            not det_results
            or det_results[0].boxes is None
            or len(det_results[0].boxes) == 0
        ):
            # 未检测到人，返回全零掩码
            return mask_combined

        boxes = det_results[0].boxes.xyxy.cpu().numpy().tolist()  # [[x1,y1,x2,y2], ...]

        # ── Step 2: SAM2 根据边界框生成精细掩码 ─────────────────────────────
        try:
            sam_results = self.sam_model(img_bgr, bboxes=boxes, verbose=False)
        except TypeError:
            # 兼容旧版本接口
            sam_results = self.sam_model(img_bgr, verbose=False)

        if not sam_results or sam_results[0].masks is None:
            return mask_combined

        masks_tensor = sam_results[0].masks.data.cpu().numpy()  # (N, H', W')

        for mask in masks_tensor:
            # 缩放至原图尺寸
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_binary = (mask > threshold).astype(np.uint8) * 255
            mask_combined = np.maximum(mask_combined, mask_binary)

        # ── Step 3: 可选膨胀 ─────────────────────────────────────────────────
        if dilate_pixels > 0:
            kernel_size = int(dilate_pixels)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask_combined = cv2.dilate(mask_combined, kernel, iterations=1)

        return mask_combined

    def mask_out_person(
        self,
        img_bgr: np.ndarray,
        fill_value: Tuple[int, int, int] = (0, 0, 0),
        dilate_pixels: int = 0,
        threshold: float = 0.5,
        conf: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将图片中的人像区域涂抹为指定颜色。

        Returns:
            (img_masked, mask): 涂抹后的图像 和 二值 Mask。
        """
        mask = self.get_mask(img_bgr, threshold=threshold, dilate_pixels=dilate_pixels, conf=conf)
        img_masked = img_bgr.copy()
        img_masked[mask == 255] = fill_value
        return img_masked, mask


# ── 全局单例 ────────────────────────────────────────────────────────────────
_segmenter: Optional[PersonSegmenter] = None


def get_person_mask(
    img_bgr: np.ndarray,
    device: Optional[str] = None,
    threshold: float = 0.5,
    dilate_pixels: int = 0,
    conf: float = 0.3,
    yolo_model: str = 'yolo11l.pt',
    sam_model: str = 'sam2_b.pt',
) -> np.ndarray:
    """
    便捷函数：获取人像二值掩码（0/255）。
    使用与 detect_mask.run_sam2_human_segmentation 相同的 YOLO11+SAM2 流程。
    """
    global _segmenter
    if _segmenter is None:
        _segmenter = PersonSegmenter(yolo_model=yolo_model, sam_model=sam_model, device=device)
    return _segmenter.get_mask(img_bgr, threshold=threshold, dilate_pixels=dilate_pixels, conf=conf)


def remove_person(
    img_bgr: np.ndarray,
    device: Optional[str] = None,
    fill_value: Tuple[int, int, int] = (0, 0, 0),
    dilate_pixels: int = 0,
    threshold: float = 0.5,
    conf: float = 0.3,
    yolo_model: str = 'yolo11l.pt',
    sam_model: str = 'sam2_b.pt',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    便捷函数：将图像中的人像区域替换为指定颜色并返回 (masked_img, mask)。

    Args:
        img_bgr      : BGR numpy 图像。
        device       : 'cuda' / 'cpu' / None（自动）。
        fill_value   : 涂抹颜色（BGR），默认黑色 (0,0,0)。
        dilate_pixels: 膨胀像素数，用于覆盖手持物（如手机）。0 表示不膨胀。
        threshold    : SAM2 掩码二值化阈值（0~1）。
        conf         : YOLO 检测置信度阈值。
        yolo_model   : YOLO 模型权重名称，默认 'yolo11l.pt'。
        sam_model    : SAM2 模型权重名称，默认 'sam2_b.pt'。
    Returns:
        (img_masked, mask): 处理后图像 和 uint8 二值掩码。
    """
    global _segmenter
    if _segmenter is None:
        _segmenter = PersonSegmenter(yolo_model=yolo_model, sam_model=sam_model, device=device)
    return _segmenter.mask_out_person(
        img_bgr,
        fill_value=fill_value,
        dilate_pixels=dilate_pixels,
        threshold=threshold,
        conf=conf,
    )
