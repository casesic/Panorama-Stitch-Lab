"""PyTorch DenseKP runner — runs inference via CameraHMR demo_keypoints logic."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_CAMERAHMR_DIR = _THIS_DIR.parent.parent / "CameraHMR"
if str(_CAMERAHMR_DIR) not in sys.path:
    sys.path.insert(0, str(_CAMERAHMR_DIR))

from .base import DenseKPPerson, DenseKPResult


class PyTorchRunner:
    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        detector_backend: str = "rtmlib_yolox",
        detector_threshold: float = 0.25,
        batch_size: int = 1,
    ):
        self.checkpoint = str(checkpoint)
        self.device = device
        self.detector_backend = detector_backend
        self.detector_threshold = detector_threshold
        self.batch_size = batch_size

        self._model = None
        self._detector = None

    def _lazy_init(self):
        if self._model is not None:
            return

        import torch
        from core.densekp_trainer import DenseKP
        from detectors import build_person_detector

        self._torch_device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self._model = DenseKP.load_from_checkpoint(
            self.checkpoint, strict=False, map_location=self._torch_device
        ).to(self._torch_device).eval()

        self._detector = build_person_detector(
            backend=self.detector_backend,
            threshold=self.detector_threshold,
            device=self.device,
        )

    def load(self) -> "PyTorchRunner":
        self._lazy_init()
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._detector is not None

    def close(self) -> None:
        self._detector = None
        self._model = None

    def run(self, image_bgr: np.ndarray, image_path: str, camera_name: str = "") -> DenseKPResult:
        self._lazy_init()
        h, w = image_bgr.shape[:2]

        detections = self._detector.detect(image_bgr)
        return self._run_single_stage(image_bgr, detections, image_path, camera_name, h, w)

    def _run_single_stage(self, image_bgr, detections, image_path, camera_name, h, w):
        from core.datasets.dataset import Dataset
        from core.densekp_trainer import DenseKP

        persons = []
        for idx, det in enumerate(detections):
            try:
                bbox = det["bbox_xyxy"].tolist()
                det_score = float(det.get("score", 0.0))

                boxes = np.array([bbox]).astype(np.float32)
                bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
                bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

                import torch
                from core.utils.train_utils import denormalize_images
                from core.datasets.utils import gen_trans_from_patch_cv, trans_points2d_parallel
                from core.constants import IMAGE_SIZE

                dataset = Dataset(image_bgr, bbox_center, bbox_scale, None, False, image_path)
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=1, shuffle=False, num_workers=0,
                )

                for batch in dataloader:
                    from core.utils import recursive_to
                    batch = recursive_to(batch, self._torch_device)
                    with torch.no_grad():
                        out = self._model(batch)

                    pred_keypoints = out['pred_keypoints'][0].detach().cpu().numpy()

                    batch_box_center = batch["box_center"][0].detach().cpu().numpy()
                    batch_box_size = float(batch["box_size"][0].detach().cpu().numpy())

                    keypoints_crop_xy = (pred_keypoints[:, :2] + 0.5) * IMAGE_SIZE
                    trans = gen_trans_from_patch_cv(
                        float(batch_box_center[0]), float(batch_box_center[1]),
                        batch_box_size, batch_box_size,
                        float(IMAGE_SIZE), float(IMAGE_SIZE), 1.0, 0.0,
                    )
                    inv_trans = cv2.invertAffineTransform(trans)
                    keypoints_orig = trans_points2d_parallel(keypoints_crop_xy, inv_trans).astype(np.float32)
                    raw_scores = pred_keypoints[:, 2].astype(np.float32)

                    persons.append(DenseKPPerson(
                        person_index=idx,
                        bbox=bbox,
                        keypoints=keypoints_orig,
                        raw_scores=raw_scores,
                        detection_score=det_score,
                        source="pytorch",
                    ))
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("DenseKP inference failed for person %d: %s", idx, e)

        return DenseKPResult(
            image_path=image_path,
            image_width=w,
            image_height=h,
            persons=persons,
            camera_name=camera_name,
            error_message=None if persons else "No persons processed",
        )
