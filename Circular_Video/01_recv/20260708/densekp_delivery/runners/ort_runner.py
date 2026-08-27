"""ONNX Runtime DenseKP runner — runs inference via CameraHMR deploy/demo_keypoints_ort logic."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_CAMERAHMR_DIR = _THIS_DIR.parent.parent / "CameraHMR"
if str(_CAMERAHMR_DIR) not in sys.path:
    sys.path.insert(0, str(_CAMERAHMR_DIR))

from .base import DenseKPPerson, DenseKPResult

IMAGE_SIZE = 256
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


class ORTRunner:
    def __init__(
        self,
        onnx_path: str,
        device: str = "cuda",
        detector_backend: str = "rtmlib_yolox",
        detector_threshold: float = 0.25,
        batch_size: int = 1,
    ):
        self.onnx_path = str(onnx_path)
        self.device = device
        self.detector_backend = detector_backend
        self.detector_threshold = detector_threshold
        self.batch_size = batch_size

        self._session = None
        self._detector = None
        self._ort_in_name = None
        self._ort_out_name = None
        self.profile_timing: dict[str, float] = {
            "ort_init_ms": 0.0,
            "detector_init_ms": 0.0,
            "detector_inference_ms": 0.0,
            "detector_preprocess_ms": 0.0,
            "detector_ort_ms": 0.0,
            "detector_postprocess_ms": 0.0,
            "bbox_postprocess_ms": 0.0,
            "densekp_preprocess_ms": 0.0,
            "densekp_crop_resize_ms": 0.0,
            "densekp_normalize_ms": 0.0,
            "densekp_tensor_prepare_ms": 0.0,
            "densekp_inference_ms": 0.0,
            "densekp_postprocess_ms": 0.0,
            "densekp_ort_call_count": 0.0,
            "densekp_ort_batch_item_count": 0.0,
        }
        self.session_creation_count = 0
        self.detector_initialization_count = 0

    def _lazy_init(self):
        if self._session is not None:
            return

        import onnxruntime as ort
        from detectors import build_person_detector

        available_providers = ort.get_available_providers()
        if self.device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "cuda":
            print("[ort] CUDA not available, using CPU")
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        start = time.perf_counter()
        self._session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.profile_timing["ort_init_ms"] += (
            time.perf_counter() - start
        ) * 1000.0
        self.session_creation_count += 1
        self._ort_in_name = self._session.get_inputs()[0].name
        self._ort_out_name = self._session.get_outputs()[0].name

        start = time.perf_counter()
        self._detector = build_person_detector(
            backend=self.detector_backend,
            threshold=self.detector_threshold,
            device=self.device,
        )
        self.profile_timing["detector_init_ms"] += (
            time.perf_counter() - start
        ) * 1000.0
        self.detector_initialization_count += 1

    def load(self) -> "ORTRunner":
        """Create the ORT session and detector once."""
        self._lazy_init()
        return self

    @property
    def loaded(self) -> bool:
        return self._session is not None and self._detector is not None

    @property
    def execution_providers(self) -> list[str]:
        return [] if self._session is None else list(self._session.get_providers())

    def close(self) -> None:
        """Release references owned by this runner."""
        self._detector = None
        self._session = None
        self._ort_in_name = None
        self._ort_out_name = None

    def run(self, image_bgr: np.ndarray, image_path: str, camera_name: str = "") -> DenseKPResult:
        self._lazy_init()
        h, w = image_bgr.shape[:2]

        start = time.perf_counter()
        detector_before = dict(getattr(self._detector, "profile_timing", {}))
        detections = self._detector.detect(image_bgr)
        self.profile_timing["detector_inference_ms"] += (
            time.perf_counter() - start
        ) * 1000.0
        detector_after = getattr(self._detector, "profile_timing", {})
        for source, target in (
            ("preprocess_ms", "detector_preprocess_ms"),
            ("inference_ms", "detector_ort_ms"),
            ("postprocess_ms", "detector_postprocess_ms"),
        ):
            self.profile_timing[target] += float(detector_after.get(source, 0.0)) - float(
                detector_before.get(source, 0.0))
        return self._run_single_stage(image_bgr, detections, image_path, camera_name, h, w)

    def run_batch(
        self,
        items: list[tuple[np.ndarray, str, str]],
        densekp_batch_size: int | None = None,
    ) -> list[DenseKPResult]:
        """Run serial B=1 detection, then one global DenseKP crop queue.

        The shipped detector graph has a fixed batch dimension of one.  This
        method therefore deliberately preserves its production inference and
        only batches the dynamic-batch DenseKP graph across images.  The tuple
        index is the restoration key, so unequal/empty results cannot reorder
        frame, camera, or ROI metadata.
        """
        self._lazy_init()
        detections_per_item = []
        for image, _, _ in items:
            before = dict(getattr(self._detector, "profile_timing", {}))
            started = time.perf_counter()
            detections_per_item.append(self._detector.detect(image))
            self.profile_timing["detector_inference_ms"] += (
                time.perf_counter() - started) * 1000.0
            after = getattr(self._detector, "profile_timing", {})
            for source, target in (
                ("preprocess_ms", "detector_preprocess_ms"),
                ("inference_ms", "detector_ort_ms"),
                ("postprocess_ms", "detector_postprocess_ms"),
            ):
                self.profile_timing[target] += float(after.get(source, 0.0)) - float(
                    before.get(source, 0.0))
        return self._run_detected_batch(
            items, detections_per_item,
            self.batch_size if densekp_batch_size is None else densekp_batch_size,
        )

    def run_bboxes(
        self,
        items: list[tuple[np.ndarray, str, str, list[list[float]]]],
        densekp_batch_size: int = 1,
    ) -> list[DenseKPResult]:
        """Run DenseKP directly from externally supplied xyxy boxes.

        This is the track-guided inference primitive.  It deliberately knows
        nothing about tracks or temporal policy and never consults detector
        scores.  Each box is converted to the same detection dictionary used
        by the production detector path, then enters identical preprocessing.
        """
        self._lazy_init()
        runner_items = [(image, path, camera) for image, path, camera, _ in items]
        detections = [[{
            "bbox_xyxy": np.asarray(box, dtype=np.float32),
            "score": 1.0,
        } for box in boxes] for _, _, _, boxes in items]
        return self._run_detected_batch(
            runner_items, detections, densekp_batch_size)

    def _run_detected_batch(self, items, detections_per_item, batch_size):
        persons_per_item: list[list[DenseKPPerson]] = [[] for _ in items]
        jobs = []
        for item_index, ((image, _, _), detections) in enumerate(
            zip(items, detections_per_item)
        ):
            if not detections:
                continue
            started = time.perf_counter()
            boxes = np.stack([d["bbox_xyxy"] for d in detections]).astype(np.float32)
            scales = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
            centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
            self.profile_timing["bbox_postprocess_ms"] += (
                time.perf_counter() - started) * 1000.0
            for person_index in range(len(detections)):
                jobs.append((item_index, person_index, image, detections[person_index],
                             boxes[person_index], centers[person_index], scales[person_index]))

        prepared = []
        mean = 255.0 * np.asarray(IMAGE_MEAN, np.float32)
        std = 255.0 * np.asarray(IMAGE_STD, np.float32)
        for job in jobs:
            _, _, image, _, _, center, scale = job
            bbox_size = float(_expand_to_aspect_ratio(
                scale * 200, target_aspect_ratio=None).max())
            started = time.perf_counter()
            patch, _ = _generate_image_patch_cv2(
                image, float(center[0]), float(center[1]), bbox_size, bbox_size,
                IMAGE_SIZE, IMAGE_SIZE, False, 1.0, 0)
            self.profile_timing["densekp_crop_resize_ms"] += (
                time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            array = _convert_cvimg_to_tensor(patch[:, :, ::-1])
            self.profile_timing["densekp_tensor_prepare_ms"] += (
                time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            # Preserve the production B1 operation order exactly.  Vectorized
            # broadcasting is shorter but can change float32 rounding before
            # ORT, obscuring the actual batch-inference parity measurement.
            for channel in range(min(image.shape[2], 3)):
                array[channel] = (
                    array[channel] - mean[channel]) / std[channel]
            self.profile_timing["densekp_normalize_ms"] += (
                time.perf_counter() - started) * 1000.0
            prepared.append((job, bbox_size, array))

        batch_size = max(1, int(batch_size))
        for start_index in range(0, len(prepared), batch_size):
            chunk = prepared[start_index:start_index + batch_size]
            started = time.perf_counter()
            tensor = np.stack([entry[2] for entry in chunk]).astype(np.float32)
            self.profile_timing["densekp_tensor_prepare_ms"] += (
                time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            output = self._session.run(
                [self._ort_out_name], {self._ort_in_name: tensor})[0]
            self.profile_timing["densekp_ort_call_count"] += 1
            self.profile_timing["densekp_ort_batch_item_count"] += len(chunk)
            self.profile_timing["densekp_inference_ms"] += (
                time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            for local_index, (job, bbox_size, _) in enumerate(chunk):
                item_index, person_index, _, detection, box, center, _ = job
                prediction = output[local_index]
                crop_xy = (prediction[:, :2] + 0.5) * IMAGE_SIZE
                keypoints = _crop_xy_to_orig_xy_by_affine(
                    crop_xy, center, bbox_size)
                persons_per_item[item_index].append(DenseKPPerson(
                    person_index=person_index, bbox=box.tolist(),
                    keypoints=keypoints,
                    raw_scores=prediction[:, 2].astype(np.float32),
                    detection_score=float(detection.get("score", 0.0)),
                    source="ort",
                ))
            self.profile_timing["densekp_postprocess_ms"] += (
                time.perf_counter() - started) * 1000.0

        results = []
        for (image, image_path, camera_name), persons in zip(items, persons_per_item):
            height, width = image.shape[:2]
            results.append(DenseKPResult(
                image_path=image_path, image_width=width, image_height=height,
                persons=persons, camera_name=camera_name,
                error_message=None if persons else "No persons processed",
            ))
        return results

    def _run_single_stage(self, image_bgr, detections, image_path, camera_name, h, w):
        persons = []
        if len(detections) == 0:
            return DenseKPResult(
                image_path=image_path, image_width=w, image_height=h,
                persons=[], camera_name=camera_name,
            )

        start = time.perf_counter()
        boxes = np.stack([d["bbox_xyxy"] for d in detections]).astype(np.float32)
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.profile_timing["bbox_postprocess_ms"] += (
            time.perf_counter() - start
        ) * 1000.0

        batches = iter(_build_densekp_batches(
            image_bgr, bbox_center, bbox_scale, self.batch_size,
            profile_timing=self.profile_timing,
        ))
        while True:
            start = time.perf_counter()
            try:
                batch_idx, batch = next(batches)
            except StopIteration:
                break
            self.profile_timing["densekp_preprocess_ms"] += (
                time.perf_counter() - start
            ) * 1000.0
            start = time.perf_counter()
            ort_out = self._session.run([self._ort_out_name], {self._ort_in_name: batch['img']})[0]
            self.profile_timing["densekp_ort_call_count"] += 1
            self.profile_timing["densekp_ort_batch_item_count"] += batch['img'].shape[0]
            self.profile_timing["densekp_inference_ms"] += (
                time.perf_counter() - start
            ) * 1000.0

            start = time.perf_counter()
            for ind in range(batch['img'].shape[0]):
                global_ind = batch_idx * self.batch_size + ind
                pred_keypoints = ort_out[ind]
                batch_box_center = batch['box_centers'][ind]
                batch_box_size = float(batch['box_sizes'][ind])

                keypoints_crop_xy = (pred_keypoints[:, :2] + 0.5) * IMAGE_SIZE
                keypoints_orig = _crop_xy_to_orig_xy_by_affine(
                    keypoints_crop_xy, batch_box_center, batch_box_size,
                )
                raw_scores = pred_keypoints[:, 2].astype(np.float32)
                bbox = boxes[global_ind].tolist()
                det_score = float(detections[global_ind].get("score", 0.0))

                persons.append(DenseKPPerson(
                    person_index=global_ind,
                    bbox=bbox,
                    keypoints=keypoints_orig,
                    raw_scores=raw_scores,
                    detection_score=det_score,
                    source="ort",
                ))
            self.profile_timing["densekp_postprocess_ms"] += (
                time.perf_counter() - start
            ) * 1000.0

        return DenseKPResult(
            image_path=image_path, image_width=w, image_height=h,
            persons=persons, camera_name=camera_name,
            error_message=None if persons else "No persons processed",
        )

def _rotate_2d(pt_2d, rot_rad):
    x, y = float(pt_2d[0]), float(pt_2d[1])
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([x * cs - y * sn, x * sn + y * cs], dtype=np.float32)


def _gen_trans_from_patch_cv(c_x, c_y, src_width, src_height,
                             dst_width, dst_height, scale, rot):
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    rot_rad = np.pi * rot / 180.0
    src_downdir = _rotate_2d(np.array([0.0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = _rotate_2d(np.array([src_w * 0.5, 0.0], dtype=np.float32), rot_rad)
    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_downdir = np.array([0.0, dst_height * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0.0], dtype=np.float32)
    src = np.float32([src_center, src_center + src_downdir, src_center + src_rightdir])
    dst = np.float32([dst_center, dst_center + dst_downdir, dst_center + dst_rightdir])
    return cv2.getAffineTransform(src, dst)


def _generate_image_patch_cv2(img, c_x, c_y, bb_width, bb_height,
                              patch_width, patch_height, do_flip, scale, rot):
    img_height, img_width, img_channels = img.shape
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1.0
    trans = _gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height,
                                     patch_width, patch_height, scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
    return img_patch, trans


def _convert_cvimg_to_tensor(cvimg):
    return np.ascontiguousarray(cvimg.transpose(2, 0, 1)).astype(np.float32)


def _expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    if target_aspect_ratio is None:
        return input_shape
    try:
        w, h = input_shape
    except (ValueError, TypeError):
        return input_shape
    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    return np.array([w_new, h_new])


def _trans_points2d_parallel(keypoints_2d, trans):
    ones = np.ones((keypoints_2d.shape[0], 1))
    keypoints_aug = np.hstack([keypoints_2d, ones])
    transformed = np.dot(trans, keypoints_aug.T).T
    return transformed[:, :2]


def _crop_xy_to_orig_xy_by_affine(keypoints_crop_xy, bbox_center, box_size,
                                   crop_size=IMAGE_SIZE):
    center_x, center_y = float(bbox_center[0]), float(bbox_center[1])
    box_size = float(box_size)
    trans = _gen_trans_from_patch_cv(
        center_x, center_y, box_size, box_size,
        float(crop_size), float(crop_size), 1.0, 0.0)
    inv_trans = cv2.invertAffineTransform(trans)
    keypoints_orig_xy = _trans_points2d_parallel(keypoints_crop_xy, inv_trans)
    return keypoints_orig_xy.astype(np.float32)


def _build_densekp_batches(
    base_img, bbox_center, bbox_scale, batch_size, profile_timing=None,
):
    N = len(bbox_center)
    mean = 255.0 * np.array(IMAGE_MEAN, dtype=np.float32)
    std = 255.0 * np.array(IMAGE_STD, dtype=np.float32)

    all_imgs = [None] * N
    all_box_centers = [None] * N
    all_box_sizes = [None] * N

    for i in range(N):
        cx, cy = float(bbox_center[i][0]), float(bbox_center[i][1])
        scale = bbox_scale[i]
        bbox_size = float(_expand_to_aspect_ratio(scale * 200, target_aspect_ratio=None).max())
        start = time.perf_counter()
        patch, _ = _generate_image_patch_cv2(
            base_img, cx, cy, bbox_size, bbox_size,
            IMAGE_SIZE, IMAGE_SIZE, False, 1.0, 0)
        if profile_timing is not None:
            profile_timing["densekp_crop_resize_ms"] += (
                time.perf_counter() - start
            ) * 1000.0
        start = time.perf_counter()
        arr = _convert_cvimg_to_tensor(patch[:, :, ::-1])
        if profile_timing is not None:
            profile_timing["densekp_tensor_prepare_ms"] += (
                time.perf_counter() - start
            ) * 1000.0
        start = time.perf_counter()
        for c in range(min(base_img.shape[2], 3)):
            arr[c] = (arr[c] - mean[c]) / std[c]
        if profile_timing is not None:
            profile_timing["densekp_normalize_ms"] += (
                time.perf_counter() - start
            ) * 1000.0
        all_imgs[i] = arr
        all_box_centers[i] = bbox_center[i].copy()
        all_box_sizes[i] = bbox_size

    for batch_idx, start in enumerate(range(0, N, batch_size)):
        end = min(start + batch_size, N)
        timing_start = time.perf_counter()
        batch = {
            'img': np.stack(all_imgs[start:end], axis=0).astype(np.float32),
            'box_centers': np.stack(all_box_centers[start:end], axis=0).astype(np.float32),
            'box_sizes': np.array(all_box_sizes[start:end], dtype=np.float32),
        }
        if profile_timing is not None:
            profile_timing["densekp_tensor_prepare_ms"] += (
                time.perf_counter() - timing_start
            ) * 1000.0
        yield batch_idx, batch
