"""Detector-free HMR2 initialization from externally supplied xyxy boxes."""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
from pathlib import Path

import numpy as np


def _enable_legacy_numpy_aliases_for_chumpy() -> None:
    """Confine the NumPy 1.26/chumpy compatibility shim to HMR2 startup."""
    aliases = {
        "bool": bool, "int": int, "float": float, "complex": complex,
        "object": object, "unicode": str, "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


@dataclass(frozen=True)
class HMR2Initialization:
    person_index: int
    bbox_xyxy: np.ndarray
    global_orient: np.ndarray       # [1,3,3] rotation matrices
    body_pose: np.ndarray           # [23,3,3] rotation matrices
    betas: np.ndarray               # [10]
    pred_cam: np.ndarray            # [3] weak-perspective crop camera
    pred_cam_t: np.ndarray          # [3] perspective crop translation
    pred_cam_t_full: np.ndarray     # [3] full-image camera translation
    pred_keypoints_2d: np.ndarray   # [J,2], HMR2 normalized crop coordinates
    pred_vertices_shape: tuple[int, ...]
    pred_vertices_min: float
    pred_vertices_max: float


@dataclass(frozen=True)
class HMR2PreparedBatch:
    cpu_image_tensor: "object"
    person_ids: np.ndarray
    boxes_xyxy: np.ndarray
    box_center: np.ndarray
    box_size: np.ndarray
    image_size: np.ndarray


class HMR2Initializer:
    """Load HMR2 once and batch arbitrary external person boxes on CUDA."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda"):
        _enable_legacy_numpy_aliases_for_chumpy()
        import torch

        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("HMR2Initializer requires an available CUDA device")
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        from hmr2.models import load_hmr2

        self.torch = torch
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint
        self.model, self.model_cfg = load_hmr2(str(checkpoint))
        self.model = self.model.to(self.device).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.load_count = 1

    def prepare(self, image_bgr: np.ndarray,
                boxes_xyxy: np.ndarray) -> HMR2PreparedBatch:
        """Run the official ViTDetDataset crop path without any detector."""
        from hmr2.datasets.vitdet_dataset import ViTDetDataset

        image = np.asarray(image_bgr)
        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image_bgr must be HxWx3, got {image.shape}")
        if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) == 0:
            raise ValueError(f"boxes_xyxy must be non-empty [B,4], got {boxes.shape}")
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("boxes_xyxy must be finite with x2>x1 and y2>y1")
        dataset = ViTDetDataset(self.model_cfg, image, boxes, train=False)
        # Upstream ViTDetDataset prints one downsampling factor per item.
        with contextlib.redirect_stdout(io.StringIO()):
            items = [dataset[index] for index in range(len(dataset))]
        tensor = self.torch.from_numpy(np.stack([item["img"] for item in items])).float()
        return HMR2PreparedBatch(
            cpu_image_tensor=tensor,
            person_ids=np.asarray([item["personid"] for item in items], np.int32),
            boxes_xyxy=boxes.copy(),
            box_center=np.stack([item["box_center"] for item in items]).astype(np.float32),
            box_size=np.asarray([item["box_size"] for item in items], np.float32),
            image_size=np.stack([item["img_size"] for item in items]).astype(np.float32),
        )

    def transfer(self, prepared: HMR2PreparedBatch) -> dict:
        torch = self.torch
        return {
            "img": prepared.cpu_image_tensor.to(self.device, non_blocking=False),
            "personid": torch.from_numpy(prepared.person_ids).to(self.device),
            "box_center": torch.from_numpy(prepared.box_center).to(self.device),
            "box_size": torch.from_numpy(prepared.box_size).to(self.device),
            "img_size": torch.from_numpy(prepared.image_size).to(self.device),
        }

    def forward(self, gpu_batch: dict) -> dict:
        with self.torch.inference_mode():
            return self.model(gpu_batch)

    def postprocess(self, prepared: HMR2PreparedBatch,
                    output: dict) -> list[HMR2Initialization]:
        params = output["pred_smpl_params"]
        orient = params["global_orient"].detach().cpu().numpy()
        pose = params["body_pose"].detach().cpu().numpy()
        betas = params["betas"].detach().cpu().numpy()
        cameras = output["pred_cam"].detach().cpu().numpy()
        camera_t = output["pred_cam_t"].detach().cpu().numpy()
        keypoints = output["pred_keypoints_2d"].detach().cpu().numpy()
        vertices = output["pred_vertices"].detach().cpu().numpy()
        results = []
        focal = (float(self.model_cfg.EXTRA.FOCAL_LENGTH)
                 / float(self.model_cfg.MODEL.IMAGE_SIZE)
                 * float(prepared.image_size.max()))
        for index, person_id in enumerate(prepared.person_ids):
            center=prepared.box_center[index]; size=prepared.box_size[index]
            image_size=prepared.image_size[index]; cam=cameras[index]
            denominator=size*cam[0]+1e-9
            full_camera=np.array([
                2*(center[0]-image_size[0]/2)/denominator+cam[1],
                2*(center[1]-image_size[1]/2)/denominator+cam[2],
                2*focal/denominator,
            ],dtype=np.float32)
            results.append(HMR2Initialization(
                person_index=int(person_id), bbox_xyxy=prepared.boxes_xyxy[index].copy(),
                global_orient=orient[index].astype(np.float32),
                body_pose=pose[index].astype(np.float32),
                betas=betas[index].astype(np.float32),
                pred_cam=cameras[index].astype(np.float32),
                pred_cam_t=camera_t[index].astype(np.float32),
                pred_cam_t_full=full_camera,
                pred_keypoints_2d=keypoints[index].astype(np.float32),
                pred_vertices_shape=tuple(vertices[index].shape),
                pred_vertices_min=float(vertices[index].min()),
                pred_vertices_max=float(vertices[index].max()),
            ))
        return results

    def infer(self, image_bgr: np.ndarray,
              boxes_xyxy: np.ndarray) -> list[HMR2Initialization]:
        prepared = self.prepare(image_bgr, boxes_xyxy)
        gpu_batch = self.transfer(prepared)
        output = self.forward(gpu_batch)
        return self.postprocess(prepared, output)
