from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

from batch_utils import load_reviewed_entries


MASK_ALPHA = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment reviewed ERP crops with SAM3 and save mask, RGBA, and visualization outputs."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Local SAM3 model directory OR huggingface model id, e.g. facebook/sam3",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input root directory containing cam2/ and cam4/ subdirectories",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory")
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json")
    parser.add_argument("--text", type=str, default="person", help='Text prompt, default "person"')
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Mask threshold")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N reviewed frames; 0 means process all",
    )
    return parser.parse_args()


def _is_git_lfs_pointer(file_path: Path) -> bool:
    if not file_path.exists() or file_path.stat().st_size > 4096:
        return False
    with file_path.open("rb") as file:
        head = file.read(256)
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _validate_local_model_dir(model_dir: Path) -> Optional[str]:
    if not model_dir.exists():
        return f"Model directory not found: {model_dir}"

    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists() and _is_git_lfs_pointer(safetensors_path):
        return (
            f"Invalid local model file: {safetensors_path} is a Git LFS pointer, not real weights.\n"
            "Fix options:\n"
            "1) Download real weights into this folder (replace model.safetensors), or\n"
            "2) Run with --model_dir facebook/sam3 to load from Hugging Face directly."
        )

    # Transformers Sam3Model cannot load complex dtype tensors from safetensors.
    # This happens if model.safetensors was naively converted from official sam3.pt.
    if safetensors_path.exists():
        try:
            with safe_open(str(safetensors_path), framework="pt") as file:
                for key in file.keys():
                    if str(file.get_slice(key).get_dtype()).startswith("C"):
                        return (
                            f"Incompatible local model file: {safetensors_path} contains complex dtype tensors.\n"
                            "This is usually caused by converting official sam3.pt directly and is not"
                            " compatible with transformers Sam3Model.\n"
                            "Use an official Hugging Face converted checkpoint for facebook/sam3"
                            " (or load via original SAM3 codebase instead of transformers)."
                        )
        except Exception:
            pass
    return None


def render_segmentation(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    all_masks: list[np.ndarray] | None = None,
    positive_points: list[list[int]] | None = None,
    alpha: float = MASK_ALPHA,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    seg = image_bgr.copy()
    mask_palette = [
        (0, 120, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 180, 0),
        (180, 0, 255),
        (0, 255, 255),
    ]
    masks_to_draw = all_masks if all_masks else [mask]
    for index, mask_item in enumerate(masks_to_draw):
        if mask_item.shape != (h, w):
            mask_item = cv2.resize(
                mask_item.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        overlay = seg.copy()
        overlay[mask_item] = mask_palette[index % len(mask_palette)]
        seg = cv2.addWeighted(overlay, alpha, seg, 1 - alpha, 0)

    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg, contours, -1, (0, 255, 255), 2)
    if positive_points is not None:
        for x, y in positive_points:
            cv2.circle(seg, (int(x), int(y)), 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(seg, (int(x), int(y)), 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return seg


def load_sam3(model_ref: str) -> tuple[Sam3Model, Sam3Processor, str]:
    model_dir = Path(model_ref)

    if model_dir.exists():
        local_error = _validate_local_model_dir(model_dir)
        if local_error is not None:
            raise RuntimeError(local_error)
        load_ref = str(model_dir)
    else:
        load_ref = model_ref

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading SAM3 model from: {load_ref}")
    model = Sam3Model.from_pretrained(load_ref).to(device)
    processor = Sam3Processor.from_pretrained(load_ref)
    return model, processor, device


def segment_with_text_prompt(
    model: Sam3Model,
    processor: Sam3Processor,
    device: str,
    image_rgb: Image.Image,
    text: str,
    threshold: float,
    mask_threshold: float,
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    inputs = processor(images=image_rgb, text=text, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

    masks = results.get("masks")
    if masks is None or len(masks) == 0:
        return None, []

    all_masks = [mask.detach().cpu().numpy().astype(bool) for mask in masks]
    mask_areas = [int(mask.sum()) for mask in all_masks]
    best_mask = all_masks[int(np.argmax(mask_areas))]
    return best_mask, [best_mask]


def _process_camera_frame(
    model: Sam3Model,
    processor: Sam3Processor,
    device: str,
    input_dir: Path,
    output_dir: Path,
    camera_name: str,
    frame_idx: int,
    text: str,
    threshold: float,
    mask_threshold: float,
) -> None:
    image_path = input_dir / camera_name / f"{frame_idx:05d}.jpg"
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Warning: image not found, skipping {camera_name} frame {frame_idx:05d}: {image_path}")
        return

    image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    mask, all_masks = segment_with_text_prompt(
        model=model,
        processor=processor,
        device=device,
        image_rgb=image_rgb,
        text=text,
        threshold=threshold,
        mask_threshold=mask_threshold,
    )
    if mask is None:
        print(f"Warning: no SAM3 mask found, skipping {camera_name} frame {frame_idx:05d}")
        return

    mask_dir = output_dir / camera_name / "mask"
    rgba_dir = output_dir / camera_name / "rgba"
    vis_dir = output_dir / camera_name / "vis"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rgba_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    mask_uint8 = mask.astype(np.uint8) * 255
    cv2.imwrite(str(mask_dir / f"{frame_idx:05d}.png"), mask_uint8)

    rgba_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba_image[~mask, 3] = 0
    cv2.imwrite(str(rgba_dir / f"{frame_idx:05d}.png"), rgba_image)

    vis_image = render_segmentation(image_bgr, mask, all_masks=all_masks)
    cv2.imwrite(
        str(vis_dir / f"{frame_idx:05d}.jpg"),
        vis_image,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    reviewed_path = Path(args.reviewed)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    for camera_name in ("cam2", "cam4"):
        if not (input_dir / camera_name).exists():
            raise FileNotFoundError(f"Camera input directory not found: {input_dir / camera_name}")

    reviewed_pairs = load_reviewed_entries(reviewed_path)
    frame_indices = [frame_idx for _, frame_idx in reviewed_pairs]
    if args.limit > 0:
        frame_indices = frame_indices[: args.limit]

    if not frame_indices:
        print(f"No reviewed frames found in {reviewed_path}")
        return

    model, processor, device = load_sam3(args.model_dir)

    total_steps = len(frame_indices) * 2
    with tqdm(total=total_steps, desc="Segmenting SAM3", unit="frame") as progress:
        for frame_idx in frame_indices:
            _process_camera_frame(
                model=model,
                processor=processor,
                device=device,
                input_dir=input_dir,
                output_dir=output_dir,
                camera_name="cam2",
                frame_idx=frame_idx,
                text=args.text,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
            )
            progress.update(1)

            _process_camera_frame(
                model=model,
                processor=processor,
                device=device,
                input_dir=input_dir,
                output_dir=output_dir,
                camera_name="cam4",
                frame_idx=frame_idx,
                text=args.text,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
            )
            progress.update(1)


if __name__ == "__main__":
    main()

r'''
python src/script/test_function/debug038_seg_sam3.py \
  --model_dir tools/SAM3_and_SAM3_BODY/SAM3 \
  --input_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Output/20260311_riding/seg/equirect_crop_rotate_sam3 \
  --reviewed data_process/Input/20260311_riding/review.json \
  --text person \
  --threshold 0.5 \
  --mask_threshold 0.5 \
  --limit 0
'''
r'''
python src/script/test_function/debug038_seg_sam3.py \
  --model_dir tools/SAM3_and_SAM3_BODY/SAM3 \
  --input_dir data_process/Output/20260320_riding_dataset1/image/equirect_crop_rotate \
  --output_dir data_process/Output/20260320_riding_dataset1/seg/equirect_crop_rotate_sam3 \
  --reviewed data_process/Input/20260320_riding/dataset1/review.json \
  --text person \
  --threshold 0.5 \
  --mask_threshold 0.5 \
  --limit 0
'''
