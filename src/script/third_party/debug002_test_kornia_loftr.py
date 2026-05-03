"""
Quick local smoke test for Kornia LoFTR.

Example:
    python src/script/third_party/debug002_test_kornia_loftr.py --img0 data_process/Output/20260311_riding/seg/equirect_crop_rotate_sam3/cam2/rgba/00513.png --img1 data_process/Output/20260311_riding/seg/equirect_crop_rotate_sam3/cam4/rgba/00513.png
"""

import argparse
from pathlib import Path

import cv2
import torch
from kornia.feature import LoFTR


MAX_WIDTH = 640
MAX_HEIGHT = 480


def parse_args():
    parser = argparse.ArgumentParser(description="Test whether Kornia LoFTR works on this machine.")
    parser.add_argument("--img0", required=True, help="Path to the first image.")
    parser.add_argument("--img1", required=True, help="Path to the second image.")
    parser.add_argument(
        "--output",
        default="data_process/Vis/loftr_matches/loftr_matches.jpg",
        help="Path to save match visualization. Use an empty string to skip saving.",
    )
    return parser.parse_args()


def read_image_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def resize_to_limit(image, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    height, width = image.shape[:2]
    new_width = min(width, max_width)
    new_height = min(height, max_height)

    if (new_width, new_height) == (width, height):
        return image

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def image_to_loftr_tensor(image_bgr, device):
    image_bgr = resize_to_limit(image_bgr)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(image_gray).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    return tensor, image_bgr


def build_matcher(device):
    last_error = None
    for pretrained in ("outdoor", "indoor_new"):
        try:
            matcher = LoFTR(pretrained=pretrained).eval().to(device)
            print(f"Loaded LoFTR(pretrained={pretrained!r}) on {device}.")
            return matcher
        except Exception as exc:
            last_error = exc
            print(f"Failed to load LoFTR(pretrained={pretrained!r}): {exc}")

    raise RuntimeError("Failed to load LoFTR with both 'outdoor' and 'indoor_new'.") from last_error


def draw_matches(image0_bgr, image1_bgr, keypoints0, keypoints1, confidence, output_path):
    if keypoints0.size == 0:
        print("No matches to visualize.")
        return

    image0_rgb = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2RGB)
    image1_rgb = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)

    keypoints0_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints0]
    keypoints1_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints1]
    matches = [
        cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=float(1.0 - confidence[i]))
        for i in range(len(keypoints0_cv))
    ]

    vis_rgb = cv2.drawMatches(
        image0_rgb,
        keypoints0_cv,
        image1_rgb,
        keypoints1_cv,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_bgr)
    print(f"Saved match visualization to: {output_path}")


def print_preview(name, values, max_items=5):
    print(f"{name} first {min(max_items, len(values))}:")
    for item in values[:max_items]:
        print(f"  {item}")


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image0_bgr = read_image_bgr(args.img0)
    image1_bgr = read_image_bgr(args.img1)

    tensor0, resized0_bgr = image_to_loftr_tensor(image0_bgr, device)
    tensor1, resized1_bgr = image_to_loftr_tensor(image1_bgr, device)
    print(f"Image0 tensor shape: {tuple(tensor0.shape)}")
    print(f"Image1 tensor shape: {tuple(tensor1.shape)}")

    matcher = build_matcher(device)

    with torch.inference_mode():
        correspondences = matcher({"image0": tensor0, "image1": tensor1})

    keypoints0 = correspondences["keypoints0"].detach().cpu().numpy()
    keypoints1 = correspondences["keypoints1"].detach().cpu().numpy()
    confidence = correspondences["confidence"].detach().cpu().numpy()

    print(f"Match count: {len(keypoints0)}")
    print_preview("keypoints0", keypoints0)
    print_preview("keypoints1", keypoints1)
    print_preview("confidence", confidence)

    if args.output:
        draw_matches(resized0_bgr, resized1_bgr, keypoints0, keypoints1, confidence, args.output)


if __name__ == "__main__":
    main()
