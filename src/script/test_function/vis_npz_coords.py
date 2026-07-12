import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import cv2
import numpy as np


def draw_points(img, points, color, radius=3):
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), radius, color, -1)


def main():
    parser = argparse.ArgumentParser(description="Visualize keypoints_orig_xy and keypoints_global_orig_xy from NPZ")
    parser.add_argument("--npz", required=True, help="Path to .npz file")
    parser.add_argument("--s2_img", default=None, help="S2 rotated-crop image for keypoints_orig_xy (optional)")
    parser.add_argument("--orig_img", default=None, help="Original full image for keypoints_global_orig_xy (optional)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz)
    stem = Path(args.npz).stem

    if "keypoints_orig_xy" in data and args.s2_img:
        img = cv2.imread(args.s2_img)
        if img is not None:
            draw_points(img, data["keypoints_orig_xy"], (0, 255, 0))
            out_path = out_dir / f"{stem}_kp_orig_on_s2.jpg"
            cv2.imwrite(str(out_path), img)
            print(f"keypoints_orig_xy ({len(data['keypoints_orig_xy'])} pts) on S2 → {out_path}")

    if "keypoints_global_orig_xy" in data and args.orig_img:
        img = cv2.imread(args.orig_img)
        if img is not None:
            draw_points(img, data["keypoints_global_orig_xy"], (0, 0, 255))
            out_path = out_dir / f"{stem}_kp_global_on_orig.jpg"
            cv2.imwrite(str(out_path), img)
            print(f"keypoints_global_orig_xy ({len(data['keypoints_global_orig_xy'])} pts) on orig → {out_path}")


if __name__ == "__main__":
    main()


r'''
# ---- Usage Example (run from repo root) ----

python src/script/test_function/vis_npz_coords.py \
  --npz /home/yuyue/Projects/pano-stitch/data_process/Vis/crop_v1/513_full_cam2_p0_0.npz \
  --s2_img /home/yuyue/Projects/pano-stitch/data_process/Vis/crop_v1/513_full_cam2_s2_p0.jpg \
  --orig_img /home/yuyue/Projects/pano-stitch/data_process/Vis/crop_v1/513_full_cam2_s1_detect.jpg \
  --out_dir /home/yuyue/Projects/pano-stitch/data_process/Vis/crop_v1
'''
