from pathlib import Path
import argparse
import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_rotate_code(frame_id: int):
    """
    返回 OpenCV 旋转代码。
    当前规则：
    - 03495 之前：左转 90 度
    - 03495 到 05381：右转 90 度
    - 05951 到 05954：左转 90 度
    """
    if frame_id < 3495:
        return cv2.ROTATE_90_COUNTERCLOCKWISE

    if 3495 <= frame_id <= 5381:
        return cv2.ROTATE_90_CLOCKWISE

    if 5951 <= frame_id <= 5954:
        return cv2.ROTATE_90_COUNTERCLOCKWISE

    # 没有明确规则的帧，默认不旋转
    return None


def rotate_folder_by_frame_rule(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    print(f"[INFO] {input_dir}: {len(image_paths)} images")

    for img_path in image_paths:
        try:
            frame_id = int(img_path.stem)
        except ValueError:
            print(f"[WARN] skip non-numeric filename: {img_path.name}")
            continue

        rotate_code = get_rotate_code(frame_id)

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] failed to read: {img_path}")
            continue

        if rotate_code is None:
            print(f"[WARN] no rotate rule for {img_path.name}, copy without rotation")
            rotated = img
        else:
            rotated = cv2.rotate(img, rotate_code)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), rotated)

    print(f"[DONE] saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--cams", nargs="+", default=["cam2", "cam4"])
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    for cam in args.cams:
        in_dir = input_root / cam
        out_dir = output_root / cam

        if not in_dir.exists():
            print(f"[WARN] missing folder: {in_dir}")
            continue

        rotate_folder_by_frame_rule(in_dir, out_dir)


if __name__ == "__main__":
    main()
    
r'''
python src/script/test_function/debug001_rotate_cam_by_frame_rule.py \
  --input_root data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_root data_process/Output/20260311_riding/image/equirect_crop_rotate_upright \
  --cams cam2 cam4
'''
