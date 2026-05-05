from pathlib import Path

import cv2


MASK_PATH = Path(r"data_process/Output/20260311_riding/raft_stitch/unrotated_repair_mask.png")
FRAME_GLOB = Path(r"data_process/Output/20260311_riding/raft_stitch").glob("frame_*")
IMAGE_PATTERN = "raft_*_unrotated_frame_*.png"

INPAINT_RADIUS = 7
OVERWRITE = False


def output_path_for(image_path: Path) -> Path:
    if OVERWRITE:
        return image_path
    return image_path.with_name(f"{image_path.stem}_repaired{image_path.suffix}")


def main() -> None:
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {MASK_PATH}")

    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    processed = 0
    skipped = 0

    frame_dirs = sorted(path for path in FRAME_GLOB if path.is_dir())
    if not frame_dirs:
        print("Warning: no frame directories found.")

    for frame_dir in frame_dirs:
        image_paths = sorted(frame_dir.glob(IMAGE_PATTERN))
        if not image_paths:
            print(f"{frame_dir}: no matching unrotated images.")
            continue

        for image_path in image_paths:
            if image_path.stem.endswith("_repaired"):
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                skipped += 1
                print(f"Warning: failed to read image, skipped: {image_path}")
                continue

            if image.shape[:2] != mask.shape:
                raise ValueError(
                    f"Image size {image.shape[:2]} does not match mask size {mask.shape}: {image_path}"
                )

            repaired = cv2.inpaint(image, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
            out_path = output_path_for(image_path)
            if not cv2.imwrite(str(out_path), repaired):
                raise IOError(f"Failed to write repaired image: {out_path}")

            processed += 1
            print(f"{frame_dir.name}: repaired {image_path.name} -> {out_path.name}")

    print(f"Done. processed={processed}, skipped={skipped}, overwrite={OVERWRITE}")


if __name__ == "__main__":
    main()
