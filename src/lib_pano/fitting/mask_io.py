from pathlib import Path


def get_mask_paths(frame_idx: int, seg_dir: Path) -> tuple[Path, Path]:
    return (
        seg_dir / "cam2" / "mask" / f"{frame_idx:05d}.png",
        seg_dir / "cam4" / "mask" / f"{frame_idx:05d}.png",
    )


def is_seg_valid(frame_idx: int, seg_dir: Path, invalid_frames: set[int]) -> bool:
    if frame_idx in invalid_frames:
        return False
    mask_c2, mask_c4 = get_mask_paths(frame_idx, seg_dir)
    return mask_c2.exists() or mask_c4.exists()
