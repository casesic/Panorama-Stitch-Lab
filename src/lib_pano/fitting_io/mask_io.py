"""SAM3 分割 mask 路径获取和有效性判断。"""

from pathlib import Path


def get_mask_paths(frame_idx: int, seg_dir: Path) -> tuple[Path, Path]:
    """返回 cam2 和 cam4 的 SAM3 mask PNG 路径。"""
    return (
        seg_dir / "cam2" / "mask" / f"{frame_idx:05d}.png",
        seg_dir / "cam4" / "mask" / f"{frame_idx:05d}.png",
    )


def is_seg_valid(frame_idx: int, seg_dir: Path, invalid_frames: set[int]) -> bool:
    """判断该帧是否有可用的 SAM3 分割 mask（任一相机存在即可）。"""
    if frame_idx in invalid_frames:
        return False
    mask_c2, mask_c4 = get_mask_paths(frame_idx, seg_dir)
    return mask_c2.exists() or mask_c4.exists()
