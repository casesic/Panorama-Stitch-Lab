"""双视角拟合结果保存：将 cam2/cam4 的 SMPL-X 参数字典写入 pkl 和 meta JSON。"""

import json
import pickle
from pathlib import Path

import torch


def save_segment_result(
    result_c2,
    result_c4,
    shared_beta: torch.Tensor,
    output_dir: Path,
    frame_idxs: list[int],
    valid_frame_count_c2: int,
    valid_frame_count_c4: int,
) -> None:
    start = frame_idxs[0]
    end = frame_idxs[-1]
    shared_beta_list = [float(x) for x in shared_beta.detach().cpu().tolist()]

    for cam_id, result, valid_frame_count in (
        ("cam2", result_c2, valid_frame_count_c2),
        ("cam4", result_c4, valid_frame_count_c4),
    ):
        cam_dir = output_dir / cam_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        seq_path = cam_dir / f"seg_{start}_{end}_seq.pkl"
        meta_path = cam_dir / f"seg_{start}_{end}_meta.json"

        with seq_path.open("wb") as file:
            pickle.dump(result, file)

        meta = {
            "camera": cam_id,
            "start": start,
            "end": end,
            "frame_idxs": frame_idxs,
            "valid_frame_count": valid_frame_count,
            "total_frame_count": len(frame_idxs),
            "shared_beta": shared_beta_list,
        }
        with meta_path.open("w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False, indent=2)
