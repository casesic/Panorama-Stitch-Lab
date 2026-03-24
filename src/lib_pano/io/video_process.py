import cv2 as cv
import numpy as np
from typing import Optional, Tuple
import os

def extract_video_frames_range(
    path_video: str,
    output_dir: str,
    start_frame: int = 0,
    num_frames: int = 196,
    output_size: Optional[Tuple[int, int]] = None,
    rotate_left_90: bool = True,
) -> None:
    """
    从视频的 start_frame 帧开始，连续导出 num_frames 帧。

    Args:
        output_size: 指定输出分辨率 (width, height)，例如 (1280, 960)。
                     为 None 时保持旋转后的原始分辨率。
        rotate_left_90: 是否向左旋转 90 度，默认 True。
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv.VideoCapture(path_video)
    if not cap.isOpened():
        print(f'错误：无法打开视频文件 {path_video}')
        return
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    saved = 0
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f'警告：第 {start_frame + i} 帧读取失败，提前结束（共 {total} 帧）')
            break
        if rotate_left_90:
            rotated = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = frame
        if output_size is not None:
            out_w, out_h = output_size
            rotated = cv.resize(rotated, (int(out_w), int(out_h)), interpolation=cv.INTER_AREA)
        out_path = os.path.join(output_dir, f'frame_{start_frame + i:05d}.png')
        cv.imwrite(out_path, rotated)
        saved += 1
    cap.release()
    if output_size is None:
        print(f'已导出 {saved} 帧（起始帧={start_frame}，原分辨率）→ {output_dir}')
    else:
        print(f'已导出 {saved} 帧（起始帧={start_frame}，输出分辨率={output_size[0]}x{output_size[1]}）→ {output_dir}')


def extract_video_frames(path_video: str, output_dir: str, frame_index: int = 1) -> None:
    """
    从视频中提取帧并保存到指定目录。
    """
    cap = cv.VideoCapture(path_video)
    if not cap.isOpened():
        print(f'错误：无法打开视频文件 {path_video}')
        return
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    output_path = f'{output_dir}/frame_{frame_index:05d}_lrv.png'
    cv.imwrite(output_path, frame)
    cap.release()
    print(f'已保存帧 {frame_index} 到 {output_path}')