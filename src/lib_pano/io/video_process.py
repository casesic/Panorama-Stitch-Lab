import cv2 as cv
import numpy as np

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
    output_path = f'{output_dir}/frame_{frame_index:05d}.png'
    cv.imwrite(output_path, frame)
    cap.release()
    print(f'已保存帧 {frame_index} 到 {output_path}')