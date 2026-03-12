"""
SMPLFitterSample 使用示例 —— 支持图片 / 视频输入
=================================================

完整管线：
  图片/视频 → [RTMPose] 2D 关键点检测 → [SMPLFitterSample] 3D SMPL 拟合

前置条件：
  pip install torch smplx numpy opencv-python rtmlib

  SMPL 模型目录（smpl_full/）需包含：
    SMPL_NEUTRAL.pkl / SMPL_MALE.pkl / SMPL_FEMALE.pkl
  可选：J_regressor_extra.npy（9×6890，从 SPIN/HMR 仓库下载）

  设置环境变量：
    $env:LIB_SMPL_MODEL_DIR = "D:\\path\\to\\smpl_full"
  或直接修改 smpl_runner.py 顶部的 _DEFAULT_SMPL_DIR

用法：
  # 单张图片
  python example_smpl_fitter.py --input photo.jpg --fx 800 --fy 800

  # 视频（每隔 5 帧采样一次）
  python example_smpl_fitter.py --input video.mp4 --fx 800 --fy 800 --stride 5

  # 指定输出目录
  python example_smpl_fitter.py --input video.mp4 --fx 800 --fy 800 --out_dir ./output
"""

import argparse
import os
import sys
import os.path as osp

import cv2
import numpy as np
import torch

# 将 human3d 目录加入路径
_HERE = osp.dirname(osp.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# 将 src 目录加入路径（用于 lib_pano）
_SRC = osp.abspath(osp.join(_HERE, '..', '..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import json
from typing import Dict, Optional
from constants_skeleton import COCO_17
from smpl_fitter_sample import SMPLFitterSample


# =============================================================================
# 读取 camera_calib.json 中的 omni-radtan 参数
# =============================================================================

def load_omni_params_from_calib(calib_path: str, cam_id: str) -> Dict:
    """
    从 camera_calib.json 中读取指定相机的 omni-radtan 参数，
    返回可直接传入 fit_from_multi_frame_seq(omni_params=...) 的字典。

    camera_calib.json 格式（K = [xi, fx, fy, cx, cy]）：
      distortion: [k1, k2, p1, p2]
      K         : [xi, fx, fy, cx, cy]

    Args:
        calib_path: camera_calib.json 文件路径
        cam_id    : 相机名称，如 'camera2' / 'camera4' / 'camera6'
    Returns:
        dict with keys: xi, fx, fy, cx, cy, k1, k2, p1, p2
    """
    with open(calib_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data['camera_configs']['results']
    if cam_id not in results:
        available = [k for k in results if isinstance(results[k], dict) and 'K' in results[k]]
        raise KeyError(f"相机 '{cam_id}' 不在标定文件中。可用: {available}")
    cam = results[cam_id]
    xi, fx, fy, cx, cy = cam['K']           # 注意：第一个元素是 xi
    k1, k2, p1, p2    = cam['distortion']
    params = dict(xi=xi, fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, p1=p1, p2=p2)
    print(f"[标定] {cam_id}: xi={xi:.4f} fx={fx:.1f} fy={fy:.1f} "
          f"cx={cx:.1f} cy={cy:.1f} k1={k1:.4f} k2={k2:.4f}")
    return params


# =============================================================================
# DetectionResult：单帧关键点检测结果容器
# =============================================================================

class DetectionResult:
    """
    单帧关键点检测结果。
    keypoints      : (J, 3)  float32 —— (x_pixel, y_pixel, confidence)
    keypoints_format: dict   —— 关节名称 -> 关节索引，如 COCO_17
    """
    def __init__(self, keypoints: np.ndarray, keypoints_format: dict):
        assert keypoints.ndim == 2 and keypoints.shape[1] == 3
        self.keypoints = keypoints.astype(np.float32)
        self.keypoints_format = keypoints_format


# =============================================================================
# 关键点检测：RTMPose（rtmlib）
# =============================================================================

def build_rtmpose_detector(device: str = 'cpu'):
    """
    初始化 RTMPose 检测器。
    输出格式为 COCO-17（与 COCO_17 字典对齐）。
    依赖：pip install rtmlib onnxruntime(-gpu)
    """
    from rtmlib import Body
    body = Body(mode='balanced', to_openpose=False,
                backend='onnxruntime', device=device)
    return body


def detect_keypoints_single(detector, frame_bgr: np.ndarray,
                             conf_thresh: float = 0.3) -> DetectionResult | None:
    """
    对单帧 BGR 图像运行 RTMPose，返回置信度最高的人的 COCO-17 关键点。

    Returns:
        DetectionResult 或 None（未检测到人）
    """
    all_kps, all_sc = detector(frame_bgr)   # (N,17,2), (N,17)
    if len(all_kps) == 0:
        return None
    # 选置信度最高的人（用上半身 11 个关节均分）
    best_i = int(np.argmax([sc[:11].mean() for sc in all_sc]))
    kps = all_kps[best_i].astype(np.float32)   # (17, 2) —— (x, y)
    sc  = all_sc[best_i].astype(np.float32)    # (17,)
    # 合并为 (17, 3): (x, y, confidence)
    kps_with_conf = np.concatenate([kps, sc[:, None]], axis=1)  # (17, 3)
    # 低置信度置零（不参与优化）
    kps_with_conf[sc < conf_thresh, 2] = 0.0
    return DetectionResult(kps_with_conf, COCO_17)


# =============================================================================
# 输入读取：图片 / 视频
# =============================================================================

def load_frames_from_image(img_path: str):
    """从单张图片读取一帧，返回 [(frame_bgr, (H, W))]"""
    frame = cv2.imread(img_path)
    if frame is None:
        raise FileNotFoundError(f"无法读取图片：{img_path}")
    H, W = frame.shape[:2]
    print(f"[输入] 图片 {osp.basename(img_path)}  {W}×{H}")
    return [frame], H, W


def load_frames_from_video(video_path: str, stride: int = 1, max_frames: int = 120):
    """
    从视频中按步长采样帧。
    stride    : 每隔 stride 帧取一帧（视频较长时加大以提速）
    max_frames: 最多取多少帧
    Returns: frames_list, H, W
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[输入] 视频 {osp.basename(video_path)}  {W}×{H}  {fps:.1f}fps  共 {total} 帧  stride={stride}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
        if len(frames) >= max_frames:
            break
        idx += 1
    cap.release()
    print(f"[输入] 实际使用 {len(frames)} 帧")
    return frames, H, W


# =============================================================================
# 主函数
# =============================================================================

def run_pipeline(
    input_path: str,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    omni_params: Optional[Dict] = None,
    stride: int = 1,
    max_frames: int = 120,
    conf_thresh: float = 0.3,
    device: str = 'cpu',
    out_dir: str = './smpl_output',
):
    """
    完整管线：图片/视频 → 2D检测 → SMPL拟合 → 保存结果。

    相机参数二选一：
      针孔相机：提供 fx, fy（cy, cy 默认图像中心）
      鱼眼相机：提供 omni_params（来自 load_omni_params_from_calib）

    Args:
        input_path : 输入图片（.jpg/.png）或视频（.mp4/.avi）路径
        fx, fy     : 针孔相机焦距（像素），使用针孔模式时填写
        cx, cy     : 针孔主点（默认图像中心）
        omni_params: omni-radtan 鱼眼参数字典，使用鱼眼模式时填写
        stride     : 视频采样步长（图片忽略此参数）
        max_frames : 视频最大采样帧数
        conf_thresh: 关键点置信度阈值（低于此值不参与优化）
        device     : 'cpu' 或 'cuda'
        out_dir    : 结果保存目录
    """
    if omni_params is None and fx is None:
        raise ValueError("请提供 fx/fy（针孔模式）或 omni_params（鱼眼模式）之一")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. 读取帧 ───────────────────────────────────────────────
    ext = osp.splitext(input_path)[1].lower()
    if ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
        frames, H, W = load_frames_from_image(input_path)
    elif ext in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
        frames, H, W = load_frames_from_video(input_path, stride=stride, max_frames=max_frames)
    else:
        raise ValueError(f"不支持的文件格式：{ext}  请传入图片或视频文件。")

    # 构建相机参数
    if omni_params is not None:
        # 鱼眼模式：omni_params 已包含所有参数
        camera_K = None
        print(f"[相机] 模式=omni-radtan  xi={omni_params['xi']:.4f} "
              f"fx={omni_params['fx']:.1f} fy={omni_params['fy']:.1f} "
              f"cx={omni_params['cx']:.1f} cy={omni_params['cy']:.1f}")
    else:
        # 针孔模式
        if fy is None:
            fy = fx
        if cx is None:
            cx = W / 2.0
        if cy is None:
            cy = H / 2.0
        camera_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        print(f"[相机] 模式=针孔  fx:{fx} fy:{fy} cx:{cx:.1f} cy:{cy:.1f}")

    # ── 2. 关键点检测 ────────────────────────────────────────────
    print("[检测] 初始化 RTMPose...")
    detector = build_rtmpose_detector(device=device)

    detection_results = []
    for i, frame in enumerate(frames):
        dr = detect_keypoints_single(detector, frame, conf_thresh=conf_thresh)
        if dr is None:
            print(f"  ⚠ 第 {i} 帧未检测到人，跳过")
            continue
        valid = int((dr.keypoints[:, 2] > conf_thresh).sum())
        print(f"  帧 {i:03d}  有效关节 {valid}/17")
        detection_results.append(dr)

    if len(detection_results) == 0:
        raise RuntimeError("所有帧均未检测到人体关键点，请检查输入文件。")
    print(f"[检测] 共得到 {len(detection_results)} 帧有效检测结果")

    # ── 3. SMPL 拟合 ─────────────────────────────────────────────
    print("[SMPL] 初始化 SMPLFitterSample...")
    fitter = SMPLFitterSample(dim_shape=10)

    # 热启动：假设人在相机前方约 3m
    smpl_init = SMPLFitterSample.SMPLParamSeq(
        num_batch=len(detection_results), dim_shape=10, gender="neutral"
    )
    smpl_init.transl[:, 2] = 3.0   # Z 轴初始化为 3m

    print(f"[SMPL] 开始优化（{len(detection_results)} 帧）...")
    smpl_result = fitter.fit_from_multi_frame_seq(
        detection_results_list=detection_results,
        camera=camera_K,
        smpl_param_init=smpl_init,
        omni_params=omni_params,
    )
    print("[SMPL] 优化完成！")

    # ── 4. 前向推理，取最终顶点和关节点 ─────────────────────────
    verts, joints_54 = fitter.forward_smpl_param_seq(smpl_result)
    # verts    : (N, 6890, 3)  SMPL mesh 顶点（相机坐标系）
    # joints_54: (N, 54,  3)  54 个关节点（相机坐标系）

    N = len(detection_results)
    orient_np  = smpl_result.orient.detach().cpu().numpy()   # (N, 3)
    transl_np  = smpl_result.transl.detach().cpu().numpy()   # (N, 3)
    pose_np    = smpl_result.pose.detach().cpu().numpy()     # (N, 69)
    beta_np    = smpl_result.beta.detach().cpu().numpy()     # (10,)
    verts_np   = verts.detach().cpu().numpy()                # (N, 6890, 3)
    joints_np  = joints_54.detach().cpu().numpy()            # (N, 54,  3)

    print(f"\n[结果]")
    print(f"  帧数       : {N}")
    print(f"  orient     : {orient_np.shape}  (轴角，弧度)")
    print(f"  transl     : {transl_np.shape}  (相机坐标系，米)")
    print(f"  pose (body): {pose_np.shape}    (69维，23个关节轴角)")
    print(f"  beta       : {beta_np.shape}    (形状参数)")
    print(f"  verts      : {verts_np.shape}   (SMPL mesh 顶点)")
    print(f"  joints_54  : {joints_np.shape}  (54 关节)")
    print(f"  transl 第0帧: {transl_np[0]}")

    # ── 5. 保存结果 ──────────────────────────────────────────────
    np.save(osp.join(out_dir, 'smpl_orient.npy'),  orient_np)
    np.save(osp.join(out_dir, 'smpl_transl.npy'),  transl_np)
    np.save(osp.join(out_dir, 'smpl_pose.npy'),    pose_np)
    np.save(osp.join(out_dir, 'smpl_beta.npy'),    beta_np)
    np.save(osp.join(out_dir, 'smpl_verts.npy'),   verts_np)
    np.save(osp.join(out_dir, 'smpl_joints54.npy'), joints_np)
    print(f"\n[保存] 结果已写入 {osp.abspath(out_dir)}/")

    # 可选：将关键点投影回图像，做可视化验证
    _save_projection_vis(frames, detection_results, joints_np, camera_K, out_dir,
                         omni_params=omni_params)

    return smpl_result, verts, joints_54


def _save_projection_vis(
    frames: list,
    detection_results: list,
    joints_np: np.ndarray,
    K: Optional[np.ndarray],
    out_dir: str,
    omni_params: Optional[Dict] = None,
):
    """把拟合后的 SMPL 关节点投影回图像并叠加显示（可选，仅用于验证）。"""
    from constants_skeleton import COCO_17
    # 只可视化前 5 帧，避免太慢
    n_vis = min(5, len(detection_results))
    for i in range(n_vis):
        frame = frames[i].copy()
        det_kps = detection_results[i].keypoints     # (17, 3) detected

        # 投影 SMPL 54 关节到图像
        j3d = joints_np[i]                           # (54, 3)
        if omni_params is not None:
            from lib_pano.camera.omni_camera import omni_project_np
            proj_2d = omni_project_np(
                j3d,
                fx=omni_params['fx'], fy=omni_params['fy'],
                cx=omni_params['cx'], cy=omni_params['cy'],
                k1=omni_params['k1'], k2=omni_params['k2'],
                p1=omni_params['p1'], p2=omni_params['p2'],
                xi=omni_params['xi'],
            )  # (54, 2)
        else:
            proj = (j3d @ K.T)                           # (54, 3)
            proj_2d = proj[:, :2] / (proj[:, 2:3] + 1e-9)  # (54, 2)

        # 画检测到的关键点（绿色）
        for _, idx in COCO_17.items():
            x, y, c = det_kps[idx]
            if c > 0.3:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # 画 SMPL 投影点（红色）—— 取前 24 个标准关节
        for j in range(24):
            x, y = proj_2d[j]
            cv2.circle(frame, (int(x), int(y)), 4, (0, 60, 255), -1)

        out_path = osp.join(out_dir, f'vis_frame_{i:03d}.jpg')
        cv2.imwrite(out_path, frame)
    if n_vis > 0:
        print(f"[可视化] 已保存前 {n_vis} 帧投影可视化到 {out_dir}/vis_frame_*.jpg")


# =============================================================================
# 命令行入口
# =============================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description='图片/视频 → RTMPose 关键点检测 → SMPL 3D 拟合'
    )
    parser.add_argument('--input',  required=True,
                        help='输入图片（.jpg/.png）或视频（.mp4/.avi）路径')
    parser.add_argument('--fx',     type=float, default=None, help='针孔相机焦距 fx（像素），针孔模式时必填')
    parser.add_argument('--fy',     type=float, default=None, help='针孔相机焦距 fy（像素），默认同 fx')
    parser.add_argument('--cx',     type=float, default=None, help='主点 cx（默认图像宽/2）')
    parser.add_argument('--cy',     type=float, default=None, help='主点 cy（默认图像高/2）')
    parser.add_argument('--calib',  type=str,   default=None,
                        help='camera_calib.json 路径（鱼眼模型时使用）')
    parser.add_argument('--cam_id', type=str,   default='camera2',
                        help='标定文件中的相机名称，如 camera2 / camera4 / camera6（默认 camera2）')
    parser.add_argument('--stride', type=int,   default=5,
                        help='视频帧采样步长（每隔 N 帧取一帧），默认 5')
    parser.add_argument('--max_frames', type=int, default=120,
                        help='视频最多采样帧数，默认 120')
    parser.add_argument('--conf',   type=float, default=0.3,
                        help='关键点置信度阈值，默认 0.3')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='推理设备，默认 cpu')
    parser.add_argument('--out_dir', default='./smpl_output',
                        help='结果保存目录，默认 ./smpl_output')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()

    # 确定相机模型
    omni_params = None
    if args.calib is not None:
        omni_params = load_omni_params_from_calib(args.calib, args.cam_id)
    elif args.fx is None:
        raise SystemExit(
            "错误：请提供 --calib（鱼眼模式）或 --fx（针孔模式）之一。\n"
            "示例（鱼眼）: python example_smpl_fitter.py --input video.mp4 "
            "--calib data_process/Input/calib_params/camera_calib.json --cam_id camera2\n"
            "示例（针孔）: python example_smpl_fitter.py --input photo.jpg --fx 800"
        )

    run_pipeline(
        input_path=args.input,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        omni_params=omni_params,
        stride=args.stride,
        max_frames=args.max_frames,
        conf_thresh=args.conf,
        device=args.device,
        out_dir=args.out_dir,
    )
