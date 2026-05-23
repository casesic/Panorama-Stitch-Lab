# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

这是一个全景图像/视频拼接管线，将多路鱼眼相机画面（2-3路，通常是 cam2/cam4/cam6）拼接为 360° 等距柱状全景图。管线同时处理背景对齐（通过光流）和人体合成（通过基于 SMPL 的三维重建与语义光流）。

## 架构

```
src/lib_pano/           # 核心库 (可通过 `lib_pano` 导入)
  camera/               # 鱼眼相机模型 (Double Sphere / Omni)，鱼眼↔等距柱状投影
  calibration/          # 等距柱状图的投影与旋转工具
  io/                   # 标定文件读取 (JSON/YAML 内参、外参)
  pipeline/             # 批处理管线框架：配置校验、路径管理、帧索引
  flow/                 # 光流 (RAFT)、语义光流 (SMPL顶点→光流)、变形场
  human3d/              # SMPL-X 序列加载、关键点检测、骨骼常量
  correspondence/       # SMPL 顶点稀疏对应 (IO、过滤、可视化)
  composite/            # 人体混合 (RGBA放置、语义光流变形、接缝混合)
  segmentation/         # 跨帧Mask融合 (基于SAM的mask、IoU匹配)
  stitch/               # 顶层基于光流的拼接编排器
  local_setting.py      # 项目全局路径 (数据目录、标定文件、视频目录)
  logging_config.py     # 日志配置

src/script/test_function/  # ~40 个编号的调试/实验脚本 (debug001–debug053)
  run_pipeline.py          # 主流程运行器
  config_script.py         # PROJECT_ROOT 引导
  batch_utils.py           # 批处理工具

tools/                  # 第三方模型 (不由核心库管理)
  4D-Humans/, CameraHMR/, MeshPose/, RAFT/, GroundingDINO/
  SAM3_and_SAM3_BODY/, detectron2/, detectron2_camerahmr/, egowholemocap/
  PixelStitch/, WeaklySupervisedStitchingNetwork/, SingleFitting/, smpl_models/

data_process/           # 输入输出数据目录 (大文件已在gitignore中)
  Input/                # 标定参数、图像、特征匹配、人体物体
  Output/               # 管线输出
  Vis/                  # 可视化输出
```

## 核心概念

### 相机模型: Double Sphere (DS)
鱼眼相机使用 Kannala-Brandt 通用模型。标定文件为 OpenCV YAML 格式，包含字段: `camera_model: ds`, `intrinsics: [fx, fy, cx, cy]`, `distortion_coeffs: [xi, alpha, ...]`。代码同时支持 DS 和旧版 Omni-RadTan 模型。

### 坐标系约定
- 世界坐标系: Z=上, X=前, Y=左
- lon: 绕Z轴旋转 (向左为正), lat: 绕Y轴旋转 (向上为正)
- `R_wor_to_cam` / `T_wor_to_cam`: 世界坐标系到相机坐标系的变换
- 等距柱状投影: 水平 = 经度 [0, 2π], 垂直 = 纬度 [-π/2, π/2]
- 相机坐标系: X=右, Y=下, Z=前

### 管线步骤
管线由编号步骤组成，每个步骤对应一个 `debugXXX_*.py` 脚本。步骤通过 `run_pipeline.py --steps 020,021,... --config pipeline_config.json` 运行。每个步骤的必填/可选字段定义在 `pipeline/config_validation.py`。

关键步骤编号（大致）:
- 020: 导出/审查源图像
- 021: 鱼眼 → 等距柱状投影
- 024: 4D-Humans HMR 转换
- 025: 根据检测框裁剪等距柱状图
- 026: 在 ERP 上运行 RTMLib 关键点检测
- 029: 双视角 ERP SMPL 拟合
- 031: 从拟合的 SMPL 计算语义光流
- 037: 人体合成 (语义光流变形 + 混合)
- 038: SAM3 分割
- 040: GroundingDINO 检测框检测
- 043: Mask 融合

### 光流拼接
`stitch/flow_stitcher.py::stitch_with_flow()` 编排完整的两相机拼接:
1. 根据源/目标有效mask计算 ROI
2. 在 ROI 上运行 RAFT 光流
3. 可选地与 SMPL 导出的语义光流先验结合
4. 构建变形场 (基于网格或基于距离，双向)
5. 应用变形、混合和接缝修复

### 人体合成
人体与背景分开处理:
1. SMPL/SMPL-X 模型对各相机检测到的关键点进行拟合
2. 三维顶点投影到等距柱状图，生成稠密"语义光流"
3. 语义光流指导人体区域的逐像素变形
4. 使用 `composite/human_blend.py` 混合 (边界感知混合)

## 常用命令

```bash
# 在所有数据集上运行单个管线步骤
python src/script/test_function/run_pipeline.py --config path/to/config.json --steps 021

# 运行多个步骤
python src/script/test_function/run_pipeline.py --config path/to/config.json --steps 020,021,025

# 仅在指定数据集上运行
python src/script/test_function/run_pipeline.py --config path/to/config.json --steps 021 --datasets dataset1,dataset2

# 空运行 (仅打印命令不执行)
python src/script/test_function/run_pipeline.py --config path/to/config.json --steps 021 --dry_run

# 列出所有可用的步骤ID及对应脚本
python src/script/test_function/run_pipeline.py --list_steps

# 仅验证配置不运行
python src/script/test_function/run_pipeline.py --config path/to/config.json --steps 021 --validate_only
```

## 开发注意事项

- 从 `lib_pano` 导入前，必须将 `src/` 加入 `sys.path`。使用 `config_script.py` 或 `pipeline/paths.py::ensure_src_path()` 作为引导。
- 设置 `PYRIGHT_TMPDIR` 和 `TMPDIR` 环境变量 (见 `.env`) 以避免 tmpfs 空间不足问题。
- 建筑物/背景的 RAFT 光流使用 `raft-kitti.pth` 检查点，其他模型在此数据上效果不佳。
- `.claudeignore` 排除了 `tools/` 第三方模型和大型数据目录，避免占用上下文。
- 标定数据位于 `data_process/Input/calib_params/`，同时支持 JSON (旧版 Omni-RadTan) 和 YAML (Double Sphere)。
- 鱼眼原图朝下 (镜头向下安装)，图像高度大于宽度 (如 3840×2880)，标定时注意 h/w 对应关系。
- SMPL 拟合中，关节点检测置信度低于 0.5 的关键点不可靠，应考虑排除或降权。
