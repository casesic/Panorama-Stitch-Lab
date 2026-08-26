# DenseKP CUDA → C++ CUDA staged-LM

## Production chain

入口：

```text
src/script/human_fitting/run_densekp_cuda_staged_lm.py
```

运行链：

```text
startup
  -> one persistent DenseKPRuntime
  -> one ORT CUDA session + one detector

frames
  -> fisheye pair -> cached ERP remap
  -> DenseKP CUDA inference
  -> CPU decode/person match/138-point ERP_ROTATE transform
  -> inject observation into frame-specific offline C++ model bundle

batch
  -> start C++ application once
  -> C++ Stage1 + multi-frame CUDA Pose/PoseShape fitting
  -> optimized DenseKP observations
```

DenseKP 的 ONNX output 当前由 ORT 解码为 NumPy，后处理、person match 和坐标
旋转也在 CPU，因此这里没有伪装成跨进程 GPU-pointer zero-copy。传给 LM 的动态
数据只有 138 对二维点；相比 ERP remap、detector、DenseKP inference 和 LM
Jacobian，这个小型 CPU boundary 不是主要成本。

C++ model bundle 仍必须离线准备，因为它包含每帧 HMR 初始化和 SMPL-X 的 204
support model 数据。runtime 不运行 Python SMPL-X/reference LM。静态 bundle 文件
使用符号链接复用，每帧只生成 observation、metadata 和少量 DenseKP local rows，
不会复制整份 fixture。

## 示例

```bash
conda run -n camerahmr python \
  src/script/human_fitting/run_densekp_cuda_staged_lm.py \
  --frames 513-516 \
  --fisheye-root data_process/Output/20260311_riding/image/fisheye \
  --onnx-path Circular_Video/01_recv/20260708/CameraHMR/deploy/densekp_onnx/artifacts/densekp_b_256x256.onnx \
  --base-fixture-root data_process/Vis/cpp_reference_dump_densekp \
  --cpp-executable build/staged_lm_cuda/multiframe_staged_lm_reference_main \
  --calibration-cache-dir data_process/Cache/densekp_calibration \
  --output data_process/Vis/densekp_cuda_cpp_lm/segment_513_516 \
  --camera old
```

`--frames` 支持 `513-516` 或 `513,514,515,516`。程序会强制检查
`CUDAExecutionProvider` 已激活，并检查 C++ summary 的 evaluator backend。

输出包括：

```text
output/
  observations/frame_*.json
  cpp_lm/frame_*/optimized_densekp.json
  cpp_lm/multiframe_summary.txt
  densekp_cuda_staged_lm_report.json
```

## 513--516 实测

真实 fisheye、calibration cache hit：

```text
DenseKP runtime startup       2107.39 ms（一次）
DenseKP + observation export 3895.19 ms / 4 frames
C++ process wall              339.00 ms
C++ LM internal               258.23 ms / 4 frames = 15.49 FPS
steady chain excluding startup 4234.18 ms = 0.945 FPS
```

ORT session creation 和 detector initialization 均为 1 次，四帧只启动一次 C++
multi-frame fitting。端到端瓶颈仍是 fisheye→ERP、detector 和 DenseKP delivery，
不是 C++ LM。

```text
DENSEKP_CUDA_CPP_LM_CONNECTED=PASS
```
