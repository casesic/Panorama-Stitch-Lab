# DenseKP CUDA → C++ CUDA staged-LM 阅读指南

## 完整数据流

```text
cam2/cam4 fisheye JPEG
  -> parallel image read/decode
  -> fisheye -> normal ERP (resident LUT, CUDA remap with CPU fallback)
  -> person detector
  -> DenseKP ORT inference: 138 semantic points per detected person
  -> cam2/cam4 person matching
  -> finite + geometry + valid-region filtering
  -> ERP -> ERP_ROTATE point transform
  -> DenseKP observation JSON
  -> inject observations into 204-support C++ model fixtures
  -> one multi-frame C++ process
       -> Stage1
       -> Pose CUDA
       -> shared-beta PoseShape CUDA
  -> optimized DenseKP: 138 ERP_ROTATE point pairs
```

`keypoint_value` 只为 legacy/debug NPZ 兼容而保存，不决定 validity，也不进入 confidence、
loss、IRLS 或任何 fitting weight。

本项目运行时临时目录统一放在 `data_process/tmp`，不使用系统共享 `/tmp`。

## 从哪里开始

先打开
[`run_densekp_cuda_staged_lm.py`](../../../src/script/human_fitting/run_densekp_cuda_staged_lm.py)。
它负责 CLI、高层 orchestration 和资源生命周期。production runner 直接构造公开
`DenseKPConfig`，不会拼接伪 CLI 参数。

Python 推荐阅读顺序：

1. `src/script/human_fitting/run_densekp_cuda_staged_lm.py`
2. `densekp_delivery/config.py`
3. `densekp_delivery/runtime.py`
4. `densekp_delivery/pipeline.py`
5. `densekp_delivery/image_io.py`
6. `densekp_delivery/camera/fisheye_equirect.py`
7. `densekp_delivery/runners/ort_runner.py`
8. `densekp_delivery/person_match.py`
9. `densekp_delivery/valid_filter.py`
10. `densekp_delivery/camera/erp_rotate.py`
11. `src/lib_pano/fitting_io/densekp_delivery_io.py`
12. `src/lib_pano/fitting_io/densekp_runtime_export.py`

C++ 推荐阅读顺序：

1. `cpp/staged_lm/apps/multiframe_staged_lm_reference_main.cpp`
2. `src/multiframe.cpp`
3. `src/pipeline.cpp`
4. `src/lm.cpp`
5. `src/stage1_lm.cpp` / `src/stage1.cpp`
6. `src/pose_lm.cpp`
7. `src/pose_forward.cpp`
8. `src/pose_jacobian.cpp`
9. `src/pose_normal_equations.cpp`
10. `src/pose_shape_lm.cpp`
11. `src/pose_shape_forward.cpp`
12. `src/pose_shape_jacobian.cpp`
13. `src/pose_shape_normal_equations.cpp`
14. `src/pose_shape_schur.cpp`
15. `src/evaluator.cpp`
16. `src/cuda_evaluators.cpp`
17. 最后才看 `src/cuda_forward.cu`

## 三个优化阶段

| 阶段 | 输入 | 优化变量 | 输出及用途 |
|---|---|---|---|
| Stage1 | 每帧双相机 DenseKP 目标和初始局部点 | 两个相机各 3D orientation + translation | 每帧 12 维相机初值，交给 Pose |
| Pose | 固定 beta 的模型、相机初值和 DenseKP 目标 | 每帧 75 维：双相机参数 + body pose | 更准确的姿态与相机状态，交给 PoseShape |
| PoseShape | 所有帧的 Pose 结果 | 每帧 75 维 + 全序列共享 beta10 | 最终状态，用于重新 forward、回归并投影 DenseKP |

shared beta 体现在 multi-frame 参数布局尾部只有一份 beta10；每帧有独立 pose block，
PoseShape normal equations 对共享 beta 累积并由 Schur solver 求解。

## 204 support 与 138 observation

C++ SMPL-X forward 始终计算 204 个 support vertices。DenseKP sparse regressor 在最终
support vertices 及其 Jacobian之后产生 138 个语义点。valid filter 可能使实际 observation
row 少于 138。因此：

```text
support_count = 204
densekp_row_count = 138
observation_count <= 138
```

## 代码类别

- PRODUCTION：入口、delivery runtime/pipeline、runtime exporter、multi-frame staged-LM、
  evaluator 和 CUDA 实现。
- REFERENCE / PARITY：`apps/*_reference_main.cpp` 和固定 fixture；parity 不在 production runner 中执行。
- TEST / BENCHMARK / PROFILING：self-tests、timing JSON、汇总报告和专项性能文档。

第一遍可以完全跳过：self-tests、单阶段 reference apps、profiling/report 模块、legacy
NPZ/可视化、历史 bake-off 文档，以及 `cuda_forward.cu` 的 kernel 细节。

公式不在本文重复，见
[`../formulate/staged_lm_stage_formulas_and_jacobians_typora.md`](../formulate/staged_lm_stage_formulas_and_jacobians_typora.md)。
