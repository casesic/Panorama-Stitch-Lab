# Multi-frame CUDA Pose parity report

## 验证配置

- T=1：frame 513
- T=4：frames 513--516
- T=8：frames 704--711
- observation：DenseKP Sparse-B
- policy：`optimized_native_stable_p`
- 对照：同一 CUDA binary 的 `--sequential-cuda-pose` 路径

## 结果

T=1 直接使用原单帧 CUDA Pose 实现，Pose final parameters 与既有结果
bitwise exact。

T=4 和 T=8 中，每帧 `pose_trajectory.csv` 与 sequential CUDA 路径逐字节
一致；每帧 Pose final parameters、后续 PoseShape trajectory 和最终参数也
逐字节一致。因此：

```text
CONTROL_DECISIONS_EXACT=YES
TRAJECTORY_EXACT=YES
FINAL_PARAMETERS_EXACT=YES
FIRST_DIVERGENCE=NONE
TOLERANCE_CHANGE_COUNT=0
```

CPU 和 CUDA build 的 11 项 CTest 回归均通过。

batched parameter upload 与 H/g reduction 的最新 transfer 和 wall-time 结果见
`staged_lm_cuda_multiframe_pose_memory_reduction.md`。轨迹门禁保持不变。
