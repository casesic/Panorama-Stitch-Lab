# C++ staged-LM production cleanup

## 最终 production 路径

- kernel policy 固定为 `optimized_native_stable_p`，不再提供运行时 kernel policy。
- Pose value pose-offset 使用 Eigen GEMV。
- Pose/PoseShape pose-offset derivative 使用 Eigen GEMM。
- selected-transform derivative 使用 Eigen GEMM。
- DenseKP 只保留 Sparse-B materialized：先把 204 个 support vertex/Jacobian 回归为 observation，再执行 ERP 链。
- multi-frame 仍为逐帧 Stage1、逐帧 Pose、联合 PoseShape，并且只有一个 shared-beta block。

## 删除和收敛

- 完全删除 forward-reuse 的 cache、mode、shadow、attribution 和 LM 分支。
- 完全删除 performance attribution recorder、payload/hash tracing 和 dispatch counters；保留 `LMRunResult` 的轻量阶段计时。
- 删除 dense/sparse-fused DenseKP runtime dispatch，library 只暴露 `apply_densekp_operator()`。
- 删除 scalar、transpose、blocking、direct-LBS 等历史实验 kernel 与 benchmark/attribution tests。
- 通用 dense fallback、structured gradient、dense Hessian 和 quadratic form 已收敛到 `pose_shape_schur.*`。
- CMake project/library 统一命名为 `staged_lm`；可执行 target 从 45 个降为 21 个，CTest 从 32 项降为 11 项，保留阶段、pipeline、multi-frame、Schur 和 DenseKP correctness coverage。

## Correctness

- CTest：11/11 PASS。
- SAM3D canonical T=1：Stage1/Pose/PoseShape trajectory 与最终参数逐字节一致。
- DenseKP T=1 Sparse-B：trajectory、最终参数、physical beta 逐字节一致。
- DenseKP T=4：layout、finite、shared-beta 和 Schur self-test PASS。
- DenseKP T=8：所有 frame 参数和 shared physical beta 与封板 Sparse-B baseline 逐字节一致。
- Schur dense parity PASS；未修改 tolerance。

## Performance

在同一 Release/native build 下交替运行 cleanup 前后 executable，结果没有出现超过 3% 的回退。DenseKP T=8 的 `ms_per_frame` 从封板 baseline 约 41.84 ms 降至本轮中位数约 38.78 ms。单进程冷启动 T=1 波动较大，但配对样本的 DenseKP 和 SAM3D 中位数均未回退。

```text
STAGED_LM_CPP_PRODUCTION_CLEANUP_READY=PASS
```
