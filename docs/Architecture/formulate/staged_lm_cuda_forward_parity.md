# Staged-LM CUDA value-forward parity report

## 配置

- GPU：NVIDIA GeForce RTX 3090，compute capability 8.6；
- CUDA compiler：12.1.105；
- fixture：DenseKP frame 513，204 support / Sparse-B / 138 DenseKP rows；
- CPU production：`optimized_native_stable_p`；
- comparison states：CPU LM 最终 Pose 和 PoseShape parameters；
- 未修改任何 tolerance。

## CPU/CUDA forward 误差

Pose 与 PoseShape、cam2 与 cam4 合并取最大值：

| 输出 | max absolute error |
|---|---:|
| SMPL-X selected vertices | `2.384185791015625e-07` |
| DenseKP Sparse-B 3D points | `2.384185791015625e-07` |
| ERP raw residual | `4.882812500000000e-04 px` |

Residual 的误差来自 CPU libm 与 CUDA `atan2f/asinf` 的 FP32 transcendental 差异，
与此前跨 backend ERP policy 的误差量级一致。本轮只记录误差，没有改变或新增
production tolerance。

## CPU production regression

加入 CUDA 源码并链接 CUDA runtime 后，以下输出与 evaluator 抽象后的 CPU baseline
逐字节一致：

- Stage1/Pose/PoseShape trajectory；
- Stage1/Pose/PoseShape final parameters；
- final physical beta；
- optimized DenseKP JSON。

```text
TRAJECTORY_EXACT=YES
FINAL_PARAMETERS_EXACT=YES
FIRST_DIVERGENCE=NONE
TOLERANCE_CHANGE_COUNT=0
```

上述 exact 结论针对默认 CPU production trajectory。后续 Pose Jacobian CUDA
迁移的跨 backend 结果见 `staged_lm_cuda_pose_jacobian_parity.md`。

## 下一阶段：CUDA Jacobian migration

建议保持现有 CPU controller，不先迁移 LM 状态机：

1. 完成 PoseShape beta/dynamic-joint/dynamic-vertex derivative；
2. 在 CUDA 上直接做 IRLS 与 normal-equation reduction，避免完整 Jacobian 回传；
3. 最终只把小型 H/g 或 structured blocks 交给现有 CPU solve/controller。

每一步继续沿用 value -> Jacobian -> NE -> trajectory 的分层 parity，避免同时迁移
数学和控制状态机。
