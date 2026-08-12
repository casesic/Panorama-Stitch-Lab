# Pose normal equations C++ CPU reference

## Actual production chain

当前 production 对每个线性化点先用串行 `fast_raw_groups` 冻结 Charbonnier IRLS 权重，再把每组 row scale padding 到 batched layout。实际链为：

```text
raw residual [N,2]
→ w_i = alpha / sqrt(||r_i||² + eps²)
→ row scale = sqrt(w_i)，按 point-major [u,v] 重复
→ FP32 weighted residual / J_active
→ cast FP64
→ 每组 H69=JᵀJ、g69=Jᵀr
→ 按 cam2、cam4 顺序 scatter/累加到 frame 75D
→ pose anchor 加到 [12:75]
→ final H75/g75
```

单帧时每个相机的 `alpha=1/N_camera`，`camera_weight=1`，`eps` 来自 production `ObjectiveConfig.charbonnier_eps`。cam2 的 active rigid block映射到 frame `[0:6]`，cam4 映射到 `[6:12]`；两者的 active body `[6:69]` 都累加到共享 frame `[12:75]`。production 会 materialize batched `J_active` 和每组 69 维 normal equations，不构造全局多帧 dense Jacobian。

## FP32/FP64 与 pose anchor

row scaling 前后 residual/J 都保持 FP32；只有加权完成后才转 FP64 做 `JᵀJ/Jᵀr`。Pose anchor 为

```text
delta = body_pose - body_pose_anchor
r_anchor = pose_anchor_scale * delta
H_anchor[12:75,12:75] = pose_anchor_scale² I
g_anchor[12:75] = pose_anchor_scale² delta
```

其余 rigid 行列严格为零。Reference 重建的 final H/g 与 canonical batched production 均 bitwise equal。

## Cross-backend parity policy

ERP residual 已采用 pixel-space absolute gate；IRLS 的倒数和平方根会进一步放大小 residual 的 relative error，因此下游采用显式 mixed policy：

- point weights：upstream residual PASS 且 `max_abs <= 1e-7`，relative 仅作诊断；
- row scales：upstream weights PASS 且 `max_abs <= 1e-6`；
- weighted residual：upstream residual/scale PASS 且 `max_abs <= 1e-5`；
- weighted J：upstream J/scale PASS、`rel <= 1e-5` 且 `max_abs <= 1e-3`；
- per-camera `H_active`（中间诊断）：`INTERMEDIATE_H_ACTIVE_REL_TOL=1.1e-6`；
- `H_data_frame/H_frame`（最终系统门禁）：`FINAL_H_REL_TOL=1e-6`；
- g：upstream weighted J/r PASS、`rel <= 5e-6` 且 `max_abs <= 1e-4`。

Relative 指标仍全部记录，不用 Python weights 绕过 C++ 计算。最终状态以 `cpp_output/pose_normal_equations_cpp_parity.json` 为准。

这里仅给分组中间量 `H_active` 留出跨 backend FP32 输入误差经 FP64 GEMM 传播的极小余量；最终实际交给 LM 的 `H_frame` 没有放宽，继续执行更严格的 `1e-6` 门禁。

本阶段只实现到 H/g，不包含 damping、solve、candidate objective、trial acceptance、Pose/Shape 或 CUDA。Normal-equation parity 全部通过后，下一步进入 Pose LM C++，先闭环 Pose 整阶段，再开始 Pose/Shape。
