# DenseKP 与 SAM3D C++ staged-LM 性能归因

## 结论

```text
DENSEKP_VS_SAM3D_CPP_PERFORMANCE_ATTRIBUTION_READY=PASS
DENSEKP_PRIMARY_COST=dense DenseKP Jacobian regression (Q66/Q76)
DENSEKP_REGRESSION_BOTTLENECK=YES
SUPPORT_COUNT_EXPLAINS_GAP=PARTIAL
NEXT_OPTIMIZATION_EXPERIMENT=sparse/fused DenseKP Jacobian regression bake-off
```

D138 在 frame 513 上的 median 为 129.37 ms（7.73 FPS），S138 为
23.81 ms（42.01 FPS）。D138 慢 5.43 倍、增加 105.56 ms。其中当前
DenseKP value/Jacobian regression 合计约 96.18 ms，占 D138 pipeline 的
74.34%，并解释 D138 与 S138 差距的约 91.11%。因此新增 regression
operator 是当前首要瓶颈；204 support 的 pose-offset/LBS 成本确实更高，
但只解释剩余差距的一部分。

## 实验条件

- frame 513，cam2 + cam4，T=1。
- 同一 Release executable：Eigen vectorization ON、`-march=native`、单线程。
- O4 `optimized_native_stable_p`，fast-math/LTO/forward-reuse 均关闭。
- 每个 policy 先 warmup，再正式运行 20 次。
- 采用 D138/S138/S100 的轮转交错顺序，避免按 policy 分批造成温度漂移。
- 表中系统时间为 20 次的 median；分位数另外列出。
- C++ 没有 padding，padded count 等于 effective count。

frame 513 的实际点数：

| Policy | cam2 support/obs | cam4 support/obs | padded/effective |
| --- | ---: | ---: | ---: |
| D138 | 204/138 | 204/138 | 138/138 |
| S138 | 137/137 | 129/129 | 137/137、129/129 |
| S100 | 99/99 | 93/93 | 99/99、93/93 |

DenseKP 的 138 个 observation 不表示只 forward 138 个 SMPL-X vertices；
D138 始终计算 204 个 support vertices。

## System performance

| Policy | support | obs | LM lin/obj calls | Total ms | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| D138 | 204/camera | 138/camera | 29/34 | 129.37 | 7.73 |
| S138 | 137、129 | 137、129 | 29/37 | 23.81 | 42.01 |
| S100 | 99、93 | 99、93 | 29/34 | 17.87 | 55.95 |

LM calls 是 Stage1、Pose、PoseShape 的合计。accepted steps 三者均为 26；
rejected trials 分别为 D138=5、S138=8、S100=5。D138 并没有更多的总
objective calls，因此整体变慢不能归因于 LM 多跑了几次。

| Policy | p25 ms | median ms | p75 ms | Stage1 | Pose | PoseShape |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D138 | 128.96 | 129.37 | 130.02 | 2.66 | 75.24 | 51.18 |
| S138 | 23.32 | 23.81 | 24.13 | 2.73 | 12.52 | 8.31 |
| S100 | 17.61 | 17.87 | 18.61 | 1.94 | 9.45 | 6.51 |

## Pose attribution

均为一次完整 fitting 中的 median total ms；括号内为调用次数与
median ms/call。

| Component | D138 | S138 | S100 |
| --- | ---: | ---: | ---: |
| objective | 3.91 | 1.59 | 1.16 |
| analytic Jacobian | 66.39 | 8.13 | 6.14 |
| pose-offset derivative | 5.51 (12, 0.459) | 3.41 (12, 0.284) | 2.45 (12, 0.204) |
| LBS derivative | 5.06 (12, 0.422) | 3.30 (12, 0.275) | 2.43 (12, 0.203) |
| DenseKP R value | 2.07 (26, 0.080) | N/A | N/A |
| DenseKP R Jacobian Q66 | 54.47 (12, 4.539) | N/A | N/A |
| normal equations | 1.10 | 0.89 | 0.66 |
| solve | 0.24 | 0.20 | 0.20 |

Q66 regression 占 D138 Pose Jacobian 的 82.04%。`jac.erp_chain` 的旧计时
范围包含 regression；上表使用新增的独立 regression range，不能再把两者
直接相加。

扣除约 0.079 ms 的 value regression 后，D138 的 ERP projection 约
0.007 ms/call，与 S138 的 0.0064 ms/call 同量级。NE 也只由 0.89 ms
增加到 1.10 ms。这符合 ERP/residual/IRLS/NE 主要随 observation count
缩放的预期。

## PoseShape attribution

| Component | D138 | S138 | S100 |
| --- | ---: | ---: | ---: |
| objective | 1.26 | 0.59 | 0.42 |
| analytic Jacobian | 47.47 | 6.07 | 4.75 |
| pose-offset derivative | 1.74 (4, 0.434) | 1.14 (4, 0.285) | 0.82 (4, 0.205) |
| nested Pose + combined LBS derivative | 4.34 | 2.78 | 1.99 |
| DenseKP R value | 0.64 (8, 0.080) | N/A | N/A |
| DenseKP R Jacobian Q66 | 18.15 (4, 4.538) | N/A | N/A |
| DenseKP R Jacobian Q76 | 20.85 (4, 5.214) | N/A | N/A |
| normal equations | 1.06 | 0.95 | 0.76 |
| Schur solve | 0.09 | 0.08 | 0.08 |

PoseShape 当前先构造嵌套 Pose Q66 Jacobian，再构造 combined Q76 Jacobian；
因此 DenseKP 路径同时支付 Q66 和 Q76 两次 regression。两者合计占
PoseShape Jacobian 的 82.17%。beta10 增加的直接表现是 Q76 每次
5.214 ms，相比 Q66 每次 4.538 ms 高约 14.9%。

## Support-dependent kernel

以下为 Pose Jacobian 的 median ms/call：

| Kernel | S100 | S138 | D138 (204 support) | D138/S138 |
| --- | ---: | ---: | ---: | ---: |
| pose-offset derivative | 0.204 | 0.284 | 0.459 | 1.62× |
| selected-transform/LBS derivative | 0.203 | 0.275 | 0.422 | 1.54× |
| LBS downstream/final vertex derivative | 0.068 | 0.096 | 0.139 | 1.45× |

这些 kernel 基本随 support count 增长，证明 204 support 有真实成本；但其
增量远小于 DenseKP Jacobian regression 的约 4.5–5.2 ms/call。因此
DenseKP D138 的模型 forward/derivative 规模更接近 204-point surface，
但当前总耗时不能只用 204/138 解释。

## Regressor 结构与 ownership audit

```text
shape                 = [138, 204]
nnz                   = 204
density               = 0.7246377%
nnz/row min/mean/max  = 1 / 1.47826 / 3
column usage min/mean/max = 1 / 1 / 1
unused support columns = 0
```

当前 `apply_densekp_operator()` 仍遍历全部 `138×204×component×Q`，包括
超过 99% 的零权重。这解释了 Q66/Q76 regression 的高成本，并给后续
sparse/fused bake-off 提供了充分依据。

审计结果：

- regressor asset 每个进程只加载一次，以 immutable `shared_ptr` 在 camera、
  stage 和 multi-frame 间复用。
- support IDs 同样只读取一次，没有在 LM trial 内重新生成或 gather。
- 输入 support vertices/Jacobian 不做 deep copy。
- 每次 operator 调用会通过 `output.assign()` 分配/清零结果 buffer。
- value、Q66、Q76 各自生成输出；PoseShape 还同时执行嵌套 Q66 和 combined
  Q76 regression。
- 本轮只记录这些事实，没有修改或优化实现。

## T=8 轻量确认（704–711）

每项运行 5 次，表中为 median：

| Policy | Stage1 ms | Pose ms | PoseShape ms | Total ms | ms/frame | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D138 | 33.35 | 613.82 | 449.61 | 1109.29 | 138.66 | 7.21 |
| S138 | 29.60 | 100.18 | 74.14 | 206.22 | 25.78 | 38.79 |
| S100 | 14.32 | 74.78 | 57.43 | 146.89 | 18.36 | 54.46 |

三条路径均保持单一 shared beta block，参数维度仍为 `75×8+10=610`。

## Correctness gate

- profiling 前后 D138 的 Stage1/Pose/PoseShape trajectory、所有 final
  parameters 和 final physical beta 均逐字节一致。
- S100 三阶段 trajectory 与既有 O4 canonical 输出逐字节一致。
- `FIRST_DIVERGENCE=NONE`。
- `TOLERANCE_CHANGE_COUNT=0`。
- C++ tests：32/32 PASS。
- `git diff --check`：PASS。

## 下一步排序

1. **B：优化 DenseKP regression。** 首先做 sparse/fused regression
   bake-off；当前 0.72% density 使 dense contraction 明显不合理。
2. **A：优化 204-support SMPL-X derivative。** regression 降低后，
   pose-offset 和 LBS 才会成为下一层热点。
3. **C：降低 LM call count。** 当前 D138 并没有更多总 calls，不是本轮差距
   主因。

本轮没有实现 sparse kernel、fusion、support pruning、threading 或任何
算法改动。
