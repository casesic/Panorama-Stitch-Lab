# DenseKP sparse / fused regression bake-off

## 最终结论

```text
DENSEKP_SPARSE_FUSED_REGRESSION_BAKEOFF_READY=PASS

DENSEKP_SPARSE_NNZ=204
DENSEKP_SPARSE_VALUE_PARITY=PASS
DENSEKP_SPARSE_Q66_PARITY=PASS
DENSEKP_SPARSE_Q76_PARITY=PASS

DENSEKP_FITTING_TRAJECTORY_EXACT=YES
SAM3D_T1_TRAJECTORY_EXACT=YES

FIRST_DIVERGENCE=NONE
TOLERANCE_CHANGE_COUNT=0

BEST_VARIANT=sparse materialized (B)
SYSTEM_SPEEDUP=3.91x (frame513), 3.30x (T=8)
NEXT_BOTTLENECK=LBS derivative, then pose-offset derivative
```

production O4 `optimized_native_stable_p` 对 DenseKP 选择 B；原 dense A 和
fused C 保留为显式 experimental/diagnostic path。SAM3D 不经过 DenseKP
operator。

## 实现范围

固定 sparse representation：

```text
row_offsets[139]
support_indices[204]
sparse_weights[204]
```

它在加载 immutable regressor 时构建一次，并由所有 camera、stage 和 frame
共享。每行严格保持原 dense regressor 的 column order。

- A `dense`：原 `138×204` dense contraction。
- B `sparse`：row-wise 遍历 204 个非零 contribution，materialize points/J。
- C `sparse_fused`：不 materialize DenseKP 3D Jacobian，逐 row 生成 ERP
  UV Jacobian；value path 与 B 共用 sparse value regression。

没有修改 LM、IRLS、SMPL-X forward、204 support、Schur、multi-frame
结构或 SAM3D path。

## Correctness

A/B/C 对以下结果逐元素/逐字节一致：

- value regression；
- Q66、Q76 3D derivative；
- ERP UV Jacobian；
- residual、H、g（由 fitting trajectory exact 覆盖）；
- Stage1、Pose、PoseShape trajectory；
- final parameters、physical beta。

Sparse 与 dense 均按原 column order 使用 `std::fma`；跳过零权重没有改变
accumulator。因此本轮无需 tolerance 调整。

## Kernel microbenchmark

O4 Release、Eigen vectorization、`-march=native`、单线程；warmup 后每个
variant 50 次 pipeline replay，对其中数百次真实 operator call 统计 median。

| Variant | value ms/call | Q66 ms/call | Q76 ms/call |
| --- | ---: | ---: | ---: |
| A dense | 0.0776 | 4.5410 | 5.2079 |
| B sparse | 0.00218 | 0.0159 | 0.0177 |
| C sparse_fused | 0.00223 | 0.0440 | 0.0503 |

C 的 Q66/Q76 数字包含 fused ERP derivative chain；对应的 B
materialized-regression + ERP chain 分别为 0.0299/0.0295 ms，因此 B 的
完整 chain 仍比 C 快。

主要 speedup：

| Comparison | value | Q66 | Q76 |
| --- | ---: | ---: | ---: |
| A → B | 35.6× | 285.5× | 294.0× |
| A → C | 34.8× | 103.3× | 103.5× |
| B → C | 0.98× | 0.36× | 0.35× |

这里 B→C 小于 1 表示 fusion 更慢。C 对每个 parameter 重新执行小型 sparse
accumulation并立即投影，破坏了 B materialized buffer 后续连续访问的优势。

完整 frame513 中，独立计时的 DenseKP regression：

```text
A dense   96.277 ms
B sparse   0.446 ms
下降       99.54%
speedup    215.7x
```

## Full pipeline：frame 513

同一 O4 executable，A/B/C 交错运行；每个 variant 20 次，表中为 median。

| Variant | Stage1 ms | Pose ms | PoseShape ms | Total ms | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| A dense | 2.65 | 75.40 | 51.09 | 129.58 | 7.72 |
| B sparse | 2.65 | 18.65 | 11.46 | 33.16 | 30.15 |
| C sparse_fused | 2.66 | 18.69 | 11.56 | 32.91 | 30.39 |

| Variant | p25 total | median total | p75 total |
| --- | ---: | ---: | ---: |
| A | 129.04 | 129.58 | 130.30 |
| B | 32.92 | 33.16 | 33.43 |
| C | 32.74 | 32.91 | 33.27 |

C 相比 B 的总时间只低约 0.26 ms（0.78%），但 Pose 和 PoseShape
Jacobian median 均略慢；微小 total 差异来自 Stage1/control 波动，不能证明
复杂 fusion 有真实系统收益。因此按预设 promotion policy 选择 B。

## B 优化后的归因

| Component | median total ms |
| --- | ---: |
| Pose Jacobian | 11.99 |
| Pose objective | 2.82 |
| Pose NE | 0.92 |
| Pose solve | 0.24 |
| PoseShape Jacobian | 8.43 |
| PoseShape objective | 0.93 |
| PoseShape NE | 1.06 |
| Schur solve | 0.09 |
| DenseKP regression（全部） | 0.45 |

新的 kernel Top3：

1. LBS derivative family：约 9.39 ms/frame；
2. pose-offset derivative：约 7.27 ms/frame；
3. value forward/objective：约 3.75 ms/frame。

DenseKP regression 已不再是 Top1，甚至不在 Top3。本轮到此停止，不继续
优化这些新热点。

## T=8：704–711

baseline A 与 production B 各运行 5 次，表中为 median：

| Variant | Stage1 ms | Pose ms | PoseShape ms | Total ms | ms/frame | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A dense | 28.52 | 610.56 | 464.78 | 1106.05 | 138.26 | 7.23 |
| B sparse | 32.02 | 154.36 | 147.97 | 334.76 | 41.84 | 23.90 |

T=8 参数、每帧输出和唯一 shared beta block 均与 dense baseline 逐字节一致。

## Production 与保留策略

```text
optimized_native_stable_p          -> sparse materialized B (DenseKP only)
experimental_densekp_dense         -> dense baseline A
experimental_densekp_sparse        -> explicit B diagnostic alias
experimental_densekp_sparse_fused  -> fused C diagnostic
```

B 简单、快、保持 materialized observation Jacobian，且系统收益明确；C 的
复杂度没有换来可信的额外收益，因此不 promote。

## 验收

- DenseKP A/B/C fitting trajectory exact。
- SAM3D frame513 三阶段 trajectory 与既有 O4 canonical 逐字节一致。
- FIRST_DIVERGENCE=NONE。
- TOLERANCE_CHANGE_COUNT=0。
- C++ tests 全部 PASS。
- `git diff --check` PASS。
