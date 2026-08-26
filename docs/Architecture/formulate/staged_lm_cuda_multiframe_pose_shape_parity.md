# Multi-frame CUDA PoseShape parity

## 测试范围

使用真实 DenseKP C++ fixtures：

- `T=1`：frame 513；
- `T=4`：frames 513--516；
- `T=8`：frames 704--711。

CPU 与 CUDA 使用相同 loss、IRLS、参数布局、LM controller、Schur solver 和 tolerance。

## 结果

| Fixture | CUDA PoseShape | 控制决策 | 首处分歧 | 说明 |
|---|---:|---:|---:|---|
| T=1 / 513 | PASS | exact | NONE | 与 single-frame CUDA trajectory、final parameters 均为 byte-exact |
| T=4 / 513--516 | PASS | exact | NONE | accepted/rejected、solver、status、trial 数一致 |
| T=8 / 704--711 | PASS | exact | NONE | accepted/rejected、solver、status、trial 数一致 |

T=4 和 T=8 的 CUDA/CPU 浮点 normal equations 会有预期的末位差异，因此这里的
`FIRST_DIVERGENCE=NONE` 指 LM 控制轨迹没有分支分歧，不宣称 CPU/CUDA 数值 bitwise
一致。未修改任何 tolerance。

## Correctness-first benchmark

| Fixture | CPU PoseShape | CUDA PoseShape | CPU total | CUDA total |
|---|---:|---:|---:|---:|
| T=4 | 70.54 ms | 50.21 ms | 183.52 ms | 321.31 ms |
| T=8 | 137.55 ms | 99.93 ms | 350.34 ms | 432.62 ms |

PoseShape evaluation 已有收益，但当前完整 CUDA pipeline 仍受逐帧 evaluator
construction、冷启动和同步开销影响。这一轮没有做 kernel micro optimization。

```text
CONTROL_DECISIONS_EXACT=YES
FIRST_DIVERGENCE=NONE
TOLERANCE_CHANGE_COUNT=0
```
