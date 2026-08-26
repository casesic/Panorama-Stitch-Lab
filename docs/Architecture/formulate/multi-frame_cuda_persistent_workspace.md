# Multi-frame CUDA PoseShape persistent workspace

## 改动范围

本轮只移动 device buffer 的分配时机和生命周期，不改变 kernel、参数布局、数学、
solver、LM controller 或 tolerance。

`CudaMultiFramePoseShapeWorkspace` 在 evaluator 构造期间，为每帧持有：

- 一份 persistent shape value workspace；
- cam2/cam4 value-forward workspace；
- cam2/cam4 PoseShape linearization workspace；
- 固定的 base joints、joint basis、base vertices、beta shapedirs；
- shape、dynamic vertices、Pose Jacobian 和 beta Jacobian 所需的 pool blocks。

value 与 linearization 使用不同的 pool，避免两条路径因 buffer 顺序和尺寸不同而反复
扩容。所有 pool block 按已知尺寸直接 reserve，不通过执行额外 kernel 来预热。

## Allocation before/after

T=8，704--711，Nsight Systems 2023.1.2：

| 指标 | Before | After |
|---|---:|---:|
| LM runtime `cudaMalloc` calls | 2224 | 0 |
| LM runtime `cudaMalloc` API | 7.33 ms | 0 ms |
| LM runtime `cudaFree` calls | 1728 | 0 |
| LM runtime `cudaFree` API | 4.92 ms | 0 ms |
| workspace-construction `cudaMalloc` | 240 calls / 1.76 ms | 1248 calls / 5.73 ms |

After 的分配全部集中在 evaluator workspace 构造阶段，并在 evaluator 销毁时统一释放。
forward、linearize、normal-equation NVTX ranges 内没有 `cudaMalloc/cudaFree`。

## End-to-end PoseShape benchmark

未插桩、单次真实 pipeline：

| Fixture | Before | After | 改善 |
|---|---:|---:|---:|
| T=4 / 513--516 | 50.21 ms | 37.10 ms | 26.1% |
| T=8 / 704--711 | 99.93 ms | 79.78 ms | 20.2% |

这里包含 workspace construction 和最终销毁，因此收益小于 runtime allocation
归零后的局部收益。若 evaluator 跨连续 batch 长期复用，构造成本可以进一步摊薄。

## 正确性

T=1、T=4、T=8 的优化前后 trajectory CSV 与 final parameter binary 均 byte-exact。

```text
CONTROL_DECISIONS_EXACT=YES
FIRST_DIVERGENCE=NONE
TOLERANCE_CHANGE_COUNT=0
```

本轮没有做 kernel fusion、异步 memcpy 或同步策略修改。
