# Multi-frame CUDA PoseShape performance attribution

## 范围与方法

本报告只分析 multi-frame shared-beta PoseShape evaluator，不包含 Stage1、Pose、CPU
Schur 和 LM 控制。使用 Nsight Systems 2023.1.2，并用纯诊断 NVTX range 隔离：

```text
workspace construction
forward
linearize
normal equations
```

测试设备为 NVIDIA GeForce RTX 3090。fixture 分别为 513--516（T=4）和
704--711（T=8）。NVTX 只增加 profiler 标记，没有改变 kernel、数学或执行顺序。

## 总体结果

未插桩基准：

| Fixture | CPU PoseShape | CUDA PoseShape | CUDA 每帧 |
|---|---:|---:|---:|
| T=4 | 70.54 ms | 50.21 ms | 12.55 ms |
| T=8 | 137.55 ms | 99.93 ms | 12.49 ms |

Nsight 插桩运行：

| 项目 | T=4 | T=8 | 每帧趋势 |
|---|---:|---:|---:|
| marked wall time | 51.71 ms | 97.07 ms | 12.93 / 12.13 ms |
| kernel launch count | 672 | 1344 | 168 |
| GPU kernel time | 15.99 ms | 30.43 ms | 4.00 / 3.80 ms |
| synchronous memcpy API | 19.48 ms | 38.25 ms | 4.87 / 4.78 ms |
| cudaMalloc API | 5.88 ms | 9.09 ms | 1.47 / 1.14 ms |
| cudaFree API | 3.74 ms | 4.92 ms | 0.93 / 0.62 ms |
| kernel-launch API | 2.62 ms | 4.77 ms | 0.66 / 0.60 ms |
| sync API minus kernel execution | 1.70 ms | 3.49 ms | 0.43 / 0.44 ms |
| cudaMemset API | 0.15 ms | 0.30 ms | 0.04 ms |
| remaining host/per-frame overhead |约 2.1 ms |约 5.8 ms |约 0.5 / 0.7 ms |

`cudaDeviceSynchronize` 的 API 时间包含其等待的 kernel 时间，不能与 GPU kernel time
直接相加。因此表中 synchronization overhead 使用
`synchronize API wall - GPU kernel execution`，用于近似表示纯同步/调度成本。

## 分阶段

### T=4

| Range | 调用次数 | wall | launches | kernel time |
|---|---:|---:|---:|---:|
| workspace | 1 | 3.92 ms | 0 | 0 ms |
| forward | 2 | 14.74 ms | 152 | 1.46 ms |
| linearize | 2 | 26.49 ms | 488 | 9.56 ms |
| normal equations | 2 | 6.57 ms | 32 | 4.97 ms |

workspace range 中有 120 次 `cudaMalloc` 和 80 次 H2D copy；其 wall 为 3.92 ms，
其中 `cudaMalloc` API 为 0.85 ms、copy API 为 2.84 ms。运行期间 forward/linearize
仍产生 1112 次 `cudaMalloc`，说明当前 persistent workspace 还没有覆盖 shape value 与
beta derivative 的全部临时 buffer。

### T=8

| Range | 调用次数 | wall | launches | kernel time |
|---|---:|---:|---:|---:|
| workspace | 1 | 7.77 ms | 0 | 0 ms |
| forward | 2 | 25.29 ms | 304 | 2.83 ms |
| linearize | 2 | 49.38 ms | 976 | 17.83 ms |
| normal equations | 2 | 14.63 ms | 64 | 9.77 ms |

workspace range 中有 240 次 `cudaMalloc` 和 160 次 H2D copy。所有关键计数基本为
T=4 的两倍，说明当前实现是逐帧串行调度，没有形成跨帧 batch/fusion。

## GPU utilization

以 marked PoseShape wall 内 GPU kernel active time 计算 temporal utilization：

```text
T=4: 15.99 / 51.71 = 30.9%
T=8: 30.43 / 97.07 = 31.4%
```

该指标表示 GPU 有 kernel 在执行的时间占比，不等同于 kernel 内部 SM occupancy。
由于 workload 很短，`nvidia-smi` 的秒级采样不能可靠描述本阶段；因此不把瞬时
`nvidia-smi utilization.gpu` 当作结论。

## 结论

1. 当前每帧固定约 168 次 kernel launch，T=4 到 T=8 完全线性扩展；
2. kernel 本身只占约 31% wall，GPU 计算不是唯一瓶颈；
3. synchronous memcpy 是最大单项 host wall 成本，约 4.8 ms/frame；
4. workspace 只覆盖 Pose 的部分 buffer，shape forward/beta derivative 仍频繁
   `cudaMalloc/cudaFree`；
5. 每个 kernel 后的 `cudaDeviceSynchronize` 阻止跨 frame overlap；
6. 下一步若优化，应优先考虑完整 persistent workspace、减少 H2D/D2H，以及按 frame
   batching/减少同步；本轮没有实施这些优化。

其中 persistent workspace 已在后续独立步骤实现，结果见
[`multi-frame_cuda_persistent_workspace.md`](multi-frame_cuda_persistent_workspace.md)。

## 正确性

profiling 前后 T=4、T=8 的 trajectory CSV 和 final parameter binary 均 byte-exact。

```text
TRAJECTORY_UNCHANGED=YES
FINAL_PARAMETERS_EXACT=YES
CONTROL_DECISIONS_EXACT=YES
FIRST_DIVERGENCE=NONE
TOLERANCE_CHANGE_COUNT=0
```
