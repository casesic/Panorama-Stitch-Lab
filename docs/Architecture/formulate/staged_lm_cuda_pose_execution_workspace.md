# CUDA Pose Jacobian persistent workspace

## 目的与边界

本次只优化 CUDA evaluator 的执行开销，不修改 Pose Jacobian 数学、参数布局、
IRLS、normal equations、LM controller、solver 或 tolerance，也没有进行 kernel
fusion 和 approximate math。

每个 `CudaPoseEvaluator` 为 cam2/cam4 各持有一个 `CudaPoseWorkspace`：

```text
CudaPoseEvaluator construction
  -> create two workspaces
  -> establish CUDA context
  -> allocate/warm complete linearization buffer sequence

forward / linearize
  -> reset workspace cursor
  -> reuse device allocations
  -> keep Jacobian intermediates device-resident
  -> D2H only evaluator boundary outputs
```

Standalone reference API 仍会返回 materialized 中间导数，便于分层 correctness
诊断；persistent production API 只下载 residual 所需 value 和 `J_active`，因为 CPU
normal equations 不消费 `d_rotation/d_FK/d_LBS` 等中间数组。

## Frame 513 benchmark

累计 6 次双相机 Pose linearization：

```text
CPU baseline                         24.14 ms
CUDA correctness baseline           224.53 ms
CUDA persistent workspace             19.84 ms
CUDA workspace + compact D2H           16.53 ms
```

相对原 CUDA correctness baseline，累计 Jacobian 时间减少约 92.6%。当前结果不是
CUDA kernel 本身的最终性能上限，而是消除了 CUDA context/allocator cold cost 和
不必要中间结果回传后的 execution-overhead baseline。

warm 双相机单次 linearization 分桶：

```text
total          4.8095 ms
kernel         0.8928 ms
memcpy         3.7775 ms
allocation     0.0031 ms
other          0.1361 ms
```

这说明 persistent allocation 已不再是 steady-state 瓶颈。静态 per-camera 模型
资产常驻后的结果见 `staged_lm_cuda_pose_model_residency.md`。

## Correctness regression

workspace 优化前后的 CUDA 输出：

```text
trajectory bytes exact       YES
final parameter bytes exact  YES
accept/reject unchanged      YES
termination unchanged        YES
tolerance changes             0
```

CPU/CUDA 固有 parity 保持不变：

```text
residual max_abs       4.8828125e-4 px
Jacobian max_abs       1.3732910e-4
Jacobian relative L2   1.1938527e-7
```
