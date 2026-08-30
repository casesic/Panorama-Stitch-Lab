# Staged-LM CUDA value-forward evaluator

## 架构

CUDA backend 建立在现有 `EvaluatorBackend` 边界后方，CPU LM controller 没有
修改：

```text
run_lm (CPU)
  - lambda update
  - solve
  - accept/reject
  - termination
          |
          v
EvaluatorBackend
  +-- CpuPoseEvaluator / CpuPoseShapeEvaluator
  +-- CudaPoseEvaluator / CudaPoseShapeEvaluator
```

backend switch 为 `cpu|cuda`，默认是 `cpu`：

```bash
--evaluator-backend cpu
--evaluator-backend cuda
```

当前 `cuda` 已实现 Pose/PoseShape value、Jacobian 和 structured normal equations。
不存在静默的 Jacobian CPU fallback。Production 默认仍是：

```text
kernel_policy=optimized_native_stable_p
evaluator_backend=cpu
```

## CUDA forward 范围

Pose：

```text
axis-angle rotation
-> pose feature
-> pose offset
-> v_posed
-> FK
-> LBS
-> support vertices
-> DenseKP Sparse-B regression
-> ERP projection
-> raw residual
```

PoseShape 另外在 CUDA 上计算：

```text
latent beta -> bounded beta
-> shaped joints
-> selected dynamic v_shaped
-> Pose CUDA forward
```

本阶段使用直接 `cudaMalloc/cudaMemcpy` 和自然分阶段 kernel，没有 kernel fusion、
持久 device cache、自定义 allocator 或性能优化。CPU-only build 使用 stub，并在调用
CUDA backend 时返回 `CUDA_EVALUATOR_NOT_BUILT`。

## 构建

CPU-only：

```bash
cmake -S cpp/staged_lm -B build/staged_lm_cpu \
  -DCMAKE_BUILD_TYPE=Release -DSTAGED_LM_ENABLE_CUDA=OFF
```

RTX 3090 / sm_86 CUDA build：

```bash
cmake -S cpp/staged_lm -B build/staged_lm_cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTAGED_LM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86
```

Forward parity：

```bash
build/staged_lm_cuda/staged_lm_pipeline_reference_main \
  --fixture <cpp_fixture> --output <output> \
  --evaluator-backend cpu --cuda-forward-parity
```

这里 CPU backend 生成可信 LM trajectory；收敛后的 Pose/PoseShape 状态再同时送入
CPU/CUDA value forward 做中间结果对比。

## 当前边界

- CUDA Schur 或 LM solve；
- 跨 frame kernel fusion 或 batched GEMM；
- GPU 端 LM controller。

multi-frame shared-beta PoseShape 已支持 CUDA evaluator；Schur 和 LM controller 仍在
CPU，详见
[`staged_lm_cuda_multiframe_pose_shape_backend.md`](staged_lm_cuda_multiframe_pose_shape_backend.md)。
