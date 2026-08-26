# CUDA Pose Jacobian parity 与 benchmark

## 测试对象

- fixture：真实 DenseKP frame 513
- GPU：RTX 3090
- 路径：CUDA Pose residual/Jacobian + CPU normal equations/LM controller
- tolerance 修改：0

## 静态线性化 parity

```text
residual max_abs       4.8828125e-4 px
Jacobian max_abs       1.373291015625e-4
Jacobian relative L2   1.1938526567e-7
```

差异来源是 CPU libm 与 CUDA float32 transcendental/FMA 的正常跨 backend
舍入差异，没有调整 tolerance。

## 完整 Pose LM

控制轨迹一致：两条路径均为 5 次 accepted、1 次 rejected，所有 trial 的
accept/reject、求解 backend、终止状态均一致。最终 objective 分别为
11.052181（CPU）和 11.052165（CUDA Pose Jacobian），最终参数 max_abs 为
5.2452087e-6、relative L2 为 9.3220822e-7。

但是结果不是 bitwise exact。第一处数值差异发生在 initial lambda：

```text
CPU   15.865358847438010
CUDA  15.865352737753163
```

因此严格按验收词义记录：

```text
TRAJECTORY_EXACT=NO
FIRST_DIVERGENCE=POSE_INITIAL_LAMBDA
FINAL_PARAMETERS_EXACT=NO
CONTROL_DECISIONS_EXACT=YES
TOLERANCE_CHANGE_COUNT=0
```

如果要求跨 CPU/CUDA bitwise trajectory exact，只能在 GPU 结果之后重新使用
CPU Jacobian/normal equations，或复刻 CPU transcendental 实现；两者都违背本阶段
迁移 CUDA Jacobian 的目标，因此没有采用。

## 未优化 benchmark

单次双相机静态线性化：

```text
CPU Pose Jacobian    3.9186 ms
CUDA Pose Jacobian  11.1379 ms
```

完整 Pose LM 的累计 Jacobian 时间：CPU 24.14 ms，当前 CUDA 224.53 ms。
CUDA 较慢主要来自每次调用的分配、H2D/D2H、逐 kernel synchronize 和朴素导数
kernel；这是 correctness-first baseline，不是性能结论。

## 复现

```bash
staged_lm_pipeline_reference_main \
  --fixture <frame-513-cpp-fixture> \
  --output <parity-output> \
  --cuda-pose-jacobian-parity \
  --evaluator-backend cpu

staged_lm_pipeline_reference_main \
  --fixture <frame-513-cpp-fixture> \
  --output <cuda-lm-output> \
  --evaluator-backend cuda
```

