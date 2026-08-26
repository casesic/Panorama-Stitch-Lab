# CUDA Pose normal equations parity 与 benchmark

测试：DenseKP frame 513，双相机，RTX 3090。

## H/g parity

完整 CPU pipeline 与 CUDA residual/J + CUDA accumulation：

```text
H max_abs       5.4040e-2
H relative L2   2.4921e-6
g max_abs       2.9187e-3
g relative L2   2.3898e-4
```

该指标包含上游 CPU/CUDA residual/J 的既有差异；`g=J^T r` 还会放大 cancellation。
报告文件同时记录“同一 CUDA residual/J 输入，CPU accumulation 对 CUDA
accumulation”的隔离指标。

同一 CUDA residual/J 输入下：

```text
H max_abs       1.1497e-4
H relative L2   1.0767e-9
g max_abs       4.9912e-7
g relative L2   3.8181e-8
```

因此 builder 自身与 CPU accumulation 高度一致，完整链路中更大的误差主要来自
既有的 CPU/CUDA forward/Jacobian 差异，而不是 normal-equation 公式错误。

## 完整 LM

迁移前后的 CUDA Pose LM 均为：5 accepted、1 rejected、相同 trial 分支、相同
Cholesky backend、相同 termination。lambda/rho 的浮点末位不同，但没有控制分支
差异：

```text
CONTROL_DECISIONS_EXACT=YES
FIRST_DIVERGENCE=NONE
TOLERANCE_CHANGE_COUNT=0
```

这里 `FIRST_DIVERGENCE` 指 LM 状态机/分支；不表示跨 CPU/GPU 浮点结果 bitwise
一致。

## Benchmark

单次双相机 normal equations：

```text
before CPU normal equations     0.3693 ms
after GPU accumulation          0.2999 ms
after H/g D2H                   0.0551 ms
after total                     0.3937 ms
```

完整 6 次 Pose linearization：

```text
before Jacobian D2H + CPU NE
  Jacobian                 6.57 ms
  normal equations         1.52 ms

after GPU NE + H/g D2H
  Jacobian                 5.72 ms
  normal equations         2.16 ms
```

本轮主要完成数据路径迁移：完整 Jacobian D2H 已归零，只回传约 77 KiB 的双相机
FP64 H/g。当前朴素 H/g kernel 尚未优化，整体 LM 时间从约 10.99 ms 到 10.68 ms，
收益较小但路径已经为后续 reduction 优化建立正确基线。
