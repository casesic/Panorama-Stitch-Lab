# CUDA PoseShape parity report

测试数据：DenseKP frame 513，双相机，RTX 3090。

## Value/Jacobian

```text
residual max_abs          4.8828125e-4 px
beta Jacobian max_abs     2.2888184e-5
beta Jacobian relative L2 1.8497652e-7
```

## Structured normal equations

```text
A max_abs / relative L2        3.8662e-1 / 6.4191e-6
B max_abs / relative L2        1.9047e-2 / 6.7472e-6
C max_abs / relative L2        5.7294e-4 / 1.4768e-6
g_pose max_abs / relative L2   2.2304e-3 / 4.5876e-4
g_beta max_abs / relative L2   9.3722e-5 / 3.0191e-5
```

梯度相对指标受 `J^T r` cancellation 影响更明显；完整 LM 控制轨迹用于最终硬验证。

## 完整 LM

CPU 与 CUDA 均为一次 accepted trial、Schur solve 成功、相同 termination。所有
accept/reject 分支一致；lambda/rho 仅有跨 backend 浮点末位差异。

```text
CONTROL_DECISIONS_EXACT=YES
FIRST_DIVERGENCE=NONE
TOLERANCE_CHANGE_COUNT=0
```

`FIRST_DIVERGENCE` 指控制状态机分支，不表示 CPU/CUDA 数值 bitwise exact。

