# CUDA Pose evaluator GPU model residency

## 实际 storage 边界

当前 Pose evaluator 的 C++ 输入已经是 Python/offline 阶段裁剪后的最小 SMPL-X
模型，并不直接持有完整 `v_template`、全量 `shapedirs` 或 joint regressor。因而
本次常驻的是与这些资产在当前 Pose 阶段等价、且 kernel 实际消费的数据：

- `selected_v_shaped`；
- `selected_posedirs`；
- `selected_lbs_weights`；
- shaped joints 与 parent tree；
- DenseKP Sparse-B row offsets、support indices、weights、DenseKP IDs；
- observation targets。

完整 `v_template/shapedirs/joint regressor` 属于后续 CUDA PoseShape/shared-model
storage 的输入，不能在当前 Pose 接口中伪造一份未被使用的数据。

每个相机的 `CudaPoseWorkspace` 在 evaluator construction 时上传上述固定资产，
随后 `forward/linearize` 只上传当前 axis-angle 和 camera translation。所有 Jacobian
中间量保持 device-resident，CPU normal equations 只接收 residual 和 `J_active`。

## Frame 513 benchmark

warm 双相机单次 linearization：

| 项目 | residency 前 | residency 后 |
|---|---:|---:|
| kernel | 0.8928 ms | 0.8555 ms |
| H2D | 包含于 3.7775 ms | 0.0226 ms |
| D2H | 包含于 3.7775 ms | 0.1097 ms |
| memcpy total | 3.7775 ms | 0.1323 ms |
| allocation | 0.0031 ms | 0.0014 ms |
| linearization total | 4.8095 ms | 1.0945 ms |

完整 Pose LM、6 次双相机 linearization 的累计 Jacobian 时间：

```text
CPU baseline                    24.14 ms
CUDA correctness baseline      224.53 ms
CUDA persistent workspace       16.53 ms
CUDA model residency             6.57 ms
```

## Correctness

residency 前后的 CUDA trajectory 和 final parameters byte-exact；accepted/rejected、
lambda/rho、termination 均未改变。CPU/CUDA 固有 parity 仍为：

```text
residual max_abs       4.8828125e-4 px
Jacobian max_abs       1.3732910e-4
Jacobian relative L2   1.1938527e-7
```

验收：

```text
CONTROL_DECISIONS_EXACT=YES
FIRST_DIVERGENCE=NONE
TOLERANCE_CHANGE_COUNT=0
```

