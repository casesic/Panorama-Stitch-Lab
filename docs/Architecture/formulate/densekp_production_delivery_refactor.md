# DenseKP production delivery 重构

## 新运行链

```text
双鱼眼原图
  → calibration LUT cache
  → 两张普通 ERP 图
  → DenseKP inference / person match
  → 仅旋转 138 个 ERP 点
  → ERP_ROTATE observation JSON
  → runtime exporter（注入离线 C++ model bundle）
  → C++ Stage1 / Pose / PoseShape LM
  → optimized DenseKP138（ERP_ROTATE）
```

CPW 是下游消费者，不属于 DenseKP→LM 的必要计算。

运行时不再调用 `prepare_staged_lm_cpp_input.py`，也不运行 Python SMPL-X、
Python LM、reference dump 或 trajectory 生成。C++ model bundle 包含当前帧已经
准备好的 204 support 模型数据；runtime exporter 只替换 DenseKP targets、valid
IDs 和 Stage1 对应的 DenseKP local rows。

## 坐标定义

- DenseKP inference 输入：普通 ERP 图，`5760 × 2880`。
- DenseKP inference 原始输出：ERP 坐标。
- LM observation：对 138 个点应用球面旋转矩阵后的 ERP_ROTATE 坐标。
- C++ 输出：138 对 ERP_ROTATE 坐标，顺序为
  `[x_cam2, y_cam2, x_cam4, y_cam4]`。
- `keypoint_value`、confidence 和其他观测权重均不进入 LM。

旧代码实际也是在普通 ERP 图上执行 DenseKP；额外生成的 ERP_ROTATE 整图只
用于保存/可视化，并没有送入 inference。因此删除整图 warp 后，同一组 ERP
DenseKP 点的坐标结果逐值一致，实测 `max_abs = 0`。

## Calibration cache

共享 cache 默认位于：

```text
~/.cache/pano_stitch/densekp_delivery/<calibration-hash>/
```

也可以通过 `--calibration_cache_dir` 指定。缓存内容包括：

- cam2/cam4 ERP `map_x/map_y`；
- cam2/cam4 ERP valid mask；
- 完整 calibration specification JSON。

hash 包含 camera group、cam2/cam4 Double-Sphere 参数、extrinsic、ERP 尺寸和
双鱼眼输入尺寸。任一项改变都会使用新的 cache 目录，不会复用旧 LUT。缓存
不包含逐帧 ERP JPEG，避免旧流程因有损 JPEG cache 改变 DenseKP inference。

## Production exporter

入口：

```text
src/lib_pano/fitting_io/densekp_runtime_export.py
```

示例：

```bash
conda run -n camerahmr python \
  src/lib_pano/fitting_io/densekp_runtime_export.py \
  --densekp-observation output/densekp_pairs_erp_rotate.json \
  --output-json output/densekp_observation.json \
  --base-fixture cache/frame_000513/cpp_model_bundle \
  --output-fixture output/runtime_fixture
```

`base-fixture` 必须是离线准备的 DenseKP-ready C++ model bundle。该步骤是模型/
HMR 初始化资产生产，不应放在逐帧 DenseKP runtime 内。

`debug078_densekp_fitted_cpw_composite.py` 已改用 `--base_cpp_fixture` 和该轻量
exporter，不再接受 HMR、Sam3D、bbox 或 device 参数，也不再调用 reference
prepare。

## 513 帧 CPU 实测

| 阶段 | 重构前 | 重构后 |
|---|---:|---:|
| DenseKP delivery，冷 cache | 35.64 s | 10.26 s |
| DenseKP delivery，命中 calibration cache | 9.47 s | 6.25 s |
| C++ input preparation | 8.25 s | 6.18 ms |
| C++ LM + optimized points | 80 ms | 61 ms |
| CPW | 约 4.23 s | 4.23 s |

不含 CPW 的稳态链路从约 `17.80 s/frame` 降至约
`6.32 s/frame`，即从 `0.056 FPS` 提升到约 `0.158 FPS`，约 `2.8×`。

当前 DenseKP delivery 仍以单帧独立进程运行，ORT session、detector 和两张
ERP remap 尚未跨帧 batch/常驻；这部分是剩余主要瓶颈。

## Correctness

- ERP→ERP_ROTATE point-only parity：`max_abs = 0`。
- calibration cache cold/warm DenseKP outputs：逐值一致。
- runtime exporter 与 reference oracle：Stage1、Pose、PoseShape trajectory、
  final parameters、beta 和 optimized DenseKP JSON 均 bitwise exact。
- C++ regression：11/11 PASS。
- debug078 CPW：PASS；与原 debug077 图像比较 P95 为 0，超过 10 灰度的像素
  占约 1.01%，变化局限于 correspondence warp 区域。

```text
DENSEKP_PRODUCTION_DELIVERY_REFACTOR_READY=PASS
```
