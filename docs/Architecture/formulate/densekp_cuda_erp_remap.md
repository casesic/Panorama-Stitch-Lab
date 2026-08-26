# DenseKP CUDA ERP remap

## 数学契约

CPU production reference：

```text
input       uint8 BGR, 2880 x 2880 x 3
map_x/y     float32, 2880 x 5760
output      uint8 BGR, 2880 x 5760 x 3
interpolate cv2.INTER_LINEAR
border      cv2.BORDER_CONSTANT, value=0
```

CUDA kernel复现 OpenCV `INTER_LINEAR` 的 5-bit fractional map coordinate 和
15-bit integer interpolation coefficient，而不是使用浮点 `grid_sample` 近似。
cam2/cam4 LUT、输入/output device buffer 在 runtime 生命周期内常驻；每帧只传输
两张鱼眼输入并回传两张 ERP 输出。

CLI：

```text
--erp-remap-backend cpu|cuda
--erp-remap-cuda-library /path/to/libdensekp_erp_remap_cuda.so
```

默认请求 `cuda`；library或CUDA不可用时明确 warning 并回退 CPU。CPU fallback仍使用
原 `cv2.remap` 路径。

## Bake-off

PyTorch `grid_sample` 被否决：单相机 `max_abs_diff=4`，双相机含传输约
75.8 ms，比 CPU 更慢。

OpenCV-compatible CUDA kernel：

```text
cam2 max_abs_diff = 0
cam4 max_abs_diff = 0
双相机 H2D + kernel + D2H = 20.56 ms（独立 steady bake-off）
```

## 1856–1928 集成实测

| 指标 | CPU remap | CUDA remap |
|---|---:|---:|
| ERP remap | 45.103 ms/frame | 36.520 ms/frame |
| steady total | 228.513 ms/frame | 214.147 ms/frame |
| steady FPS | 4.376 | 4.670 |
| 全批次 FPS（不含 startup） | 3.809 | 3.987 |

集成 remap 降低 `19.03%`，超过 10% 保留门槛；端到端稳态 FPS 提升 `6.71%`。

## 正确性

- CPU/CUDA ERP image `max_abs_diff=0`；
- 同一 DenseKP session 的 CPU/CUDA observation exact；
- CUDA production复跑 DenseKP output exact；
- optimized DenseKP exact；
- LM trajectory exact；
- observation、ERP/ERP_ROTATE定义和 C++ fitting 均未修改。

`DENSEKP_CUDA_ERP_REMAP_READY=PASS`
