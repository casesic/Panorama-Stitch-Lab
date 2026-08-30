# CPW backward-map cache

## 数学边界

CPW backward map 不是仅由 camera calibration 决定。当前实现为：

```text
backward_map = f(src_points, dst_points, crop, k, power, output_size)
```

其中 `src_points/dst_points` 来自每帧 C++ LM 的 optimized DenseKP。不同帧的
点对变化时，map 也会变化。因此只按 calibration 跨帧复用会改变 CPW 数学，
是不安全的。

本实现使用完整 key：既包含 calibration/static geometry，也包含实际控制点
摘要。同一帧或完全相同点对的重复运行可以命中；不同帧只有在所有输入完全
一致时才会命中。

## Cache 内容

每个 cache 文件为一个未压缩 NPZ，包含：

- `map_x`：float32 backward remap x；
- `map_y`：float32 backward remap y；
- `specification`：完整 cache key/static geometry；
- `stats`：CPW map 构建统计。

Cache key 包含：

- cam2/cam4 Double-Sphere 参数和 camera group；
- ERP_ROTATE 矩阵与 ERP 尺寸；
- source/target image size；
- output crop size 与 crop bbox；
- warp direction、source/target camera；
- `k`、`power`；
- src/dst point count 与 float32 SHA-256。

## Runtime

实现文件：

```text
src/lib_pano/composite/cpw_warp_cache.py
```

`debug078_densekp_fitted_cpw_composite.py` 新增：

```text
--cpw_warp_cache_dir <path>
--cpw_warp_cache_mode off|read_write|refresh|read_only
```

默认模式为 `read_write`，默认 cache 目录为：

```text
<debug078 output>/.cpw_warp_cache
```

首次 miss 调用原 `generate_cpw_backward_map()` 并原样保存 float32 map；命中后
直接读取 `map_x/map_y`，后续 interpolation 仍调用原 OpenCV `cv2.remap`
（底层 `cv::remap`）。blending 和输出格式未修改。

## 513 帧结果

| 项目 | Cache miss | Cache hit |
|---|---:|---:|
| warp-map build | 2722.29 ms | 0 ms |
| warp-map cache load | 0 ms | 4.96 ms |
| CPW total | 4202.84 ms | 1487.66 ms |

Cache 文件约 `5 MB`。命中后 CPW 剩余主要成本为 point/image/valid-mask
preparation，约 `1.31 s`。

## Correctness

- regenerated map 与 cached map：bitwise identical；
- 两个 map 的 OpenCV warp result：bitwise identical；
- cache miss/hit CPW JPEG：bitwise identical；
- cache miss/hit optimized DenseKP JSON：bitwise identical；
- debug078 cache miss/hit：PASS。

```text
CPW_WARP_MAP_CACHE_READY=PASS
```
